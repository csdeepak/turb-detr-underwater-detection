"""SimAM Attention Heatmap Visualization — Turb-DETR.

Visualizes feature map activation before and after SimAM for a set of
sample images (optionally with turbidity applied).  Helps demonstrate
whether SimAM amplifies object-relevant activations.

NOTE — What the heatmap actually shows
---------------------------------------
SimAM reweights features: salient neurons (low surroundings energy)
receive *higher* attention weights; uniform/noisy regions receive lower
weights.  The heatmap does NOT guarantee turbidity is removed — it shows
*where* the model attends.  Interpret accordingly.

Usage
-----
    python evaluation/attention_viz.py \\
        --weights outputs/checkpoints/turb_detr.pt \\
        --images data/trash_icra19/images/test \\
        --num-images 6 \\
        --turbidity 0.6 \\
        --output-dir outputs/visualizations/attention_maps

Outputs
-------
    outputs/visualizations/attention_maps/
        attention_<stem>_clean.png      (clean image heatmaps)
        attention_<stem>_turbid.png     (turbid image heatmaps)
        attention_grid.png              (combined overview grid)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from ultralytics import RTDETR

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from augmentation.turbidity_aug import apply_turbidity
from models.simam import SimAM


# ─────────────────────────────────────────────────────────────
# Feature capture hook
# ─────────────────────────────────────────────────────────────
class FeatureCapture:
    """Captures the output of a module on each forward pass."""
    def __init__(self) -> None:
        self.features: list[torch.Tensor] | torch.Tensor | None = None
        self._handle = None

    def register(self, module: nn.Module) -> "FeatureCapture":
        self._handle = module.register_forward_hook(self._hook)
        return self

    def _hook(self, module, input, output):
        if isinstance(output, (list, tuple)):
            self.features = [f.detach().cpu() for f in output]
        else:
            self.features = output.detach().cpu()

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


# ─────────────────────────────────────────────────────────────
# Feature map → heatmap (channel-averaged, normalized)
# ─────────────────────────────────────────────────────────────
def featuremap_to_heatmap(feat: torch.Tensor, target_size: tuple[int, int]) -> np.ndarray:
    """Convert a (C, H, W) feature tensor to a normalised uint8 heatmap."""
    # Average across channels
    heatmap = feat.mean(dim=0).numpy()           # (H, W)
    # Normalize to [0, 255]
    heatmap -= heatmap.min()
    denom = heatmap.max() + 1e-9
    heatmap = (heatmap / denom * 255).astype(np.uint8)
    # Upscale to image resolution
    h, w = target_size
    heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    return heatmap


def overlay_heatmap(image_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Overlay a heatmap (colormap JET) on an RGB image."""
    colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image_rgb, 1.0 - alpha, colored, alpha, 0)


# ─────────────────────────────────────────────────────────────
# Load the backbone module from an RTDETR model
# ─────────────────────────────────────────────────────────────
def find_backbone_module(inner_model: nn.Module) -> nn.Module | None:
    if hasattr(inner_model, "model"):
        seq = inner_model.model
        for name, mod in seq.named_modules():
            lower = name.lower()
            if "backbone" in lower or "hgblock" in lower:
                return mod
        if len(seq) > 10:
            return seq[9]
    return inner_model


# ─────────────────────────────────────────────────────────────
# Single-image visualization
# ─────────────────────────────────────────────────────────────
def visualize_image(
    image_rgb: np.ndarray,
    model_with: RTDETR,      # SimAM enabled
    model_without: RTDETR,   # SimAM disabled
    capture_with: FeatureCapture,
    capture_without: FeatureCapture,
    output_path: Path,
    title: str = "",
) -> None:
    h, w = image_rgb.shape[:2]
    imgsz = 640

    def run(model: RTDETR, cap: FeatureCapture):
        cap.features = None
        model.predict(image_rgb, imgsz=imgsz, conf=0.1, verbose=False)

    run(model_without, capture_without)
    feats_before_raw = capture_without.features

    run(model_with, capture_with)
    feats_after_raw = capture_with.features

    # Pick the largest-scale feature map (first element in the list)
    def pick_feat(raw):
        if isinstance(raw, list):
            return raw[0]
        return raw

    if feats_before_raw is None or feats_after_raw is None:
        print(f"  Warning: feature capture failed for {output_path.name}")
        return

    feat_before = pick_feat(feats_before_raw)
    feat_after  = pick_feat(feats_after_raw)

    # Squeeze batch dimension if present
    if feat_before.dim() == 4:
        feat_before = feat_before[0]
    if feat_after.dim() == 4:
        feat_after = feat_after[0]

    hmap_before = featuremap_to_heatmap(feat_before, (h, w))
    hmap_after  = featuremap_to_heatmap(feat_after,  (h, w))

    overlay_before = overlay_heatmap(image_rgb, hmap_before)
    overlay_after  = overlay_heatmap(image_rgb, hmap_after)

    # ── Plot ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image_rgb)
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(overlay_before)
    axes[1].set_title("Backbone Features (before SimAM)", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(overlay_after)
    axes[2].set_title("Backbone Features (after SimAM)", fontsize=12)
    axes[2].axis("off")

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def run_attention_viz(
    weights: str,
    image_dir: str | Path,
    output_dir: str | Path,
    num_images: int = 6,
    turbidity_level: float = 0.6,
) -> None:
    image_dir  = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(
        list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    )[:num_images]
    if not img_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    print(f"Loading model: {weights}")
    model_with    = RTDETR(weights)
    model_without = RTDETR(weights)

    # Inject SimAM into model_with
    inner_with    = model_with.model
    inner_without = model_without.model
    # Detect device from model weights
    device = next(model_with.model.parameters()).device
    simam = SimAM().to(device)
    backbone_with    = find_backbone_module(inner_with)
    backbone_without = find_backbone_module(inner_without)

    # Add SimAM hook to model_with
    def simam_hook(module, input, output):
        if isinstance(output, (list, tuple)):
            return type(output)(simam(f) for f in output)
        return simam(output)

    backbone_with.register_forward_hook(simam_hook)

    # Capture hooks
    cap_without = FeatureCapture().register(backbone_without)
    cap_with    = FeatureCapture().register(backbone_with)

    overview_rows: list[np.ndarray] = []

    for img_path in img_paths:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  Skipping (unreadable): {img_path.name}")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # ─── Clean image ────────────────────────────────────
        out_clean = output_dir / f"attention_{img_path.stem}_clean.png"
        print(f"  Clean image: {img_path.name}")
        visualize_image(
            img_rgb, model_with, model_without,
            cap_with, cap_without,
            out_clean,
            title=f"Clean — {img_path.name}",
        )

        # ─── Turbid image ────────────────────────────────────
        # apply_turbidity expects BGR (OpenCV convention)
        turbid_bgr = apply_turbidity(img_bgr, level=turbidity_level)
        turbid_rgb = cv2.cvtColor(turbid_bgr, cv2.COLOR_BGR2RGB)
        out_turbid = output_dir / f"attention_{img_path.stem}_turbid.png"
        print(f"  Turbid (level={turbidity_level}) image: {img_path.name}")
        visualize_image(
            turbid_rgb, model_with, model_without,
            cap_with, cap_without,
            out_turbid,
            title=f"Turbid (level={turbidity_level}) — {img_path.name}",
        )

    cap_with.remove()
    cap_without.remove()

    print(f"\nAttention maps saved to {output_dir}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize SimAM attention before/after on backbone features"
    )
    parser.add_argument("--weights",     required=True, help="Path to .pt checkpoint")
    parser.add_argument("--images",      required=True, help="Directory of test images")
    parser.add_argument("--num-images",  type=int,  default=6,   help="Number of images to process")
    parser.add_argument("--turbidity",   type=float, default=0.6, help="Turbidity level for turbid comparison (0-1)")
    parser.add_argument("--output-dir",  default="outputs/visualizations/attention_maps")
    args = parser.parse_args()

    run_attention_viz(
        weights=args.weights,
        image_dir=args.images,
        output_dir=args.output_dir,
        num_images=args.num_images,
        turbidity_level=args.turbidity,
    )
