"""Turbidity Robustness Evaluation — Turb-DETR.

Tests how model mAP degrades across increasing turbidity levels.

Metrics use Ultralytics .val() (COCO-style AP@0.5) so the robustness curve
is on the same scale as the numbers in the main evaluation table.

Usage
-----
    python evaluation/turbidity_robustness.py \\
        --weights outputs/checkpoints/turb_detr.pt \\
        --baseline-weights outputs/checkpoints/baseline_rtdetr.pt \\
        --data-dir data/trash_icra19/images/test \\
        --label-dir data/trash_icra19/labels/test \\
        --output-dir outputs/visualizations

Outputs
-------
    outputs/visualizations/turbidity_robustness.png   (mAP vs turbidity curve)
    outputs/visualizations/turbidity_robustness.json  (raw numbers)
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from ultralytics import RTDETR

# Add project root to path so local imports work
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from augmentation.turbidity_aug import apply_turbidity

# Default class names for Trash-ICRA19
_DEFAULT_CLASS_NAMES = {0: "plastic", 1: "bottle", 2: "can", 3: "bag", 4: "net"}

# ─────────────────────────────────────────────────────────────
# Evaluate a model at a given turbidity level using Ultralytics .val()
# ─────────────────────────────────────────────────────────────
def evaluate_at_level(
    model: RTDETR,
    img_paths: list[Path],
    label_dir: Path,
    turbidity_level: float,
    num_classes: int = 5,
    conf_thresh: float = 0.001,
    class_names: dict[int, str] | None = None,
) -> float:
    """Return mAP@0.5 at a given turbidity level using Ultralytics .val().

    Writes augmented images + their original labels into a temporary YOLO
    dataset, fires ``model.val()`` on it, and returns ``metrics.box.map50``.
    This uses the same COCO-style evaluation as evaluate.py so the numbers
    are directly comparable to the main evaluation table.
    """
    if class_names is None:
        class_names = _DEFAULT_CLASS_NAMES

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        imgs_dst = tmp_dir / "images" / "test"
        lbls_dst = tmp_dir / "labels" / "test"
        imgs_dst.mkdir(parents=True)
        lbls_dst.mkdir(parents=True)

        for img_path in img_paths:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            aug = apply_turbidity(img, level=turbidity_level) if turbidity_level > 0.0 else img
            cv2.imwrite(str(imgs_dst / img_path.name), aug)
            lbl = label_dir / (img_path.stem + ".txt")
            if lbl.exists():
                shutil.copy2(lbl, lbls_dst / lbl.name)

        # Write a minimal YOLO data YAML
        data_yaml_path = tmp_dir / "data.yaml"
        data_yaml_path.write_text(
            yaml.dump({
                "path": str(tmp_dir),
                "test": "images/test",
                "nc": num_classes,
                "names": class_names,
            })
        )

        metrics = model.val(
            data=str(data_yaml_path),
            split="test",
            conf=conf_thresh,
            verbose=False,
        )
        return float(metrics.box.map50)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def run_robustness_eval(
    weights_map: dict[str, str],       # {"Turb-DETR": "path/to.pt", ...}
    data_dir: str | Path,
    label_dir: str | Path,
    output_dir: str | Path,
    turbidity_levels: list[float] | None = None,
    max_images: int = 200,
    num_classes: int = 5,
    class_names: dict[int, str] | None = None,
) -> None:
    data_dir   = Path(data_dir)
    label_dir  = Path(label_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if turbidity_levels is None:
        turbidity_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    img_paths = sorted(
        list(data_dir.glob("*.jpg")) + list(data_dir.glob("*.png"))
    )[:max_images]

    if not img_paths:
        raise FileNotFoundError(f"No images found in {data_dir}")

    print(f"Evaluating on {len(img_paths)} images across "
          f"{len(turbidity_levels)} turbidity levels\n")

    results: dict[str, list[float]] = {}

    for model_name, weights_path in weights_map.items():
        print(f"Loading model: {model_name}")
        model = RTDETR(weights_path)
        model_results: list[float] = []

        for level in turbidity_levels:
            label = f"{model_name} | turbidity={level:.1f}"
            print(f"  Evaluating {label} ...", end=" ", flush=True)
            map50 = evaluate_at_level(
                model, img_paths, label_dir, level, num_classes,
                class_names=class_names,
            )
            model_results.append(map50)
            print(f"mAP@0.5 = {map50:.4f}")

        results[model_name] = model_results
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Save raw results ──────────────────────────────────────
    output_json = {
        "turbidity_levels": turbidity_levels,
        "results": results,
    }
    json_path = output_dir / "turbidity_robustness.json"
    with open(json_path, "w") as f:
        json.dump(output_json, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # ── Plot ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    colors  = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    markers = ["o", "s", "^", "D"]

    for i, (model_name, map_values) in enumerate(results.items()):
        ax.plot(
            turbidity_levels, map_values,
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            linewidth=2, markersize=7,
            label=model_name,
        )

    ax.set_xlabel("Turbidity Level", fontsize=13)
    ax.set_ylabel("mAP@0.5", fontsize=13)
    ax.set_title("Turbidity Robustness: mAP vs Turbidity Level", fontsize=14)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    plot_path = output_dir / "turbidity_robustness.png"
    fig.tight_layout()
    fig.savefig(str(plot_path), dpi=150)
    plt.close(fig)
    print(f"Plot saved to {plot_path}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate mAP vs turbidity level for multiple models"
    )
    parser.add_argument(
        "--weights", nargs="+", required=True,
        help="Paths to model .pt checkpoints (space-separated)"
    )
    parser.add_argument(
        "--names", nargs="+",
        help="Model display names matching --weights order. "
             "Defaults to checkpoint filenames."
    )
    parser.add_argument(
        "--data-dir", required=True,
        help="Directory with test images"
    )
    parser.add_argument(
        "--label-dir", required=True,
        help="Directory with YOLO-format .txt label files"
    )
    parser.add_argument(
        "--output-dir", default="outputs/visualizations",
        help="Where to save plot and JSON"
    )
    parser.add_argument(
        "--max-images", type=int, default=200,
        help="Cap on number of test images (for speed)"
    )
    parser.add_argument(
        "--num-classes", type=int, default=5,
        help="Number of object classes"
    )
    args = parser.parse_args()

    names = args.names if args.names else [Path(w).stem for w in args.weights]
    if len(names) != len(args.weights):
        parser.error("--names count must match --weights count")

    weights_map = dict(zip(names, args.weights))
    run_robustness_eval(
        weights_map=weights_map,
        data_dir=args.data_dir,
        label_dir=args.label_dir,
        output_dir=args.output_dir,
        max_images=args.max_images,
        num_classes=args.num_classes,
    )
