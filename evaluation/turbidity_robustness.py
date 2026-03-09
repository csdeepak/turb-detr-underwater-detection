"""Turbidity Robustness Evaluation — Turb-DETR.

Tests how model mAP degrades across increasing turbidity levels.

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
import tempfile
import shutil
from pathlib import Path
from typing import NamedTuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics import RTDETR

# Add project root to path so local imports work
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from augmentation.turbidity_aug import apply_turbidity


# ─────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────
class Detection(NamedTuple):
    cls: int
    x1: float; y1: float; x2: float; y2: float
    conf: float


class GroundTruth(NamedTuple):
    cls: int
    x1: float; y1: float; x2: float; y2: float


# ─────────────────────────────────────────────────────────────
# Load YOLO-format ground truth labels
# ─────────────────────────────────────────────────────────────
def load_yolo_labels(label_path: Path, img_w: int, img_h: int) -> list[GroundTruth]:
    """Read a YOLO .txt label file and return absolute xyxy boxes."""
    gts: list[GroundTruth] = []
    if not label_path.exists():
        return gts
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])
            x1 = (cx - bw / 2) * img_w
            y1 = (cy - bh / 2) * img_h
            x2 = (cx + bw / 2) * img_w
            y2 = (cy + bh / 2) * img_h
            gts.append(GroundTruth(cls, x1, y1, x2, y2))
    return gts


# ─────────────────────────────────────────────────────────────
# IoU
# ─────────────────────────────────────────────────────────────
def iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / (union + 1e-9)


# ─────────────────────────────────────────────────────────────
# Compute AP@0.5 for a single class from detections
# ─────────────────────────────────────────────────────────────
def compute_ap50(
    dets: list[Detection],
    gts: list[GroundTruth],
    cls_id: int,
    iou_thresh: float = 0.5,
) -> float:
    class_dets = [d for d in dets if d.cls == cls_id]
    class_gts  = [g for g in gts  if g.cls == cls_id]
    if not class_gts:
        return float("nan")
    if not class_dets:
        return 0.0

    class_dets.sort(key=lambda d: d.conf, reverse=True)
    matched = [False] * len(class_gts)
    tp = []
    fp = []
    for det in class_dets:
        best_iou = 0.0
        best_idx = -1
        for i, gt in enumerate(class_gts):
            if matched[i]:
                continue
            v = iou((det.x1, det.y1, det.x2, det.y2),
                    (gt.x1,  gt.y1,  gt.x2,  gt.y2))
            if v > best_iou:
                best_iou = v
                best_idx = i
        if best_iou >= iou_thresh and best_idx >= 0:
            matched[best_idx] = True
            tp.append(1); fp.append(0)
        else:
            tp.append(0); fp.append(1)

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall    = tp_cum / (len(class_gts) + 1e-9)
    precision = tp_cum / (tp_cum + fp_cum + 1e-9)

    recall    = np.concatenate([[0.0], recall,    [1.0]])
    precision = np.concatenate([[1.0], precision, [0.0]])
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    idx = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[idx + 1] - recall[idx]) * precision[idx + 1]))


# ─────────────────────────────────────────────────────────────
# Evaluate a model at a given turbidity level
# ─────────────────────────────────────────────────────────────
def evaluate_at_level(
    model: RTDETR,
    img_paths: list[Path],
    label_dir: Path,
    turbidity_level: float,
    num_classes: int = 5,
    conf_thresh: float = 0.25,
) -> float:
    """Return mean AP@0.5 across all classes at a given turbidity level."""
    all_dets: list[Detection] = []
    all_gts:  list[GroundTruth] = []

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        aug_paths: list[Path] = []

        for img_path in img_paths:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            if turbidity_level > 0.0:
                # apply_turbidity expects BGR (OpenCV convention) — img is already BGR
                aug_bgr = apply_turbidity(img, level=turbidity_level)
            else:
                aug_bgr = img
            aug_path = tmp_dir / img_path.name
            cv2.imwrite(str(aug_path), aug_bgr)
            aug_paths.append((aug_path, img_path, img.shape[1], img.shape[0]))

        # Run batch inference
        paths_only = [str(p[0]) for p in aug_paths]
        results = model.predict(paths_only, conf=conf_thresh, verbose=False, stream=False)

        for result, (aug_path, orig_path, w, h) in zip(results, aug_paths):
            # Collect detections
            if result.boxes is not None and len(result.boxes):
                boxes = result.boxes.xyxy.cpu().numpy()
                clss  = result.boxes.cls.cpu().numpy().astype(int)
                confs = result.boxes.conf.cpu().numpy()
                for box, cls, conf in zip(boxes, clss, confs):
                    all_dets.append(Detection(cls, *box, conf))

            # Collect ground truths
            label_path = label_dir / (orig_path.stem + ".txt")
            all_gts.extend(load_yolo_labels(label_path, w, h))

    aps = []
    for cls_id in range(num_classes):
        ap = compute_ap50(all_dets, all_gts, cls_id)
        if not np.isnan(ap):
            aps.append(ap)
    return float(np.mean(aps)) if aps else 0.0


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
                model, img_paths, label_dir, level, num_classes
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
