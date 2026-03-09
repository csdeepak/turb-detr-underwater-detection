"""Failure Case Analysis — Turb-DETR.

Finds and visualizes model failure cases on a test set:

  - False Negatives (missed detections): GT box with no matching prediction
  - False Positives (ghost detections):  Prediction with no matching GT
  - Low-confidence correct detections:   TP but conf < threshold

Results are saved as annotated images, grouped by failure type.

Usage
-----
    python scripts/failure_analysis.py \\
        --weights outputs/checkpoints/turb_detr.pt \\
        --data-dir data/trash_icra19/images/test \\
        --label-dir data/trash_icra19/labels/test \\
        --output-dir outputs/visualizations/failures \\
        --max-failures 20

Outputs
-------
    outputs/visualizations/failures/
        false_negatives/   (missed GT boxes highlighted)
        false_positives/   (ghost predictions highlighted)
        low_confidence/    (correct but uncertain detections)
        summary.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import RTDETR

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────────────────────
# Class names
# ─────────────────────────────────────────────────────────────
CLASS_NAMES = {0: "plastic", 1: "bottle", 2: "can", 3: "bag", 4: "net"}


# ─────────────────────────────────────────────────────────────
# Parse YOLO labels
# ─────────────────────────────────────────────────────────────
def load_labels(label_path: Path, img_w: int, img_h: int):
    gts = []
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
            gts.append({"cls": cls, "box": [x1, y1, x2, y2]})
    return gts


# ─────────────────────────────────────────────────────────────
# IoU
# ─────────────────────────────────────────────────────────────
def iou(a, b) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


# ─────────────────────────────────────────────────────────────
# Match predictions to ground truths
# ─────────────────────────────────────────────────────────────
@dataclass
class FailureRecord:
    image_path: str
    failure_type: str          # "false_negative" | "false_positive" | "low_confidence"
    cls_name: str
    box: list                  # [x1, y1, x2, y2]
    conf: float | None


def match_image(
    img_path: Path,
    preds,
    gts: list,
    iou_thresh: float = 0.5,
    low_conf_thresh: float = 0.4,
) -> list[FailureRecord]:
    failures: list[FailureRecord] = []

    if not gts and (preds is None or len(preds) == 0):
        return failures

    # Parse predictions
    pred_boxes = []
    if preds is not None and preds.boxes is not None and len(preds.boxes):
        for box, cls, conf in zip(
            preds.boxes.xyxy.cpu().numpy(),
            preds.boxes.cls.cpu().numpy().astype(int),
            preds.boxes.conf.cpu().numpy(),
        ):
            pred_boxes.append({"cls": cls, "box": box.tolist(), "conf": float(conf)})

    gt_matched   = [False] * len(gts)
    pred_matched = [False] * len(pred_boxes)

    # Sort predictions by confidence (descending) for greedy match
    pred_order = sorted(range(len(pred_boxes)), key=lambda i: -pred_boxes[i]["conf"])

    for pi in pred_order:
        pred = pred_boxes[pi]
        best_iou = 0.0
        best_gi  = -1
        for gi, gt in enumerate(gts):
            if gt_matched[gi]:
                continue
            if gt["cls"] != pred["cls"]:
                continue
            v = iou(pred["box"], gt["box"])
            if v > best_iou:
                best_iou = v
                best_gi  = gi

        if best_iou >= iou_thresh and best_gi >= 0:
            gt_matched[best_gi]   = True
            pred_matched[pi]      = True
            # Low-confidence TP
            if pred["conf"] < low_conf_thresh:
                failures.append(FailureRecord(
                    image_path=str(img_path),
                    failure_type="low_confidence",
                    cls_name=CLASS_NAMES.get(pred["cls"], str(pred["cls"])),
                    box=pred["box"],
                    conf=pred["conf"],
                ))

    # Unmatched GTs → false negatives
    for gi, gt in enumerate(gts):
        if not gt_matched[gi]:
            failures.append(FailureRecord(
                image_path=str(img_path),
                failure_type="false_negative",
                cls_name=CLASS_NAMES.get(gt["cls"], str(gt["cls"])),
                box=gt["box"],
                conf=None,
            ))

    # Unmatched preds → false positives
    for pi, pred in enumerate(pred_boxes):
        if not pred_matched[pi]:
            failures.append(FailureRecord(
                image_path=str(img_path),
                failure_type="false_positive",
                cls_name=CLASS_NAMES.get(pred["cls"], str(pred["cls"])),
                box=pred["box"],
                conf=pred["conf"],
            ))

    return failures


# ─────────────────────────────────────────────────────────────
# Draw annotated failure image
# ─────────────────────────────────────────────────────────────
_COLORS = {
    "false_negative": (220, 50,  50),    # red  — missed GT
    "false_positive": (50,  50, 220),    # blue — ghost pred
    "low_confidence": (220, 160,  0),    # amber — uncertain TP
}

def draw_failure(
    img_bgr: np.ndarray,
    failures: list[FailureRecord],
) -> np.ndarray:
    img = img_bgr.copy()
    for f in failures:
        x1, y1, x2, y2 = [int(v) for v in f.box]
        color = _COLORS.get(f.failure_type, (128, 128, 128))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label_parts = [f.cls_name, f.failure_type.replace("_", " ")]
        if f.conf is not None:
            label_parts.append(f"{f.conf:.2f}")
        label = " | ".join(label_parts)
        cv2.putText(img, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return img


def create_failure_figure(
    img_bgr: np.ndarray,
    failures: list[FailureRecord],
    title: str,
    output_path: Path,
) -> None:
    annotated_bgr = draw_failure(img_bgr, failures)
    img_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(img_rgb)
    ax.set_title(title, fontsize=11)
    ax.axis("off")

    # Legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=[c / 255.0 for c in rgb])
        for rgb in _COLORS.values()
    ]
    ax.legend(handles, [k.replace("_", " ") for k in _COLORS],
              loc="upper right", fontsize=9, framealpha=0.7)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=120, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def run_failure_analysis(
    weights: str,
    data_dir: str | Path,
    label_dir: str | Path,
    output_dir: str | Path,
    max_failures: int = 20,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.5,
    low_conf_thresh: float = 0.40,
) -> None:
    data_dir   = Path(data_dir)
    label_dir  = Path(label_dir)
    output_dir = Path(output_dir)

    for subfolder in ["false_negatives", "false_positives", "low_confidence"]:
        (output_dir / subfolder).mkdir(parents=True, exist_ok=True)

    img_paths = sorted(
        list(data_dir.glob("*.jpg")) + list(data_dir.glob("*.png"))
    )
    if not img_paths:
        raise FileNotFoundError(f"No images found in {data_dir}")

    print(f"Loading model: {weights}")
    model = RTDETR(weights)

    all_failures: list[dict] = []
    type_counts  = {"false_negative": 0, "false_positive": 0, "low_confidence": 0}
    saved_counts = {"false_negative": 0, "false_positive": 0, "low_confidence": 0}

    for img_path in img_paths:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        h, w = img_bgr.shape[:2]

        label_path = label_dir / (img_path.stem + ".txt")
        gts        = load_labels(label_path, w, h)

        results = model.predict(str(img_path), conf=conf_thresh, verbose=False)
        preds   = results[0] if results else None

        failures = match_image(
            img_path, preds, gts,
            iou_thresh=iou_thresh,
            low_conf_thresh=low_conf_thresh,
        )
        if not failures:
            continue

        # Accumulate
        for f in failures:
            type_counts[f.failure_type] = type_counts.get(f.failure_type, 0) + 1
            all_failures.append(asdict(f))

        # Group by type and save up to max_failures per type
        by_type: dict[str, list[FailureRecord]] = {}
        for f in failures:
            by_type.setdefault(f.failure_type, []).append(f)

        for ftype, flist in by_type.items():
            if saved_counts.get(ftype, 0) >= max_failures:
                continue
            subfolder = {
                "false_negative": "false_negatives",
                "false_positive": "false_positives",
                "low_confidence": "low_confidence",
            }[ftype]
            out_path = output_dir / subfolder / f"{img_path.stem}.png"
            title    = f"{img_path.name} — {ftype.replace('_', ' ')} ({len(flist)} cases)"
            try:
                create_failure_figure(img_bgr, flist, title, out_path)
                saved_counts[ftype] = saved_counts.get(ftype, 0) + 1
            except Exception as e:
                print(f"  Warning: could not save {out_path.name}: {e}")

    # ── Summary ─────────────────────────────────────────────
    summary = {
        "total_failures": len(all_failures),
        "by_type":        type_counts,
        "failures":       all_failures,
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n── Failure Analysis Summary ──────────────────────────")
    print(f"  Total failures      : {len(all_failures)}")
    for ftype, count in type_counts.items():
        print(f"  {ftype:<22}: {count}")
    print(f"\nResults saved to {output_dir}")
    print(f"Summary JSON: {summary_path}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Identify and visualize failure cases")
    parser.add_argument("--weights",      required=True, help="Path to .pt checkpoint")
    parser.add_argument("--data-dir",     required=True, help="Test images directory")
    parser.add_argument("--label-dir",    required=True, help="YOLO label files directory")
    parser.add_argument("--output-dir",   default="outputs/visualizations/failures")
    parser.add_argument("--max-failures", type=int, default=20,
                        help="Max saved images per failure type")
    parser.add_argument("--conf",         type=float, default=0.25,  help="Detection confidence threshold")
    parser.add_argument("--iou",          type=float, default=0.50,  help="IoU threshold for matching")
    parser.add_argument("--low-conf",     type=float, default=0.40,  help="Low-confidence TP threshold")
    args = parser.parse_args()

    run_failure_analysis(
        weights=args.weights,
        data_dir=args.data_dir,
        label_dir=args.label_dir,
        output_dir=args.output_dir,
        max_failures=args.max_failures,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        low_conf_thresh=args.low_conf,
    )
