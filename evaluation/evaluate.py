"""Evaluate trained detection models on underwater debris datasets.

Supported datasets
──────────────────
  • Trash-ICRA19   — standard underwater trash benchmark (5 classes)
  • TrashCan       — larger underwater/above-water trash dataset
  • RUIE           — Real-world Underwater Image Enhancement (turbidity)

Metrics
───────
  mAP@0.5, mAP@0.5:0.95, precision, recall, inference FPS

Outputs
───────
  • Console summary table
  • JSON results file   → ``results/<run_name>_results.json``
  • CSV experiment log  → ``results/<run_name>_log.csv``

Usage
─────
  python -m evaluation.evaluate \
      --weights runs/detect/train/weights/best.pt \
      --datasets trash_icra19 trashcan ruie \
      --imgsz 640 \
      --device 0
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import torch
from ultralytics import RTDETR, YOLO


# ─────────────────────────────────────────────────────────────
# Dataset registry
# ─────────────────────────────────────────────────────────────
DATASET_CONFIGS: dict[str, dict[str, Any]] = {
    "trash_icra19": {
        "data_yaml": "configs/trash_icra19.yaml",
        "description": "Trash-ICRA19 underwater trash benchmark",
        "split": "test",
    },
    "trashcan": {
        "data_yaml": "configs/trashcan.yaml",
        "description": "TrashCan underwater / above-water debris",
        "split": "test",
    },
    "ruie": {
        "data_yaml": "configs/ruie.yaml",
        "description": "RUIE real-world turbidity images",
        "split": "test",
    },
}


# ─────────────────────────────────────────────────────────────
# Result data class
# ─────────────────────────────────────────────────────────────
@dataclass
class EvalResult:
    """Container for single-dataset evaluation results."""

    dataset: str = ""
    model_path: str = ""
    map50: float = 0.0
    map50_95: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    fps: float = 0.0
    num_images: int = 0
    imgsz: int = 640
    device: str = "cpu"
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────
def load_model(weights: str | Path) -> RTDETR | YOLO:
    """Load an Ultralytics model from a checkpoint.

    Automatically detects whether the checkpoint is RT-DETR or YOLO
    based on the model metadata.
    """
    weights = Path(weights)
    if not weights.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")

    # Ultralytics YOLO constructor handles both YOLO and RT-DETR .pt
    model = YOLO(str(weights))
    arch = getattr(model, "task", "detect")
    print(f"[evaluate] Loaded {weights.name}  (task={arch})")
    return model


# ─────────────────────────────────────────────────────────────
# FPS measurement
# ─────────────────────────────────────────────────────────────
def measure_fps(
    model: YOLO | RTDETR,
    imgsz: int = 640,
    warmup: int = 10,
    iterations: int = 100,
    device: str = "cpu",
) -> float:
    """Measure model inference FPS with a dummy input tensor.

    Parameters
    ----------
    model : YOLO | RTDETR
        Loaded Ultralytics model.
    imgsz : int
        Input image size.
    warmup : int
        Number of warm-up iterations (not timed).
    iterations : int
        Number of timed iterations.
    device : str
        ``"cpu"`` or ``"0"`` / ``"cuda:0"``.

    Returns
    -------
    float
        Frames per second.
    """
    dev = torch.device("cuda:0" if device not in ("cpu",) else "cpu")
    dummy = torch.randn(1, 3, imgsz, imgsz, device=dev)

    # Warm-up
    for _ in range(warmup):
        model.predict(source=dummy, verbose=False, device=device)

    # Timed run
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iterations):
        model.predict(source=dummy, verbose=False, device=device)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return iterations / elapsed


# ─────────────────────────────────────────────────────────────
# Core evaluator
# ─────────────────────────────────────────────────────────────
def evaluate_dataset(
    model: YOLO | RTDETR,
    dataset_key: str,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "cpu",
    conf: float = 0.001,
    iou: float = 0.6,
) -> EvalResult:
    """Evaluate a model on a single dataset.

    Parameters
    ----------
    model : YOLO | RTDETR
        Loaded Ultralytics model.
    dataset_key : str
        Key into ``DATASET_CONFIGS``.
    imgsz : int
        Inference image size.
    batch : int
        Batch size for validation.
    device : str
        Device string (``"cpu"``, ``"0"``, etc.).
    conf : float
        Confidence threshold for NMS.
    iou : float
        IoU threshold for NMS.

    Returns
    -------
    EvalResult
        Populated evaluation results.
    """
    cfg = DATASET_CONFIGS.get(dataset_key)
    if cfg is None:
        raise ValueError(
            f"Unknown dataset '{dataset_key}'. "
            f"Available: {list(DATASET_CONFIGS)}"
        )

    data_yaml = cfg["data_yaml"]
    split = cfg.get("split", "test")

    print(f"\n{'─' * 60}")
    print(f"  Evaluating on: {cfg['description']}")
    print(f"  Data YAML    : {data_yaml}")
    print(f"  Split        : {split}")
    print(f"{'─' * 60}")

    # ── Run Ultralytics validation ───────────────────────────
    metrics = model.val(
        data=data_yaml,
        split=split,
        imgsz=imgsz,
        batch=batch,
        device=device,
        conf=conf,
        iou=iou,
        verbose=True,
    )

    # ── Extract metrics ──────────────────────────────────────
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)

    # Per-class breakdown
    per_class: dict[str, dict[str, float]] = {}
    class_names = metrics.names if hasattr(metrics, "names") else {}
    if hasattr(metrics.box, "ap50"):
        ap50_arr = metrics.box.ap50()  # ndarray (num_classes,)
        for i, ap in enumerate(ap50_arr):
            name = class_names.get(i, f"class_{i}")
            per_class[name] = {"AP50": float(ap)}

    # ── Measure FPS ──────────────────────────────────────────
    fps = measure_fps(model, imgsz=imgsz, device=device)

    return EvalResult(
        dataset=dataset_key,
        model_path=str(model.ckpt_path) if hasattr(model, "ckpt_path") else "",
        map50=map50,
        map50_95=map50_95,
        precision=precision,
        recall=recall,
        fps=fps,
        num_images=metrics.box.n if hasattr(metrics.box, "n") else 0,
        imgsz=imgsz,
        device=device,
        per_class=per_class,
    )


# ─────────────────────────────────────────────────────────────
# Console table
# ─────────────────────────────────────────────────────────────
def print_summary_table(results: list[EvalResult]) -> None:
    """Print a formatted console summary table."""
    header = (
        f"{'Dataset':<16} {'mAP@0.5':>8} {'mAP@.5:.95':>11} "
        f"{'Prec':>7} {'Recall':>7} {'FPS':>7}"
    )
    sep = "─" * len(header)

    print(f"\n{'═' * len(header)}")
    print("  EVALUATION SUMMARY")
    print(f"{'═' * len(header)}")
    print(header)
    print(sep)

    for r in results:
        print(
            f"{r.dataset:<16} {r.map50:>8.4f} {r.map50_95:>11.4f} "
            f"{r.precision:>7.4f} {r.recall:>7.4f} {r.fps:>7.1f}"
        )

    print(sep)

    # Per-class details
    for r in results:
        if r.per_class:
            print(f"\n  Per-class AP@0.5 — {r.dataset}")
            for cls_name, cls_metrics in r.per_class.items():
                print(f"    {cls_name:<14} AP50={cls_metrics.get('AP50', 0):.4f}")


# ─────────────────────────────────────────────────────────────
# Output writers
# ─────────────────────────────────────────────────────────────
def save_json(results: list[EvalResult], path: Path) -> None:
    """Save evaluation results to a JSON file."""
    data = [asdict(r) for r in results]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    print(f"[evaluate] JSON saved → {path}")


def save_csv(results: list[EvalResult], path: Path) -> None:
    """Append evaluation results to a CSV experiment log."""
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()

    fieldnames = [
        "dataset", "model_path", "map50", "map50_95",
        "precision", "recall", "fps", "num_images", "imgsz", "device",
    ]

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for r in results:
            row = {k: getattr(r, k) for k in fieldnames}
            writer.writerow(row)

    print(f"[evaluate] CSV log  → {path}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate detection models on underwater debris datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file).",
    )
    p.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["trash_icra19"],
        choices=list(DATASET_CONFIGS),
        help="Datasets to evaluate on (default: trash_icra19).",
    )
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    p.add_argument("--batch", type=int, default=16, help="Validation batch size.")
    p.add_argument(
        "--device",
        type=str,
        default="0" if torch.cuda.is_available() else "cpu",
        help="Device for inference.",
    )
    p.add_argument("--conf", type=float, default=0.001, help="Confidence threshold.")
    p.add_argument("--iou", type=float, default=0.6, help="NMS IoU threshold.")
    p.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run name for output files (default: model filename).",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="results",
        help="Output directory for JSON / CSV results.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Load model ───────────────────────────────────────────
    model = load_model(args.weights)

    # ── Evaluate each dataset ────────────────────────────────
    results: list[EvalResult] = []
    for ds_key in args.datasets:
        result = evaluate_dataset(
            model=model,
            dataset_key=ds_key,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            conf=args.conf,
            iou=args.iou,
        )
        results.append(result)

    # ── Console summary ──────────────────────────────────────
    print_summary_table(results)

    # ── Persist results ──────────────────────────────────────
    run_name = args.run_name or Path(args.weights).stem
    out_dir = Path(args.out_dir)

    save_json(results, out_dir / f"{run_name}_results.json")
    save_csv(results, out_dir / f"{run_name}_log.csv")

    print("\n[evaluate] Done.")


if __name__ == "__main__":
    main()
