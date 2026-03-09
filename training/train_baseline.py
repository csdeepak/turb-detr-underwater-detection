"""Baseline RT-DETR training script for Trash-ICRA19 underwater detection.

Usage (CLI):
    python training/train_baseline.py
    python training/train_baseline.py --epochs 50 --batch 8 --imgsz 640
    python training/train_baseline.py --data configs/trash_icra19.yaml --model rtdetr-x

Usage (Colab):
    %run training/train_baseline.py --epochs 100 --batch 16
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ── Ensure project root is importable ────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from ultralytics import RTDETR

from utils.io_utils import get_device, load_yaml


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train baseline RT-DETR on Trash-ICRA19",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data & model
    p.add_argument("--data", type=str, default="configs/trash_icra19.yaml",
                   help="Path to Ultralytics dataset YAML")
    p.add_argument("--model", type=str, default="rtdetr-l",
                   help="RT-DETR variant: rtdetr-l | rtdetr-x")
    p.add_argument("--weights", type=str, default=None,
                   help="Resume from custom weights (.pt)")

    # Training hyper-parameters
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--optimizer", type=str, default="AdamW",
                   help="Optimizer: AdamW | SGD | Adam | auto")
    p.add_argument("--lr0", type=float, default=1e-4,
                   help="Initial learning rate")
    p.add_argument("--lrf", type=float, default=0.01,
                   help="Final LR as fraction of lr0 (cosine decay)")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--patience", type=int, default=15,
                   help="Early-stopping patience (0 = disabled)")

    # Runtime
    p.add_argument("--workers", type=int, default=4,
                   help="Dataloader workers")
    p.add_argument("--amp", action="store_true", default=True,
                   help="Automatic mixed precision")
    p.add_argument("--no-amp", dest="amp", action="store_false")

    # Output
    p.add_argument("--project", type=str, default="outputs",
                   help="Output project directory")
    p.add_argument("--name", type=str, default="rtdetr_baseline_trash_icra19",
                   help="Run name inside project dir")
    p.add_argument("--save-period", type=int, default=10,
                   help="Save checkpoint every N epochs")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# Logging helpers
# ─────────────────────────────────────────────────────────────
def log(msg: str) -> None:
    """Print a timestamped log line."""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def print_env_info() -> None:
    """Log environment & hardware details."""
    device = get_device()
    log(f"PyTorch       : {torch.__version__}")
    log(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"GPU           : {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        log(f"GPU memory    : {mem:.1f} GB")
    log(f"Device        : {device}")


def print_dataset_info(data_cfg_path: str) -> None:
    """Log dataset configuration summary."""
    cfg = load_yaml(data_cfg_path)
    log(f"Dataset path  : {cfg.get('path', 'N/A')}")
    log(f"Classes ({cfg.get('nc', '?')}): {list(cfg.get('names', {}).values())}")
    log(f"Train split   : {cfg.get('train', 'N/A')}")
    log(f"Val split     : {cfg.get('val', 'N/A')}")
    log(f"Test split    : {cfg.get('test', 'N/A')}")


def print_metrics(metrics) -> None:
    """Log final evaluation metrics from Ultralytics results."""
    box = metrics.box
    log("─── Evaluation Results ───────────────────────────")
    log(f"  mAP@50      : {box.map50:.4f}")
    log(f"  mAP@50-95   : {box.map:.4f}")
    log(f"  Precision   : {box.mp:.4f}")
    log(f"  Recall      : {box.mr:.4f}")

    # Per-class breakdown
    names = metrics.names
    if hasattr(box, "ap50") and box.ap50 is not None:
        log("  Per-class AP@50:")
        for i, ap in enumerate(box.ap50):
            cls_name = names.get(i, f"class_{i}")
            log(f"    {cls_name:15s} : {ap:.4f}")


# ─────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────
def train(args: argparse.Namespace):
    """Run the full training → evaluation pipeline.

    Flow:
        1. Load dataset config  →  validate paths
        2. Initialise RT-DETR   →  COCO-pretrained or custom weights
        3. Train                →  with all configured hyper-params
        4. Evaluate             →  mAP, precision, recall
        5. Save                 →  best.pt / last.pt
    """
    # ── 1. Environment ───────────────────────────────────────
    log("=" * 55)
    log("  Turb-DETR Baseline Training — Trash-ICRA19")
    log("=" * 55)
    print_env_info()

    # ── 2. Dataset info ──────────────────────────────────────
    log("")
    log("─── Dataset ─────────────────────────────────────")
    print_dataset_info(args.data)

    # ── 3. Model initialisation ──────────────────────────────
    log("")
    log("─── Model ───────────────────────────────────────")
    if args.weights:
        log(f"Resuming from weights: {args.weights}")
        model = RTDETR(args.weights)
    else:
        model_file = f"{args.model}.pt"
        log(f"Loading pretrained: {model_file}")
        model = RTDETR(model_file)

    # ── 4. Training ──────────────────────────────────────────
    log("")
    log("─── Training Config ─────────────────────────────")
    log(f"  Epochs       : {args.epochs}")
    log(f"  Batch size   : {args.batch}")
    log(f"  Image size   : {args.imgsz}")
    log(f"  Optimizer    : {args.optimizer}")
    log(f"  LR           : {args.lr0} → ×{args.lrf}")
    log(f"  Weight decay : {args.weight_decay}")
    log(f"  Warmup       : {args.warmup_epochs} epochs")
    log(f"  Patience     : {args.patience}")
    log(f"  AMP          : {args.amp}")
    log(f"  Output       : {args.project}/{args.name}")
    log("")

    start = time.time()

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        amp=args.amp,
        workers=args.workers,
        project=args.project,
        name=args.name,
        save_period=args.save_period,
        exist_ok=True,
        verbose=True,
    )

    elapsed = time.time() - start
    log(f"Training finished in {elapsed / 60:.1f} min")

    # ── 5. Evaluation ────────────────────────────────────────
    log("")
    log("─── Validation ──────────────────────────────────")
    metrics = model.val(data=args.data)
    print_metrics(metrics)

    # ── 6. Summary ───────────────────────────────────────────
    weights_dir = Path(args.project) / args.name / "weights"
    best_pt = weights_dir / "best.pt"
    log("")
    log("─── Saved Artifacts ─────────────────────────────")
    log(f"  Best weights : {best_pt} ({'✓ exists' if best_pt.exists() else '✗ missing'})")
    log(f"  Run directory: {Path(args.project) / args.name}")
    log("=" * 55)

    return results, metrics


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    train(args)
