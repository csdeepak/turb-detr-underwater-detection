"""Single ablation variant training script — called as a subprocess by ablation.py.

This script exists so that each ablation variant runs in its own Python process,
giving clean GPU memory isolation between runs.  ablation.py calls it with:

    python scripts/train_variant.py \\
        --data configs/trash_icra19.yaml \\
        --epochs 50 --batch 8 --imgsz 640 \\
        --project outputs/ablation --name A_baseline \\
        --use-simam 0 --use-turb-aug 0

Exit codes
----------
0 — training completed successfully
1 — training failed (logged to stderr)
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.turb_detr import TurbDETR
from augmentation.turbidity_aug import apply_turbidity


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a single Turb-DETR ablation variant")
    p.add_argument("--data",       type=str, required=True, help="Dataset YAML path")
    p.add_argument("--epochs",     type=int, default=50)
    p.add_argument("--batch",      type=int, default=8)
    p.add_argument("--imgsz",      type=int, default=640)
    p.add_argument("--project",    type=str, default="outputs/ablation")
    p.add_argument("--name",       type=str, required=True, help="Run name (variant id)")
    p.add_argument("--use-simam",  type=int, default=0, choices=[0, 1],
                   help="1 = inject SimAM after backbone, 0 = vanilla RT-DETR")
    p.add_argument("--use-turb-aug", type=int, default=0, choices=[0, 1],
                   help="1 = apply Beer-Lambert turbidity aug during training, 0 = skip")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    use_simam     = bool(args.use_simam)
    use_turb_aug  = bool(args.use_turb_aug)

    print(f"[train_variant] name={args.name}  simam={use_simam}  turb_aug={use_turb_aug}")

    model = TurbDETR(model_variant="rtdetr-l", use_simam=use_simam)

    # ── Turbidity augmentation callback ──────────────────────────────────
    if use_turb_aug:
        class _TurbidityTransform:
            def __call__(self, labels: dict) -> dict:
                if random.random() < 0.5:
                    img = labels.get("img")
                    if img is not None:
                        labels["img"] = apply_turbidity(img, level=random.uniform(0.1, 0.7))
                return labels

        def _on_train_start(trainer) -> None:
            ds = getattr(trainer, "train_loader", None)
            if ds is None:
                print("[train_variant] WARNING: train_loader not found — turbidity aug skipped.")
                return
            ds = trainer.train_loader.dataset
            transforms_obj = getattr(ds, "transforms", None)
            if transforms_obj is None or not hasattr(transforms_obj, "transforms"):
                print(
                    "[train_variant] WARNING: dataset.transforms has unexpected structure — "
                    "turbidity aug NOT applied. Check Ultralytics version (requires >= 8.3)."
                )
                return
            transforms_obj.transforms.insert(0, _TurbidityTransform())
            print("[train_variant] Turbidity aug injected (prob=0.5, level=0.1-0.7)")

        model.model.add_callback("on_train_start", _on_train_start)

    # ── Train ─────────────────────────────────────────────────────────────
    try:
        model.train(
            data_cfg=args.data,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            project=args.project,
            name=args.name,
            exist_ok=True,
            mixup=0.1,
            mosaic=1.0,
            degrees=5.0,
        )
    except Exception as exc:
        print(f"[train_variant] FATAL: training failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
