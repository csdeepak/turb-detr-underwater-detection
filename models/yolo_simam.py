"""YOLOv10 + SimAM Baseline — Ablation Baseline B.

This module provides a YOLOv10 model with SimAM attention injected at the
backbone feature pyramid — identical in spirit to how Turb-DETR injects
SimAM into RT-DETR.

Purpose
-------
This is a controlled ablation baseline to answer:

    "Does the mAP improvement come from SimAM, or from the transformer
     architecture of RT-DETR?"

By comparing:
  YOLOv10           vs   YOLOv10 + SimAM
  RT-DETR           vs   Turb-DETR (RT-DETR + SimAM)

you can isolate whether SimAM is architecture-agnostic or specifically
beneficial in the transformer pipeline.

IMPORTANT — Architectural difference
--------------------------------------
YOLOv10 uses a CNN backbone (CSPDarknet/backbone → PAN-FPN neck → head).
SimAM is injected *after* the backbone's last feature stage, *before* the
FPN neck.  This is structurally equivalent to Turb-DETR's injection point
(backbone → SimAM → encoder/neck).

However, YOLO does *not* have a global attention transformer encoder.  If
Turb-DETR outperforms YOLO+SimAM, part of that gap is attributable to the
transformer's global reasoning — not SimAM alone.

Usage
-----
    from models.yolo_simam import YOLOSimAM

    model = YOLOSimAM(model_variant="yolov10l", use_simam=True)
    model.train(data_cfg="configs/trash_icra19.yaml", epochs=100)
    model.validate(data_cfg="configs/trash_icra19.yaml")

CLI
---
    python models/yolo_simam.py \\
        --data configs/trash_icra19.yaml \\
        --epochs 50 --batch 8
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from ultralytics import YOLO

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.simam import SimAM


# ─────────────────────────────────────────────────────────────
# SimAM hook — identical to TurbDETR's SimAMFeatureHook
# ─────────────────────────────────────────────────────────────
class SimAMHook:
    """Forward hook that applies SimAM to backbone output tensors."""

    def __init__(self, lambda_param: float = 1e-4) -> None:
        self.simam = SimAM(lambda_param=lambda_param)
        self._handle = None

    def __call__(self, module: nn.Module, input: Any, output: Any) -> Any:
        if isinstance(output, (list, tuple)):
            return type(output)(self.simam(f) for f in output)
        if isinstance(output, torch.Tensor):
            return self.simam(output)
        return output

    def to(self, device) -> "SimAMHook":
        self.simam = self.simam.to(device)
        return self

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


# ─────────────────────────────────────────────────────────────
# Backbone locator for YOLO models
# ─────────────────────────────────────────────────────────────
def _find_yolo_backbone(inner_model: nn.Module) -> nn.Module:
    """Locate the backbone module in a YOLOv10 inner model.

    YOLOv10 inner model is an nn.Sequential of layers.  The backbone
    layers are layers 0–9 (indices 0 to 9); the FPN neck starts ~index 10.
    We hook on layer 9 (last C3/C2f backbone stage) — the same multi-scale
    feature maps that feed into the neck.
    """
    if hasattr(inner_model, "model"):
        seq = inner_model.model
        # Search for named backbone stage first
        for name, mod in seq.named_modules():
            lower = name.lower()
            if any(k in lower for k in ("backbone", "dark5", "c2f", "c3")):
                return mod
        # Fallback: hook on layer 9 (standard YOLOv8/v10 backbone depth)
        if hasattr(seq, "__len__") and len(seq) > 9:
            return seq[9]
    return inner_model


# ─────────────────────────────────────────────────────────────
# YOLOSimAM wrapper
# ─────────────────────────────────────────────────────────────
class YOLOSimAM:
    """YOLOv10 with optional SimAM attention injection.

    Parameters
    ----------
    model_variant : str
        Ultralytics YOLO model name, e.g. "yolov10l", "yolov10m", "yolov8l".
    weights : str | Path | None
        Custom weights path, or None to load COCO-pretrained defaults.
    use_simam : bool
        If True, inject SimAM after the backbone. If False, vanilla YOLO.
    simam_lambda : float
        SimAM stability constant λ.

    Examples
    --------
    >>> # YOLOv10 baseline (no SimAM)
    >>> baseline = YOLOSimAM("yolov10l", use_simam=False)
    >>> baseline.train(data_cfg="configs/trash_icra19.yaml", epochs=100)
    >>>
    >>> # YOLOv10 + SimAM
    >>> model = YOLOSimAM("yolov10l", use_simam=True)
    >>> model.train(data_cfg="configs/trash_icra19.yaml", epochs=100)
    """

    def __init__(
        self,
        model_variant: str = "yolov10l",
        weights: str | Path | None = None,
        use_simam: bool = True,
        simam_lambda: float = 1e-4,
    ) -> None:
        self.model_variant = model_variant
        self.use_simam     = use_simam

        if weights:
            self.model = YOLO(str(weights))
        else:
            self.model = YOLO(f"{model_variant}.pt")

        self._hook: SimAMHook | None = None
        if use_simam:
            self._inject_simam(simam_lambda)

    def _inject_simam(self, lambda_param: float) -> None:
        inner_model = self.model.model
        backbone    = _find_yolo_backbone(inner_model)

        hook   = SimAMHook(lambda_param=lambda_param)
        handle = backbone.register_forward_hook(hook)
        hook._handle = handle
        self._hook   = hook

        _log(
            f"SimAM injected after '{type(backbone).__name__}' "
            f"in {self.model_variant} (λ={lambda_param:.0e})"
        )

    def remove_simam(self) -> None:
        if self._hook is not None:
            self._hook.remove()
            self._hook    = None
            self.use_simam = False
            _log("SimAM hook removed — model is now vanilla YOLO.")

    # ── Training ─────────────────────────────────────────────
    def train(self, data_cfg: str, **kwargs: Any) -> Any:
        label = f"YOLOv10 + SimAM" if self.use_simam else f"Baseline {self.model_variant}"
        _log(f"Training {label}")
        return self.model.train(data=data_cfg, **kwargs)

    # ── Validation ───────────────────────────────────────────
    def validate(self, data_cfg: str, **kwargs: Any) -> Any:
        return self.model.val(data=data_cfg, **kwargs)

    # ── Inference ────────────────────────────────────────────
    def predict(self, source: str | Path, **kwargs: Any) -> Any:
        return self.model.predict(source=str(source), **kwargs)

    # ── Export ───────────────────────────────────────────────
    def export(self, fmt: str = "onnx", **kwargs: Any) -> Any:
        return self.model.export(format=fmt, **kwargs)

    # ── Info ─────────────────────────────────────────────────
    def info(self) -> str:
        lines = [
            "=" * 55,
            f"  {'YOLOv10 + SimAM' if self.use_simam else 'YOLOv10 Baseline'}",
            "=" * 55,
            f"  Variant      : {self.model_variant}",
            f"  SimAM active : {self.use_simam}",
        ]
        try:
            total     = sum(p.numel() for p in self.model.model.parameters())
            trainable = sum(p.numel() for p in self.model.model.parameters() if p.requires_grad)
            lines.append(f"  Parameters   : {total:,} total, {trainable:,} trainable")
        except Exception:
            pass
        lines.append("=" * 55)
        summary = "\n".join(lines)
        print(summary)
        return summary

    def __repr__(self) -> str:
        return f"YOLOSimAM(variant='{self.model_variant}', simam={self.use_simam})"


# ─────────────────────────────────────────────────────────────
# Internal logging
# ─────────────────────────────────────────────────────────────
def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [YOLOSimAM] {msg}")


# ─────────────────────────────────────────────────────────────
# CLI — quick train
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train YOLOv10 with optional SimAM")
    parser.add_argument("--data",    required=True, help="Dataset YAML path")
    parser.add_argument("--variant", default="yolov10l", help="YOLO variant (default: yolov10l)")
    parser.add_argument("--weights", default=None,  help="Optional custom weights .pt")
    parser.add_argument("--simam",   type=int, default=1, help="1=use SimAM, 0=baseline YOLO")
    parser.add_argument("--epochs",  type=int, default=100)
    parser.add_argument("--batch",   type=int, default=16)
    parser.add_argument("--imgsz",   type=int, default=640)
    parser.add_argument("--project", default="outputs")
    parser.add_argument("--name",    default=None)
    args = parser.parse_args()

    run_name = args.name or (f"{args.variant}_simam" if args.simam else f"{args.variant}_baseline")
    model    = YOLOSimAM(
        model_variant=args.variant,
        weights=args.weights,
        use_simam=bool(args.simam),
    )
    model.info()
    model.train(
        data_cfg=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=args.project,
        name=run_name,
        exist_ok=True,
    )
