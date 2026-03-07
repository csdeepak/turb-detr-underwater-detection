"""Turb-DETR model wrapper — integrates RT-DETR with turbidity-aware modules."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from ultralytics import RTDETR


def load_model_config(config_path: str | Path) -> dict[str, Any]:
    """Load model configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


class TurbDETR:
    """Wrapper around Ultralytics RT-DETR with project-specific defaults.

    Parameters
    ----------
    model_variant : str
        RT-DETR variant name (e.g., ``"rtdetr-l"``, ``"rtdetr-x"``).
    weights : str | Path | None
        Path to pretrained weights. ``None`` loads COCO-pretrained defaults.
    config_path : str | Path | None
        Path to ``model_config.yaml`` for architecture overrides.
    """

    def __init__(
        self,
        model_variant: str = "rtdetr-l",
        weights: str | Path | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        self.config = load_model_config(config_path) if config_path else {}
        self.model_variant = model_variant

        if weights:
            self.model = RTDETR(str(weights))
        else:
            self.model = RTDETR(f"{model_variant}.pt")

    # ── Training ─────────────────────────────────────────────
    def train(self, data_cfg: str, **kwargs: Any) -> Any:
        """Launch training with merged config + overrides."""
        return self.model.train(data=data_cfg, **kwargs)

    # ── Validation ───────────────────────────────────────────
    def validate(self, data_cfg: str, **kwargs: Any) -> Any:
        """Run validation and return metrics."""
        return self.model.val(data=data_cfg, **kwargs)

    # ── Inference ────────────────────────────────────────────
    def predict(self, source: str | Path, **kwargs: Any) -> Any:
        """Run inference on image(s)."""
        return self.model.predict(source=str(source), **kwargs)

    # ── Export ───────────────────────────────────────────────
    def export(self, fmt: str = "onnx", **kwargs: Any) -> str:
        """Export model to deployment format (onnx, torchscript, etc.)."""
        return self.model.export(format=fmt, **kwargs)
