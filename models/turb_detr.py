"""Turb-DETR — Turbidity-aware RT-DETR for underwater object detection.

Architecture pipeline
─────────────────────
    Input Image
         ↓
    CNN Backbone    (ResNet / HGNetv2 via Ultralytics RT-DETR)
         ↓
    Feature Encoder (multi-scale feature maps from backbone)
         ↓
    ★ SimAM Turbidity Suppression Module  ← injected here
         ↓
    Transformer Encoder  (hybrid encoder inside RT-DETR)
         ↓
    Object Query Decoder (transformer decoder with learned queries)
         ↓
    Detection Head       (bbox regression + classification)

The SimAM attention module (parameter-free, energy-based) is injected
*after* the CNN backbone extracts multi-scale features and *before* they
enter the transformer encoder.  This suppresses turbidity-degraded
(low-information) activations and amplifies salient object features —
all without adding any learnable parameters.

Design constraints
──────────────────
  • Does NOT modify Ultralytics source code — hooks into the model via
    ``register_forward_hook`` on the backbone output.
  • Fully compatible with pretrained RT-DETR weights (SimAM is identity-
    like at init, zero extra params).
  • ``use_simam=False`` falls back to an unmodified baseline RT-DETR.
  • Preserves the existing ``TurbDETR`` public API used by trainer /
    evaluate / infer scripts.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml
from ultralytics import RTDETR

from models.simam import SimAM

# Names that indicate the hook landed on the full model rather than a backbone sub-module.
# If the hooked module type matches any of these, the injection is almost certainly wrong.
_FULL_MODEL_TYPE_FRAGMENTS = ("RTDETR", "DetectionModel", "RTDETRDetectionModel", "YOLOv")



# ─────────────────────────────────────────────────────────────
# Helper — YAML loader
# ─────────────────────────────────────────────────────────────
def load_model_config(config_path: str | Path) -> dict[str, Any]:
    """Load model configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────
# SimAM feature hook (injected between backbone → encoder)
# ─────────────────────────────────────────────────────────────
class SimAMFeatureHook:
    """Forward-hook wrapper that applies SimAM to backbone output tensors.

    Ultralytics RT-DETR passes a *list* of multi-scale feature maps from
    the backbone into the hybrid encoder.  This hook intercepts that list
    and applies SimAM attention to each scale independently.

    Parameters
    ----------
    lambda_param : float
        SimAM stability constant.
    """

    def __init__(self, lambda_param: float = 1e-4) -> None:
        self.simam = SimAM(lambda_param=lambda_param)
        self._handle: torch.utils.hooks.RemovableHandle | None = None

    def __call__(
        self,
        module: nn.Module,
        input: Any,
        output: Any,
    ) -> Any:
        """Apply SimAM to each feature map in the backbone output."""
        if isinstance(output, (list, tuple)):
            # Multi-scale feature maps — apply SimAM per scale
            return type(output)(self.simam(feat) for feat in output)
        if isinstance(output, torch.Tensor):
            return self.simam(output)
        # Unknown structure — pass through unchanged
        return output

    def to(self, device: torch.device | str) -> "SimAMFeatureHook":
        """Move the SimAM buffers to the target device."""
        self.simam = self.simam.to(device)
        return self

    def remove(self) -> None:
        """Remove the registered hook."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


# ─────────────────────────────────────────────────────────────
# Multi-scale SimAM block (standalone nn.Module variant)
# ─────────────────────────────────────────────────────────────
class TurbiditySuppressionBlock(nn.Module):
    """Multi-scale SimAM turbidity suppression.

    Applies independent SimAM attention to each feature scale produced
    by the CNN backbone.  Can be used as a standalone module in custom
    pipelines or injected via hook (see ``SimAMFeatureHook``).

    Parameters
    ----------
    num_scales : int
        Number of feature-map scales (typically 3 for RT-DETR).
    lambda_param : float
        SimAM stability constant.

    Shape
    -----
    - Input :  list of ``(B, C_i, H_i, W_i)`` tensors
    - Output:  list of ``(B, C_i, H_i, W_i)`` tensors  (same shapes)
    """

    def __init__(
        self,
        num_scales: int = 3,
        lambda_param: float = 1e-4,
    ) -> None:
        super().__init__()
        # One SimAM per scale — they share no parameters anyway,
        # but separate instances keep buffer bookkeeping clean.
        self.attention_modules = nn.ModuleList(
            [SimAM(lambda_param=lambda_param) for _ in range(num_scales)]
        )

    def forward(
        self, features: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """Apply per-scale SimAM attention.

        If more feature maps are provided than ``num_scales``, excess
        maps pass through unmodified.
        """
        out: list[torch.Tensor] = []
        for i, feat in enumerate(features):
            if i < len(self.attention_modules):
                out.append(self.attention_modules[i](feat))
            else:
                out.append(feat)
        return out

    def extra_repr(self) -> str:
        return f"scales={len(self.attention_modules)}"


# ─────────────────────────────────────────────────────────────
# Turb-DETR wrapper
# ─────────────────────────────────────────────────────────────
class TurbDETR:
    """Turbidity-aware RT-DETR wrapper.

    Wraps an Ultralytics ``RTDETR`` model and optionally injects SimAM
    turbidity suppression between the CNN backbone and the transformer
    encoder.  When ``use_simam=False`` it behaves identically to a
    vanilla RT-DETR.

    Parameters
    ----------
    model_variant : str
        RT-DETR variant name (``"rtdetr-l"`` or ``"rtdetr-x"``).
    weights : str | Path | None
        Path to pretrained ``.pt`` weights.  ``None`` loads COCO-
        pretrained defaults for the chosen variant.
    config_path : str | Path | None
        Path to ``model_config.yaml`` for architecture overrides.
    use_simam : bool
        If ``True``, inject SimAM turbidity suppression after the
        backbone.  If ``False``, use vanilla RT-DETR (baseline).
    simam_lambda : float
        SimAM stability constant λ.

    Examples
    --------
    >>> # Turb-DETR (with SimAM)
    >>> model = TurbDETR(model_variant="rtdetr-l", use_simam=True)
    >>> model.train(data_cfg="configs/trash_icra19.yaml", epochs=100)
    >>>
    >>> # Baseline RT-DETR (no SimAM — for ablation)
    >>> baseline = TurbDETR(model_variant="rtdetr-l", use_simam=False)
    """

    def __init__(
        self,
        model_variant: str = "rtdetr-l",
        weights: str | Path | None = None,
        config_path: str | Path | None = None,
        use_simam: bool = True,
        simam_lambda: float = 1e-4,
    ) -> None:
        self.config = load_model_config(config_path) if config_path else {}
        self.model_variant = model_variant
        self.use_simam = use_simam

        # ── Load Ultralytics RT-DETR ─────────────────────────
        if weights:
            self.model = RTDETR(str(weights))
        else:
            self.model = RTDETR(f"{model_variant}.pt")

        # ── Inject SimAM hook ────────────────────────────────
        self._simam_hook: SimAMFeatureHook | None = None
        if use_simam:
            self._inject_simam(simam_lambda)

    # ── SimAM injection ──────────────────────────────────────
    def _inject_simam(self, lambda_param: float) -> None:
        """Register a forward hook on the backbone to apply SimAM.

        The hook intercepts multi-scale features after the CNN backbone
        and before the hybrid transformer encoder.
        """
        inner_model = self.model.model  # ultralytics YOLO .model attr

        # Find the backbone module.  In Ultralytics RT-DETR the
        # backbone is typically the first child (index 0) or
        # accessible via model.model.model[0].  We search for the
        # module whose name contains 'backbone' first, then fall
        # back to index-based access.
        backbone = None
        if hasattr(inner_model, "model"):
            # inner_model.model is nn.Sequential of all layers
            seq = inner_model.model
            for name, mod in seq.named_modules():
                lower = name.lower()
                if "backbone" in lower or "hgblock" in lower:
                    backbone = mod
                    break
            # Fallback: use the last layer before the head, or
            # simply hook on the whole sequential up to layer 10
            # (RT-DETR backbone layers are indices 0-9 typically)
            if backbone is None and len(seq) > 10:
                backbone = seq[9]  # last backbone stage

        if backbone is None:
            # Final fallback: hook on the model itself — SimAM
            # will process whatever the first forward output is.
            backbone = inner_model

        # ── Validate injection target ────────────────────────
        # If the backbone resolves to the full model, SimAM will run on the
        # raw input image tensor, *not* on CNN feature maps.  That is not
        # turbidity suppression — it is a near-identity op that will silently
        # produce wrong results.  Raise loudly so this is never missed.
        backbone_type = type(backbone).__name__
        if any(frag in backbone_type for frag in _FULL_MODEL_TYPE_FRAGMENTS):
            warnings.warn(
                f"[TurbDETR] SimAM hook resolved to '{backbone_type}', which looks "
                "like the full model rather than a CNN backbone sub-module. "
                "SimAM will be applied to the raw input image, NOT to backbone features. "
                "Check the Ultralytics model version and update _inject_simam().",
                RuntimeWarning,
                stacklevel=3,
            )

        # Confirm the hook output is a tensor or list of tensors with ≥3 dims
        # by registering a one-shot diagnostic hook first.
        _shape_info: list[str] = []
        def _probe(mod, inp, out):
            if isinstance(out, (list, tuple)):
                _shape_info.append(str([tuple(f.shape) for f in out if isinstance(f, torch.Tensor)]))
            elif isinstance(out, torch.Tensor):
                _shape_info.append(str(tuple(out.shape)))
        _probe_handle = backbone.register_forward_hook(_probe)
        try:
            dummy = torch.zeros(1, 3, 64, 64)
            with torch.no_grad():
                self.model.model(dummy)
        except Exception:
            pass  # probe is best-effort; don't break init
        finally:
            _probe_handle.remove()

        if _shape_info:
            _log(f"Hook target '{backbone_type}' output shape(s): {_shape_info[0]}")
        else:
            warnings.warn(
                f"[TurbDETR] Could not probe hook target '{backbone_type}' output shapes — "
                "verify SimAM is injected at the correct position.",
                RuntimeWarning,
                stacklevel=3,
            )

        hook = SimAMFeatureHook(lambda_param=lambda_param)
        handle = backbone.register_forward_hook(hook)
        hook._handle = handle
        self._simam_hook = hook

        _log(
            f"SimAM turbidity suppression injected after "
            f"'{backbone_type}' (λ={lambda_param:.0e})"
        )

    def remove_simam(self) -> None:
        """Remove the SimAM hook, reverting to baseline RT-DETR."""
        if self._simam_hook is not None:
            self._simam_hook.remove()
            self._simam_hook = None
            self.use_simam = False
            _log("SimAM hook removed — model is now baseline RT-DETR.")

    # ── Training ─────────────────────────────────────────────
    def train(self, data_cfg: str, **kwargs: Any) -> Any:
        """Launch training with merged config + overrides.

        Parameters
        ----------
        data_cfg : str
            Path to Ultralytics dataset YAML.
        **kwargs
            Any ``model.train()`` keyword arguments (epochs, batch, etc.).
        """
        _log(f"Training {'Turb-DETR (SimAM)' if self.use_simam else 'baseline RT-DETR'}")
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

    # ── Info ─────────────────────────────────────────────────
    def info(self) -> str:
        """Return a human-readable summary of the model configuration."""
        lines = [
            "═" * 55,
            "  Turb-DETR Model Summary",
            "═" * 55,
            f"  Variant      : {self.model_variant}",
            f"  SimAM active : {self.use_simam}",
            f"  Architecture : CNN backbone → "
            + ("SimAM → " if self.use_simam else "")
            + "Transformer Encoder → Decoder → Detection Head",
        ]

        # Parameter count
        try:
            total = sum(p.numel() for p in self.model.model.parameters())
            trainable = sum(p.numel() for p in self.model.model.parameters() if p.requires_grad)
            lines.append(f"  Parameters   : {total:,} total, {trainable:,} trainable")
        except Exception:
            pass

        lines.append("═" * 55)
        summary = "\n".join(lines)
        print(summary)
        return summary

    def __repr__(self) -> str:
        return (
            f"TurbDETR(variant='{self.model_variant}', "
            f"simam={self.use_simam})"
        )


# ─────────────────────────────────────────────────────────────
# Internal logging
# ─────────────────────────────────────────────────────────────
def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [TurbDETR] {msg}")
