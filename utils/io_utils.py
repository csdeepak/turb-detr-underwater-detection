"""I/O utilities — path resolution, Colab detection, config loading."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def is_colab() -> bool:
    """Detect if running inside Google Colab."""
    try:
        import google.colab  # type: ignore[import-not-found]
        return True
    except ImportError:
        return False


def get_project_root() -> Path:
    """Return the absolute path to the project root directory.

    Works both locally (VS Code) and on Colab (mounted drive).
    """
    if is_colab():
        colab_root = Path("/content/turb-detr-underwater-detection")
        if colab_root.exists():
            return colab_root
    # Fallback: walk up from this file
    return Path(__file__).resolve().parents[1]


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it does not exist. Return the Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_device() -> str:
    """Return best available compute device string for PyTorch."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
