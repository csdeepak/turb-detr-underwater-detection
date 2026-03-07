"""Visualization utilities for detection results."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Default color palette (BGR) for up to 10 classes
_PALETTE = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
    (26, 147, 52), (0, 212, 187),
]


def draw_detections(
    image: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    class_names: list[str],
    score_thresh: float = 0.25,
) -> np.ndarray:
    """Draw bounding boxes with labels on an image.

    Parameters
    ----------
    image : np.ndarray
        BGR image (H, W, 3).
    boxes : np.ndarray
        Bounding boxes in xyxy format, shape (N, 4).
    scores : np.ndarray
        Confidence scores, shape (N,).
    class_ids : np.ndarray
        Integer class indices, shape (N,).
    class_names : list[str]
        Human-readable class names.
    score_thresh : float
        Minimum score for display.
    """
    vis = image.copy()
    for box, score, cls_id in zip(boxes, scores, class_ids):
        if score < score_thresh:
            continue
        x1, y1, x2, y2 = map(int, box)
        color = _PALETTE[int(cls_id) % len(_PALETTE)]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{class_names[int(cls_id)]} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(vis, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return vis


def plot_metrics_curve(
    metrics: dict[str, list[float]],
    title: str = "Training Metrics",
    save_path: str | Path | None = None,
) -> None:
    """Plot training metric curves (loss, mAP, etc.).

    Parameters
    ----------
    metrics : dict
        Mapping of metric_name → list of per-epoch values.
    title : str
        Plot title.
    save_path : str | Path | None
        If provided, save figure to this path.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, values in metrics.items():
        ax.plot(values, label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
