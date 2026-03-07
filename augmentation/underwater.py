"""Underwater-specific augmentations — turbidity, color shifts, caustics."""

from __future__ import annotations

import cv2
import numpy as np


def simulate_turbidity(
    image: np.ndarray,
    intensity: float = 0.3,
    color: tuple[int, int, int] = (120, 150, 100),
) -> np.ndarray:
    """Overlay a semi-transparent fog layer to simulate underwater turbidity.

    Parameters
    ----------
    image : np.ndarray
        BGR input image.
    intensity : float
        Blending factor in [0, 1]. Higher = more turbid.
    color : tuple
        BGR tint color of the turbid water.
    """
    fog = np.full_like(image, color, dtype=np.uint8)
    return cv2.addWeighted(image, 1 - intensity, fog, intensity, 0)


def underwater_color_shift(
    image: np.ndarray,
    blue_gain: float = 1.2,
    red_loss: float = 0.7,
) -> np.ndarray:
    """Simulate wavelength-dependent absorption (red lost first).

    Parameters
    ----------
    image : np.ndarray
        BGR input image (uint8).
    blue_gain : float
        Multiplier for blue channel (> 1 = stronger blue cast).
    red_loss : float
        Multiplier for red channel (< 1 = red attenuation).
    """
    result = image.astype(np.float32)
    result[:, :, 0] = np.clip(result[:, :, 0] * blue_gain, 0, 255)   # B
    result[:, :, 2] = np.clip(result[:, :, 2] * red_loss, 0, 255)    # R
    return result.astype(np.uint8)


def add_caustic_pattern(
    image: np.ndarray,
    strength: float = 0.15,
    scale: float = 50.0,
) -> np.ndarray:
    """Add procedural caustic light patterns (sinusoidal interference).

    Parameters
    ----------
    image : np.ndarray
        BGR input image.
    strength : float
        Intensity of the caustic overlay.
    scale : float
        Spatial frequency of the pattern.
    """
    h, w = image.shape[:2]
    x = np.arange(w, dtype=np.float32)
    y = np.arange(h, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    pattern = (
        np.sin(xx / scale * 2 * np.pi)
        * np.cos(yy / scale * 2 * np.pi)
    )
    pattern = ((pattern + 1) / 2 * 255 * strength).astype(np.uint8)
    pattern = np.stack([pattern] * 3, axis=-1)

    return cv2.add(image, pattern)
