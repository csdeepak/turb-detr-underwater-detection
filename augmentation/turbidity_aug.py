"""Physically-inspired underwater turbidity simulation.

Models the key optical phenomena of turbid underwater environments:
  - **Beer–Lambert light attenuation** (exponential decay with depth/distance)
  - **Wavelength-dependent colour absorption** (red lost first → green/blue cast)
  - **Backscatter haze** (veiling light from suspended particles)
  - **Forward-scatter blur** (reduced contrast & edge softening)
  - **Particle noise** (floating sediment / marine snow)
  - **Scattering artifacts** (local intensity perturbations)

All functions accept BGR uint8 images (OpenCV convention) and return the same.
The module is designed to slot into both Albumentations and raw PyTorch
DataLoader pipelines with zero dependencies beyond NumPy + OpenCV.

Usage:
    from augmentation.turbidity_aug import apply_turbidity, visualize_turbidity
    augmented = apply_turbidity(image, level=0.5)
    visualize_turbidity(image)              # shows light / medium / heavy

Reference:
    Jaffe, J.S. (1990). "Computer modeling and the design of optimal
    underwater imaging systems." IEEE J. Oceanic Eng.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
# Typical attenuation coefficients (per-metre) in turbid coastal water.
# Order: B, G, R  (OpenCV BGR convention)
_ATTEN_COEFFS = np.array([0.02, 0.03, 0.10], dtype=np.float32)

# Default backscatter veiling-light colour (greenish-blue, BGR)
_VEIL_COLOR = np.array([140, 160, 90], dtype=np.float32)


# ─────────────────────────────────────────────────────────────
# Core: colour attenuation (Beer–Lambert per channel)
# ─────────────────────────────────────────────────────────────
def adjust_color_attenuation(
    image: np.ndarray,
    depth: float = 5.0,
    coeffs: np.ndarray | None = None,
) -> np.ndarray:
    """Simulate wavelength-dependent absorption via the Beer–Lambert law.

    Each channel is attenuated as  I_out = I_in × exp(-c × d)  where *c*
    is the per-channel attenuation coefficient and *d* is the virtual depth.

    Parameters
    ----------
    image : np.ndarray
        BGR uint8 image.
    depth : float
        Simulated optical depth / distance (metres).  Larger → more
        red-loss and stronger blue-green cast.
    coeffs : np.ndarray | None
        Per-channel attenuation coefficients [B, G, R].
        Defaults to typical turbid coastal water values.

    Returns
    -------
    np.ndarray
        Colour-attenuated BGR uint8 image.
    """
    if coeffs is None:
        coeffs = _ATTEN_COEFFS

    transmission = np.exp(-coeffs * depth)  # shape (3,)
    result = image.astype(np.float32) * transmission[np.newaxis, np.newaxis, :]
    return np.clip(result, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────
# Core: backscatter haze
# ─────────────────────────────────────────────────────────────
def _add_backscatter_haze(
    image: np.ndarray,
    level: float,
    veil_color: np.ndarray | None = None,
) -> np.ndarray:
    """Blend a uniform veiling-light layer to simulate backscatter.

    Parameters
    ----------
    image : np.ndarray
        BGR uint8 image.
    level : float
        Haze intensity in [0, 1].  0 = no haze, 1 = fully veiled.
    veil_color : np.ndarray | None
        BGR veiling-light colour.  Defaults to greenish-blue.
    """
    if veil_color is None:
        veil_color = _VEIL_COLOR

    veil = np.full_like(image, veil_color, dtype=np.float32)
    blended = image.astype(np.float32) * (1 - level) + veil * level
    return np.clip(blended, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────
# Core: forward-scatter blur (contrast reduction)
# ─────────────────────────────────────────────────────────────
def _apply_forward_scatter(
    image: np.ndarray,
    level: float,
) -> np.ndarray:
    """Reduce contrast and soften edges via Gaussian blur blending.

    Parameters
    ----------
    image : np.ndarray
        BGR uint8 image.
    level : float
        Scatter strength in [0, 1].  Controls blur kernel size and
        blend ratio.
    """
    ksize = int(3 + level * 20) | 1  # ensure odd
    blurred = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=0)
    alpha = level * 0.6  # don't fully replace — blend
    result = cv2.addWeighted(image, 1 - alpha, blurred, alpha, 0)
    return result


# ─────────────────────────────────────────────────────────────
# Core: particle / scattering noise
# ─────────────────────────────────────────────────────────────
def add_scattering_noise(
    image: np.ndarray,
    density: float = 0.002,
    intensity_range: tuple[int, int] = (180, 255),
    size_range: tuple[int, int] = (1, 3),
    seed: int | None = None,
) -> np.ndarray:
    """Add bright particle specks to simulate marine snow / sediment.

    Parameters
    ----------
    image : np.ndarray
        BGR uint8 image.
    density : float
        Fraction of pixels that receive a particle (0–1).
    intensity_range : tuple[int, int]
        Min/max brightness of particles.
    size_range : tuple[int, int]
        Min/max radius (pixels) of each particle circle.
    seed : int | None
        RNG seed for reproducibility.

    Returns
    -------
    np.ndarray
        Image with particle noise overlay.
    """
    rng = np.random.default_rng(seed)
    h, w = image.shape[:2]
    n_particles = int(h * w * density)

    if n_particles == 0:
        return image.copy()

    result = image.copy()
    ys = rng.integers(0, h, size=n_particles)
    xs = rng.integers(0, w, size=n_particles)
    radii = rng.integers(size_range[0], size_range[1] + 1, size=n_particles)
    intensities = rng.integers(
        intensity_range[0], intensity_range[1] + 1, size=n_particles,
    )

    for y, x, r, v in zip(ys, xs, radii, intensities):
        cv2.circle(result, (int(x), int(y)), int(r), (int(v), int(v), int(v)), -1)

    # Soften particles slightly so they look natural
    return cv2.GaussianBlur(result, (3, 3), sigmaX=0.5)


# ─────────────────────────────────────────────────────────────
# Composite: apply_turbidity
# ─────────────────────────────────────────────────────────────
def apply_turbidity(
    image: np.ndarray,
    level: float = 0.5,
    seed: int | None = None,
) -> np.ndarray:
    """Apply a composite, physically-inspired turbidity augmentation.

    Chains the following effects proportional to *level*:
        1. Colour attenuation  (Beer–Lambert)
        2. Backscatter haze    (veiling light)
        3. Forward-scatter blur (contrast loss)
        4. Particle noise      (marine snow)

    Parameters
    ----------
    image : np.ndarray
        BGR uint8 input image.
    level : float
        Turbidity severity in [0, 1].
        0.0 = clear water, 1.0 = extremely turbid.
    seed : int | None
        RNG seed for particle noise reproducibility.

    Returns
    -------
    np.ndarray
        Augmented BGR uint8 image, same shape as input.
    """
    level = float(np.clip(level, 0.0, 1.0))

    # 1. Colour attenuation — depth scales with level
    depth = 2.0 + level * 18.0  # 2 m (clear) → 20 m (heavy)
    result = adjust_color_attenuation(image, depth=depth)

    # 2. Backscatter haze
    haze_strength = level * 0.55
    result = _add_backscatter_haze(result, level=haze_strength)

    # 3. Forward-scatter blur
    scatter_strength = level * 0.7
    result = _apply_forward_scatter(result, level=scatter_strength)

    # 4. Particle noise (marine snow) — density increases with level
    particle_density = level * 0.004
    if particle_density > 0:
        result = add_scattering_noise(
            result,
            density=particle_density,
            seed=seed,
        )

    return result


# ─────────────────────────────────────────────────────────────
# Preset helpers
# ─────────────────────────────────────────────────────────────
_PRESETS: dict[str, float] = {
    "clear": 0.0,
    "light": 0.25,
    "medium": 0.50,
    "heavy": 0.80,
    "extreme": 1.0,
}


def apply_turbidity_preset(
    image: np.ndarray,
    preset: Literal["clear", "light", "medium", "heavy", "extreme"] = "medium",
    seed: int | None = None,
) -> np.ndarray:
    """Apply turbidity at a named preset level.

    Parameters
    ----------
    image : np.ndarray
        BGR uint8 image.
    preset : str
        One of ``"clear"``, ``"light"``, ``"medium"``, ``"heavy"``, ``"extreme"``.
    seed : int | None
        RNG seed.
    """
    level = _PRESETS.get(preset, 0.5)
    return apply_turbidity(image, level=level, seed=seed)


# ─────────────────────────────────────────────────────────────
# PyTorch-compatible callable transform
# ─────────────────────────────────────────────────────────────
class TurbidityTransform:
    """PyTorch-pipeline-compatible turbidity augmentation.

    Accepts either a BGR NumPy array or a ``torch.Tensor`` (C, H, W)
    and returns the same type.

    Parameters
    ----------
    level_range : tuple[float, float]
        Random turbidity level is sampled uniformly from this range.
    p : float
        Probability of applying the transform.
    """

    def __init__(
        self,
        level_range: tuple[float, float] = (0.1, 0.7),
        p: float = 0.5,
    ) -> None:
        self.level_range = level_range
        self.p = p

    def __call__(self, image: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng()
        if rng.random() > self.p:
            return image

        # Handle torch.Tensor input
        is_tensor = False
        try:
            import torch
            if isinstance(image, torch.Tensor):
                is_tensor = True
                device = image.device
                # (C, H, W) float [0,1] → (H, W, C) uint8
                np_img = (image.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        except ImportError:
            pass

        if not is_tensor:
            np_img = image

        level = rng.uniform(*self.level_range)
        result = apply_turbidity(np_img, level=level)

        if is_tensor:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(result).float().permute(2, 0, 1) / 255.0
            return tensor.to(device)

        return result

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"level_range={self.level_range}, p={self.p})"
        )


# ─────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────
def visualize_turbidity(
    image: np.ndarray,
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (18, 5),
) -> None:
    """Display or save a 4-panel comparison: original → light → medium → heavy.

    Parameters
    ----------
    image : np.ndarray
        BGR uint8 input image.
    save_path : str | Path | None
        If provided, saves the figure instead of showing it.
    figsize : tuple[int, int]
        Matplotlib figure size.
    """
    import matplotlib.pyplot as plt

    panels = {
        "Original": image,
        "Light (0.25)": apply_turbidity(image, level=0.25, seed=42),
        "Medium (0.50)": apply_turbidity(image, level=0.50, seed=42),
        "Heavy (0.80)": apply_turbidity(image, level=0.80, seed=42),
    }

    fig, axes = plt.subplots(1, 4, figsize=figsize)
    for ax, (title, img) in zip(axes, panels.items()):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.axis("off")

    plt.suptitle("Turbidity Augmentation — Simulated Underwater Conditions",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()


# ─────────────────────────────────────────────────────────────
# CLI demo
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Turbidity augmentation demo")
    p.add_argument("image", type=str, help="Path to input image")
    p.add_argument("--save", type=str, default=None, help="Save path for visualization")
    args = p.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {args.image}")

    visualize_turbidity(img, save_path=args.save)
