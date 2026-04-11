"""
Calibrate synthetic turbidity parameters against UFO-120 real images.

Uses PSNR/SSIM to match synthetic degradation to real turbidity.
This makes your augmentation parameters scientifically defensible.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from .jaffe_mcglamery import jaffe_mcglamery_turbidity


def compute_real_turbidity_stats(clean_dir: str, turbid_dir: str) -> dict:
    """Compute PSNR/SSIM stats between real clean-turbid pairs from UFO-120."""
    clean_path = Path(clean_dir)
    turbid_path = Path(turbid_dir)

    psnr_vals, ssim_vals = [], []

    for clean_img_path in sorted(clean_path.glob("*.png")):
        turbid_img_path = turbid_path / clean_img_path.name
        if not turbid_img_path.exists():
            continue

        clean_img = cv2.imread(str(clean_img_path))
        turbid_img = cv2.imread(str(turbid_img_path))

        if clean_img is None or turbid_img is None:
            continue

        # Resize to match if needed
        if clean_img.shape != turbid_img.shape:
            turbid_img = cv2.resize(turbid_img, (clean_img.shape[1], clean_img.shape[0]))

        psnr_vals.append(psnr(clean_img, turbid_img))
        ssim_vals.append(ssim(clean_img, turbid_img, channel_axis=2))

    return {
        "psnr_mean": np.mean(psnr_vals),
        "psnr_std": np.std(psnr_vals),
        "ssim_mean": np.mean(ssim_vals),
        "ssim_std": np.std(ssim_vals),
        "n_pairs": len(psnr_vals),
    }


def find_matching_c(
    sample_images: list[np.ndarray],
    target_psnr: float,
    c_range: tuple = (0.01, 0.50),
    n_steps: int = 50,
) -> float:
    """Binary search for attenuation coefficient c that produces target PSNR."""
    c_values = np.linspace(c_range[0], c_range[1], n_steps)
    best_c, best_diff = c_values[0], float("inf")

    for c in c_values:
        psnr_vals = []
        for img in sample_images:
            turbid = jaffe_mcglamery_turbidity(img, c=c, seed=42)
            psnr_vals.append(psnr(img, turbid))
        mean_psnr = np.mean(psnr_vals)
        diff = abs(mean_psnr - target_psnr)
        if diff < best_diff:
            best_diff = diff
            best_c = c

    return best_c


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ufo120-clean", required=True)
    parser.add_argument("--ufo120-turbid", required=True)
    args = parser.parse_args()

    stats = compute_real_turbidity_stats(args.ufo120_clean, args.ufo120_turbid)
    print(f"UFO-120 Real Turbidity Stats:")
    print(f"  PSNR: {stats['psnr_mean']:.2f} ± {stats['psnr_std']:.2f}")
    print(f"  SSIM: {stats['ssim_mean']:.4f} ± {stats['ssim_std']:.4f}")
    print(f"  Pairs: {stats['n_pairs']}")
