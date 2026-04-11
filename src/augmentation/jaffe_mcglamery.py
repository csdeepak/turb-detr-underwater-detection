"""
Jaffe-McGlamery Turbidity Simulation

Physically-grounded underwater image degradation based on:
- McGlamery, B.L. (1975). "A Computer Model for Underwater Camera Systems"
- Jaffe, J.S. (1990). "Computer Modeling and the Design of Optimal Underwater Imaging Systems"

Turbidity levels:
- Light:  c=0.05 (clear coastal water)
- Medium: c=0.15 (moderate turbidity, river outflow)
- Heavy:  c=0.30 (high turbidity, harbor/estuarine)

These values should be CALIBRATED against UFO-120 real turbid images
using PSNR/SSIM matching. See calibrate.py.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

TURBIDITY_LEVELS = {
    "light": {"c": 0.05, "depth": 3.0},
    "medium": {"c": 0.15, "depth": 3.0},
    "heavy": {"c": 0.30, "depth": 3.0},
}


def jaffe_mcglamery_turbidity(
    image: np.ndarray,
    c: float = 0.15,
    depth: float = 3.0,
    backscatter_green: float = 0.3,
    backscatter_blue: float = 0.1,
    noise_scale: float = 0.02,
    seed: int | None = None,
) -> np.ndarray:
    """
    Apply physically-grounded turbidity simulation to an image.

    Args:
        image: Input BGR image (uint8, 0-255)
        c: Attenuation coefficient. Higher = more turbid.
        depth: Simulated water depth in meters.
        backscatter_green: Green channel backscatter intensity (0-1).
        backscatter_blue: Blue channel backscatter intensity (0-1).
        noise_scale: Base noise standard deviation for particle scatter.
        seed: Random seed for reproducible noise generation.

    Returns:
        Turbid image (uint8, 0-255)
    """
    rng = np.random.RandomState(seed)
    img = image.astype(np.float32) / 255.0

    # Beer-Lambert attenuation
    attenuation = np.exp(-c * depth)

    # Wavelength-dependent backscatter (BGR order in OpenCV)
    backscatter = np.zeros_like(img)
    backscatter[:, :, 0] = backscatter_blue * (1 - attenuation)   # Blue
    backscatter[:, :, 1] = backscatter_green * (1 - attenuation)  # Green
    backscatter[:, :, 2] = 0.05 * (1 - attenuation)              # Red (minimal)

    # Compose: attenuated direct signal + backscatter veiling light
    turbid = img * attenuation + backscatter

    # Particle scatter noise (proportional to turbidity)
    noise_std = noise_scale * c * 10
    noise = rng.normal(0, noise_std, img.shape).astype(np.float32)
    turbid = np.clip(turbid + noise, 0, 1)

    return (turbid * 255).astype(np.uint8)


def augment_dataset(
    input_dir: str,
    output_dir: str,
    levels: list[str] | None = None,
    copy_labels: bool = True,
    label_dir: str | None = None,
    seed: int = 42,
):
    """
    Apply turbidity augmentation to an entire image directory.
    Optionally copies corresponding label files (same annotations, only image changes).
    """
    if levels is None:
        levels = ["light", "medium", "heavy"]

    input_path = Path(input_dir)
    image_files = sorted(
        f for f in input_path.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )

    if not image_files:
        raise FileNotFoundError(f"No images found in {input_dir}")

    print(f"Found {len(image_files)} images in {input_dir}")

    for level in levels:
        if level not in TURBIDITY_LEVELS:
            raise ValueError(f"Unknown level: {level}")

        params = TURBIDITY_LEVELS[level]
        img_out = Path(output_dir) / level / "images"
        img_out.mkdir(parents=True, exist_ok=True)

        if copy_labels and label_dir:
            lbl_out = Path(output_dir) / level / "labels"
            lbl_out.mkdir(parents=True, exist_ok=True)

        print(f"\nGenerating {level} turbidity (c={params['c']})...")

        for i, img_path in enumerate(tqdm(image_files)):
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  WARNING: Could not read {img_path.name}, skipping")
                continue

            turbid = jaffe_mcglamery_turbidity(
                img,
                c=params["c"],
                depth=params["depth"],
                seed=seed + i,  # Deterministic per-image noise
            )
            cv2.imwrite(str(img_out / img_path.name), turbid)

            # Copy label file (bounding boxes don't change with turbidity)
            if copy_labels and label_dir:
                label_file = Path(label_dir) / f"{img_path.stem}.txt"
                if label_file.exists():
                    import shutil
                    shutil.copy2(label_file, lbl_out / label_file.name)

    print(f"\nDone. Augmented images saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jaffe-McGlamery turbidity augmentation")
    parser.add_argument("--input", required=True, help="Input image directory")
    parser.add_argument("--output", required=True, help="Output base directory")
    parser.add_argument("--labels", default=None, help="Label directory to copy")
    parser.add_argument("--levels", nargs="+", default=["light", "medium", "heavy"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    augment_dataset(
        input_dir=args.input,
        output_dir=args.output,
        levels=args.levels,
        copy_labels=args.labels is not None,
        label_dir=args.labels,
        seed=args.seed,
    )
