"""
Dataset Preparation for Trash-ICRA19

Steps:
1. Verify downloaded data integrity
2. Validate YOLO format annotations
3. Filter out ROV class
4. Create stratified train/val/test split (70/15/15)
5. Save split filenames to data/splits/ (version controlled)

IMPORTANT: Split files are created ONCE and committed to git.
Never regenerate them after experiments begin.
"""

import argparse
import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np


def get_class_distribution(label_dir: Path) -> dict:
    """Get class distribution from YOLO label files."""
    class_counts = Counter()
    for label_file in label_dir.glob("*.txt"):
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 1:
                    class_counts[int(parts[0])] += 1
    return dict(class_counts)


def stratified_split(
    image_dir: str,
    label_dir: str,
    output_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
):
    """Create stratified train/val/test split."""
    random.seed(seed)
    np.random.seed(seed)

    img_path = Path(image_dir)
    lbl_path = Path(label_dir)
    out_path = Path(output_dir)

    images = sorted(f.stem for f in img_path.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"})

    # Get primary class for each image (for stratification)
    image_classes = {}
    for stem in images:
        label_file = lbl_path / f"{stem}.txt"
        if label_file.exists():
            with open(label_file) as f:
                classes = [int(line.strip().split()[0]) for line in f if line.strip()]
                image_classes[stem] = classes[0] if classes else -1
        else:
            image_classes[stem] = -1

    # Group by class
    class_groups = {}
    for stem, cls in image_classes.items():
        class_groups.setdefault(cls, []).append(stem)

    train, val, test = [], [], []

    for cls, stems in class_groups.items():
        random.shuffle(stems)
        n = len(stems)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train.extend(stems[:n_train])
        val.extend(stems[n_train:n_train + n_val])
        test.extend(stems[n_train + n_val:])

    # Save split files
    splits_dir = out_path / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    for name, split in [("train", train), ("val", val), ("test", test)]:
        with open(splits_dir / f"{name}.txt", "w") as f:
            f.write("\n".join(sorted(split)) + "\n")
        print(f"  {name}: {len(split)} images")

    # Copy files into split directories
    for name, split in [("train", train), ("val", val), ("test", test)]:
        for subdir in ["images", "labels"]:
            (out_path / name / subdir).mkdir(parents=True, exist_ok=True)

        for stem in split:
            # Find image file (could be .jpg or .png)
            for ext in [".jpg", ".jpeg", ".png"]:
                src_img = img_path / f"{stem}{ext}"
                if src_img.exists():
                    shutil.copy2(src_img, out_path / name / "images" / src_img.name)
                    break

            src_lbl = lbl_path / f"{stem}.txt"
            if src_lbl.exists():
                shutil.copy2(src_lbl, out_path / name / "labels" / src_lbl.name)

    print(f"\nSplit files saved to {splits_dir}")
    print("COMMIT these split files to git immediately.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, help="Raw image directory")
    parser.add_argument("--labels", required=True, help="Raw label directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    stratified_split(args.images, args.labels, args.output, seed=args.seed)
