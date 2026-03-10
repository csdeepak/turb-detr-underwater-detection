"""Carve a held-out test split from TrashCAN 1.0 training data.

TrashCAN 1.0 ships with only train + val splits.  Using val for both early-
stopping and test evaluation introduces data leakage.  This script moves a
random 20% of training images (stratified by class) into a new test split so
you can report a genuine held-out test mAP.

Run once before training:
    python scripts/convert_trashcan_split.py --root ./data/trashcan

After running, uncomment the ``test:`` line in ``configs/trashcan.yaml`` and
update evaluate.py / benchmark_models.py DATASET_CONFIGS split to "test".
"""

from __future__ import annotations

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def collect_split(
    imgs_dir: Path,
    lbls_dir: Path,
) -> list[tuple[Path, Path | None]]:
    """Return list of (image_path, label_path_or_None) for a split dir."""
    pairs: list[tuple[Path, Path | None]] = []
    for img in sorted(imgs_dir.iterdir()):
        if img.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        lbl = lbls_dir / (img.stem + ".txt")
        pairs.append((img, lbl if lbl.exists() else None))
    return pairs


def get_dominant_class(label_path: Path | None) -> int:
    """Return the most frequent class id in a label file, or -1 if none."""
    if label_path is None or not label_path.exists():
        return -1
    counts: dict[int, int] = defaultdict(int)
    for line in label_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if parts:
            counts[int(parts[0])] += 1
    return max(counts, key=counts.get) if counts else -1


def stratified_split(
    pairs: list[tuple[Path, Path | None]],
    test_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[tuple[Path, Path | None]], list[tuple[Path, Path | None]]]:
    """Split pairs into (train_keep, test) using per-class stratification."""
    rng = random.Random(seed)

    # Group by dominant class
    buckets: dict[int, list[tuple[Path, Path | None]]] = defaultdict(list)
    for pair in pairs:
        cls = get_dominant_class(pair[1])
        buckets[cls].append(pair)

    train_keep: list[tuple[Path, Path | None]] = []
    test: list[tuple[Path, Path | None]] = []

    for cls_id, cls_pairs in buckets.items():
        rng.shuffle(cls_pairs)
        n_test = max(1, int(len(cls_pairs) * test_fraction))
        test.extend(cls_pairs[:n_test])
        train_keep.extend(cls_pairs[n_test:])

    return train_keep, test


def move_to_split(
    pairs: list[tuple[Path, Path | None]],
    dst_imgs: Path,
    dst_lbls: Path,
    copy: bool = False,
) -> int:
    """Copy or move image+label pairs to destination directories."""
    dst_imgs.mkdir(parents=True, exist_ok=True)
    dst_lbls.mkdir(parents=True, exist_ok=True)

    op = shutil.copy2 if copy else shutil.move
    moved = 0
    for img_path, lbl_path in pairs:
        op(str(img_path), dst_imgs / img_path.name)
        if lbl_path is not None and lbl_path.exists():
            op(str(lbl_path), dst_lbls / lbl_path.name)
        moved += 1
    return moved


def run(root: Path, test_fraction: float, seed: int, copy: bool) -> None:
    train_imgs = root / "images" / "train"
    train_lbls = root / "labels" / "train"
    test_imgs  = root / "images" / "test"
    test_lbls  = root / "labels" / "test"

    if not train_imgs.is_dir():
        raise FileNotFoundError(f"Train images not found: {train_imgs}")

    if test_imgs.is_dir() and any(test_imgs.iterdir()):
        print(f"[WARN] {test_imgs} already exists and is non-empty — aborting.")
        print("       Delete it first if you want to re-split.")
        return

    pairs = collect_split(train_imgs, train_lbls)
    if not pairs:
        raise FileNotFoundError(f"No images found in {train_imgs}")

    print(f"Found {len(pairs)} training samples.")

    train_keep, test_pairs = stratified_split(pairs, test_fraction, seed)

    print(f"Keeping {len(train_keep)} in train, moving {len(test_pairs)} to test.")
    print(f"({'copy' if copy else 'move'} mode)")

    n = move_to_split(test_pairs, test_imgs, test_lbls, copy=copy)
    print(f"✓ {n} samples written to {test_imgs}")

    if not copy:
        # Remove empty old label files if any were moved
        print(f"Training split now has {len(train_keep)} samples.")

    print(
        "\nNext steps:\n"
        "  1. Uncomment 'test: images/test' in configs/trashcan.yaml\n"
        "  2. Update DATASET_CONFIGS split to 'test' in evaluate.py and benchmark_models.py\n"
        "  3. Do NOT train on these test images."
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Carve a held-out test split from TrashCAN training data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--root", type=str, default="./data/trashcan",
                   help="TrashCAN YOLO dataset root")
    p.add_argument("--test-fraction", type=float, default=0.2,
                   help="Fraction of training data to move to test (0–1)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility")
    p.add_argument("--copy", action="store_true", default=False,
                   help="Copy instead of move (preserve original train split)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        root=Path(args.root),
        test_fraction=args.test_fraction,
        seed=args.seed,
        copy=args.copy,
    )
