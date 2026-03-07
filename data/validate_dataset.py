"""Validate Trash-ICRA19 dataset before training.

Checks:
    1. Directory structure (images/ + labels/ per split)
    2. Image ↔ label pairing (missing files in either direction)
    3. YOLO bbox format (class_id x_center y_center width height)
    4. Corrupted / unreadable images
    5. Per-class object counts
    6. Summary statistics

Usage:
    python data/validate_dataset.py
    python data/validate_dataset.py --root /content/datasets/trash_icra19
    python data/validate_dataset.py --root ./data/trash_icra19 --fix
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
SPLITS = ("train", "val", "test")
DEFAULT_ROOT = Path(__file__).resolve().parent / "trash_icra19"
CLASS_NAMES = {0: "plastic", 1: "bottle", 2: "can", 3: "bag", 4: "net"}
NUM_CLASSES = len(CLASS_NAMES)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def log(msg: str, level: str = "INFO") -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [{level:5s}] {msg}")


def log_error(msg: str) -> None:
    log(msg, level="ERROR")


def log_warn(msg: str) -> None:
    log(msg, level="WARN")


# ─────────────────────────────────────────────────────────────
# 1. Directory structure
# ─────────────────────────────────────────────────────────────
def check_directory_structure(root: Path) -> tuple[bool, dict[str, dict[str, Path]]]:
    """Verify images/ and labels/ directories exist for each split.

    Returns (all_ok, paths_dict).
    """
    log("Checking directory structure …")
    all_ok = True
    paths: dict[str, dict[str, Path]] = {}

    for split in SPLITS:
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        paths[split] = {"images": img_dir, "labels": lbl_dir}

        if img_dir.is_dir():
            log(f"  ✓ {img_dir.relative_to(root)}")
        else:
            log_error(f"  ✗ MISSING {img_dir.relative_to(root)}")
            all_ok = False

        if lbl_dir.is_dir():
            log(f"  ✓ {lbl_dir.relative_to(root)}")
        else:
            log_error(f"  ✗ MISSING {lbl_dir.relative_to(root)}")
            all_ok = False

    return all_ok, paths


# ─────────────────────────────────────────────────────────────
# 2. Image ↔ Label pairing
# ─────────────────────────────────────────────────────────────
def check_pairing(
    img_dir: Path,
    lbl_dir: Path,
) -> tuple[list[Path], list[Path], list[Path]]:
    """Return (paired_images, images_without_labels, labels_without_images)."""
    image_files = {
        f.stem: f for f in img_dir.iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS
    }
    label_files = {
        f.stem: f for f in lbl_dir.iterdir()
        if f.suffix == ".txt"
    }

    img_stems = set(image_files)
    lbl_stems = set(label_files)

    paired = sorted(image_files[s] for s in img_stems & lbl_stems)
    imgs_missing_lbl = sorted(image_files[s] for s in img_stems - lbl_stems)
    lbls_missing_img = sorted(label_files[s] for s in lbl_stems - img_stems)

    return paired, imgs_missing_lbl, lbls_missing_img


# ─────────────────────────────────────────────────────────────
# 3. YOLO bbox validation
# ─────────────────────────────────────────────────────────────
def validate_yolo_label(
    label_path: Path,
    num_classes: int,
) -> tuple[int, list[str]]:
    """Validate a single YOLO-format label file.

    Returns (object_count, list_of_error_messages).
    """
    errors: list[str] = []
    count = 0
    text = label_path.read_text().strip()

    if not text:
        # Empty label = background image (valid in YOLO format)
        return 0, []

    for line_no, line in enumerate(text.splitlines(), start=1):
        parts = line.strip().split()
        if len(parts) != 5:
            errors.append(
                f"{label_path.name}:{line_no} — expected 5 values, got {len(parts)}: '{line.strip()}'"
            )
            continue

        try:
            cls_id = int(parts[0])
            x_c, y_c, w, h = (float(v) for v in parts[1:])
        except ValueError:
            errors.append(f"{label_path.name}:{line_no} — non-numeric value: '{line.strip()}'")
            continue

        # Class ID range
        if cls_id < 0 or cls_id >= num_classes:
            errors.append(f"{label_path.name}:{line_no} — class {cls_id} out of range [0, {num_classes - 1}]")

        # Normalised coordinates must be in (0, 1]
        for name, val in [("x_center", x_c), ("y_center", y_c), ("width", w), ("height", h)]:
            if val <= 0 or val > 1.0:
                errors.append(f"{label_path.name}:{line_no} — {name}={val:.6f} outside (0, 1]")

        count += 1

    return count, errors


# ─────────────────────────────────────────────────────────────
# 4. Corrupted image detection
# ─────────────────────────────────────────────────────────────
def check_image_integrity(image_path: Path) -> str | None:
    """Return an error string if the image is corrupted, else None."""
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return f"cv2.imread returned None: {image_path.name}"
        if img.size == 0:
            return f"Empty image array: {image_path.name}"
    except Exception as exc:
        return f"Exception reading {image_path.name}: {exc}"
    return None


# ─────────────────────────────────────────────────────────────
# 5. Per-class counting
# ─────────────────────────────────────────────────────────────
def count_classes(label_dir: Path) -> Counter:
    """Count objects per class across all label files."""
    counts: Counter = Counter()
    for lbl in label_dir.glob("*.txt"):
        text = lbl.read_text().strip()
        if not text:
            continue
        for line in text.splitlines():
            parts = line.strip().split()
            if len(parts) >= 1:
                try:
                    counts[int(parts[0])] += 1
                except ValueError:
                    continue
    return counts


# ─────────────────────────────────────────────────────────────
# Main validation pipeline
# ─────────────────────────────────────────────────────────────
def validate_dataset(
    root: Path,
    check_images: bool = True,
    max_errors: int = 20,
) -> bool:
    """Run all validation checks. Returns True if dataset is clean."""
    log("=" * 60)
    log(f"  Dataset Validation — {root.name}")
    log("=" * 60)
    log(f"Root: {root.resolve()}")
    log("")

    ok = True

    # ── 1. Directory structure ───────────────────────────────
    struct_ok, paths = check_directory_structure(root)
    if not struct_ok:
        log_error("Directory structure incomplete — cannot continue.")
        return False
    log("")

    total_images = 0
    total_labels = 0
    total_objects = 0
    total_corrupted = 0
    total_unpaired_imgs = 0
    total_unpaired_lbls = 0
    all_label_errors: list[str] = []
    global_class_counts: Counter = Counter()

    for split in SPLITS:
        img_dir = paths[split]["images"]
        lbl_dir = paths[split]["labels"]

        if not img_dir.is_dir() or not lbl_dir.is_dir():
            continue

        log(f"─── {split.upper()} ──────────────────────────────────────")

        # ── 2. Pairing ───────────────────────────────────────
        paired, imgs_no_lbl, lbls_no_img = check_pairing(img_dir, lbl_dir)
        n_imgs = len(paired) + len(imgs_no_lbl)
        n_lbls = len(paired) + len(lbls_no_img)

        total_images += n_imgs
        total_labels += n_lbls
        total_unpaired_imgs += len(imgs_no_lbl)
        total_unpaired_lbls += len(lbls_no_img)

        log(f"  Images: {n_imgs:>6,}   Labels: {n_lbls:>6,}   Paired: {len(paired):>6,}")

        if imgs_no_lbl:
            log_warn(f"  {len(imgs_no_lbl)} image(s) without labels")
            for p in imgs_no_lbl[:5]:
                log_warn(f"    → {p.name}")
            if len(imgs_no_lbl) > 5:
                log_warn(f"    … and {len(imgs_no_lbl) - 5} more")
            ok = False

        if lbls_no_img:
            log_warn(f"  {len(lbls_no_img)} label(s) without images")
            for p in lbls_no_img[:5]:
                log_warn(f"    → {p.name}")
            if len(lbls_no_img) > 5:
                log_warn(f"    … and {len(lbls_no_img) - 5} more")
            ok = False

        # ── 3. YOLO format validation ────────────────────────
        split_objects = 0
        for img_path in paired:
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            count, errors = validate_yolo_label(lbl_path, NUM_CLASSES)
            split_objects += count
            all_label_errors.extend(errors)

        total_objects += split_objects
        log(f"  Objects: {split_objects:>6,}")

        if all_label_errors:
            ok = False

        # ── 4. Corrupted images ──────────────────────────────
        if check_images:
            corrupted: list[str] = []
            all_imgs = sorted(
                f for f in img_dir.iterdir()
                if f.suffix.lower() in IMAGE_EXTENSIONS
            )
            for img_path in all_imgs:
                err = check_image_integrity(img_path)
                if err:
                    corrupted.append(err)
            if corrupted:
                total_corrupted += len(corrupted)
                log_error(f"  {len(corrupted)} corrupted image(s):")
                for msg in corrupted[:5]:
                    log_error(f"    → {msg}")
                ok = False
            else:
                log(f"  Image integrity: ✓ all readable")

        # ── 5. Class distribution ────────────────────────────
        split_counts = count_classes(lbl_dir)
        global_class_counts += split_counts
        log("")

    # ─────────────────────────────────────────────────────────
    # 6. Summary statistics
    # ─────────────────────────────────────────────────────────
    log("=" * 60)
    log("  DATASET SUMMARY")
    log("=" * 60)
    log(f"  Total images         : {total_images:>7,}")
    log(f"  Total label files    : {total_labels:>7,}")
    log(f"  Total objects        : {total_objects:>7,}")
    log(f"  Unpaired images      : {total_unpaired_imgs:>7,}")
    log(f"  Unpaired labels      : {total_unpaired_lbls:>7,}")
    log(f"  Corrupted images     : {total_corrupted:>7,}")
    log(f"  Label format errors  : {len(all_label_errors):>7,}")
    log("")

    # Class distribution table
    log("  Class Distribution:")
    log(f"  {'ID':>4s}  {'Name':15s} {'Count':>8s}  {'%':>6s}  Bar")
    log(f"  {'─'*4}  {'─'*15} {'─'*8}  {'─'*6}  {'─'*30}")
    for cls_id in range(NUM_CLASSES):
        name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
        count = global_class_counts.get(cls_id, 0)
        pct = (count / total_objects * 100) if total_objects > 0 else 0
        bar = "█" * int(pct / 2)
        log(f"  {cls_id:>4d}  {name:15s} {count:>8,}  {pct:>5.1f}%  {bar}")
    log("")

    # Label errors (truncated)
    if all_label_errors:
        shown = all_label_errors[:max_errors]
        log_error(f"Label format errors ({len(all_label_errors)} total, showing {len(shown)}):")
        for msg in shown:
            log_error(f"  → {msg}")
        log("")

    # Final verdict
    if ok:
        log("✓ Dataset validation PASSED — ready for training.")
    else:
        log_warn("⚠ Dataset validation completed WITH WARNINGS — review issues above.")

    log("=" * 60)
    return ok


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate Trash-ICRA19 dataset structure & annotations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--root", type=str, default=str(DEFAULT_ROOT),
        help="Dataset root directory",
    )
    p.add_argument(
        "--skip-image-check", action="store_true",
        help="Skip slow image corruption check (cv2.imread per file)",
    )
    p.add_argument(
        "--max-errors", type=int, default=20,
        help="Max label errors to display",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root = Path(args.root).resolve()

    if not root.is_dir():
        log_error(f"Dataset root does not exist: {root}")
        sys.exit(1)

    passed = validate_dataset(
        root=root,
        check_images=not args.skip_image_check,
        max_errors=args.max_errors,
    )
    sys.exit(0 if passed else 1)
