"""Convert RUIE-OD annotations to YOLO format for Ultralytics training/evaluation.

RUIE (Real-world Underwater Image Enhancement) detection subset (RUIE-OD)
uses a custom XML annotation format.  This script converts those annotations
to YOLO .txt format and organises the dataset into the standard YOLO layout:

    data/ruie/
        images/
            val/
            test/
        labels/
            val/
            test/

Source structure expected under --root:
    <root>/
        RUIE_OD/
            images/    (or JPEGImages/)
            Annotations/   (Pascal VOC .xml files)

RUIE-OD has no train split — annotations exist for val and test only.

Usage
-----
    python scripts/convert_ruie.py --root /path/to/RUIE --out ./data/ruie

    # With explicit annotation dir:
    python scripts/convert_ruie.py \\
        --root /path/to/RUIE_OD \\
        --out  ./data/ruie \\
        --split test

Class mapping
-------------
RUIE-OD uses these categories (subset used here, mapped to Trash-ICRA19 IDs):
    holothurian (sea cucumber) → dropped (no Trash-ICRA19 equivalent)
    echinus      (sea urchin)  → dropped
    scallop                    → dropped
    starfish                   → dropped

    If RUIE-OD only contains turbidity images without relevant trash classes,
    this converter will produce label files but annotation counts may be zero.
    In that case RUIE is useful only for cross-domain turbidity robustness
    testing (run infer.py over the images, do NOT expect detection mAP).
"""

from __future__ import annotations

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# Class mapping: RUIE-OD label string → Trash-ICRA19 class id
# Add / modify entries here if your RUIE subset uses different labels.
# ─────────────────────────────────────────────────────────────
RUIE_CLASS_MAP: dict[str, int] = {
    "plastic":    0,
    "bottle":     1,
    "can":        2,
    "bag":        3,
    "net":        4,
    "garbage":    0,   # generic → plastic
    "waste":      0,
    "debris":     0,
}

# Labels that should be silently skipped (marine fauna, not trash)
RUIE_SKIP_LABELS = {
    "holothurian", "echinus", "scallop", "starfish", "fish",
    "seacucumber", "sea_cucumber", "urchin",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


# ─────────────────────────────────────────────────────────────
# Pascal VOC XML → YOLO txt
# ─────────────────────────────────────────────────────────────
def convert_xml_to_yolo(xml_path: Path) -> tuple[list[str], int, int]:
    """Parse a Pascal VOC XML annotation and return YOLO lines.

    Returns
    -------
    lines : list[str]
        YOLO format lines: "class_id cx cy w h" (normalised).
    img_w : int
        Image width from annotation.
    img_h : int
        Image height from annotation.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    if size is None:
        raise ValueError(f"No <size> element in {xml_path}")
    img_w = int(size.findtext("width", "0"))
    img_h = int(size.findtext("height", "0"))
    if img_w == 0 or img_h == 0:
        raise ValueError(f"Zero image dimensions in {xml_path}")

    lines: list[str] = []
    for obj in root.findall("object"):
        label = (obj.findtext("name") or "").strip().lower()
        if label in RUIE_SKIP_LABELS:
            continue
        cls_id = RUIE_CLASS_MAP.get(label)
        if cls_id is None:
            # Unknown label — skip with a warning (don't crash)
            print(f"  [WARN] Unknown label '{label}' in {xml_path.name} — skipped.")
            continue

        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        xmin = float(bndbox.findtext("xmin", "0"))
        ymin = float(bndbox.findtext("ymin", "0"))
        xmax = float(bndbox.findtext("xmax", "0"))
        ymax = float(bndbox.findtext("ymax", "0"))

        # Clamp to image bounds
        xmin = max(0.0, min(xmin, img_w))
        xmax = max(0.0, min(xmax, img_w))
        ymin = max(0.0, min(ymin, img_h))
        ymax = max(0.0, min(ymax, img_h))

        if xmax <= xmin or ymax <= ymin:
            print(f"  [WARN] Degenerate box in {xml_path.name} — skipped.")
            continue

        cx = (xmin + xmax) / 2.0 / img_w
        cy = (ymin + ymax) / 2.0 / img_h
        bw = (xmax - xmin) / img_w
        bh = (ymax - ymin) / img_h
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    return lines, img_w, img_h


# ─────────────────────────────────────────────────────────────
# Locate RUIE sub-directories
# ─────────────────────────────────────────────────────────────
def find_ruie_dirs(root: Path) -> tuple[Path, Path]:
    """Return (image_dir, annotation_dir) from the RUIE-OD root."""
    # Check for RUIE_OD sub-folder first
    for sub in ("RUIE_OD", "ruie_od", "detection", "."):
        candidate = root / sub if sub != "." else root
        if not candidate.exists():
            continue
        img_dir = next(
            (candidate / d for d in ("images", "JPEGImages", "imgs") if (candidate / d).exists()),
            None,
        )
        ann_dir = next(
            (candidate / d for d in ("Annotations", "annotations", "labels", "xml")
             if (candidate / d).exists()),
            None,
        )
        if img_dir and ann_dir:
            return img_dir, ann_dir

    raise FileNotFoundError(
        f"Could not find image and annotation directories under {root}. "
        "Expected: images/ (or JPEGImages/) and Annotations/ (or annotations/)."
    )


# ─────────────────────────────────────────────────────────────
# Main conversion
# ─────────────────────────────────────────────────────────────
def convert(
    root: Path,
    out: Path,
    split: str = "test",
    val_fraction: float = 0.2,
) -> None:
    """Convert RUIE-OD to YOLO layout.

    Parameters
    ----------
    root : Path
        Path to the RUIE-OD dataset root.
    out : Path
        Output dataset root (data/ruie/).
    split : str
        Which split to assign all images to (``"test"`` or ``"val"``).
        If ``"auto"``, the first 80% of images become ``val``, last 20% become ``test``.
    val_fraction : float
        Only used when split="auto". Fraction for val (remainder → test).
    """
    img_dir, ann_dir = find_ruie_dirs(root)
    print(f"Images      : {img_dir}")
    print(f"Annotations : {ann_dir}")

    xml_files = sorted(ann_dir.glob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No .xml annotation files found in {ann_dir}")

    print(f"Found {len(xml_files)} annotation files.")

    if split == "auto":
        n_val = max(1, int(len(xml_files) * val_fraction))
        splits_map: dict[str, list[Path]] = {
            "val":  xml_files[:n_val],
            "test": xml_files[n_val:],
        }
    else:
        splits_map = {split: xml_files}

    total_imgs = total_labels = 0

    for split_name, files in splits_map.items():
        imgs_dst = out / "images" / split_name
        lbls_dst = out / "labels" / split_name
        imgs_dst.mkdir(parents=True, exist_ok=True)
        lbls_dst.mkdir(parents=True, exist_ok=True)

        n_i = n_l = 0
        for xml_path in files:
            stem = xml_path.stem

            # Find corresponding image
            img_src = None
            for ext in IMAGE_EXTENSIONS:
                candidate = img_dir / (stem + ext)
                if candidate.exists():
                    img_src = candidate
                    break

            if img_src is None:
                print(f"  [WARN] No image found for {xml_path.name} — skipped.")
                continue

            # Convert annotation
            try:
                yolo_lines, _, _ = convert_xml_to_yolo(xml_path)
            except Exception as exc:
                print(f"  [WARN] Skipping {xml_path.name}: {exc}")
                continue

            # Copy image
            shutil.copy2(img_src, imgs_dst / img_src.name)
            n_i += 1

            # Write label (always write file, even if empty — Ultralytics expects it)
            lbl_path = lbls_dst / (stem + ".txt")
            lbl_path.write_text("\n".join(yolo_lines))
            if yolo_lines:
                n_l += 1

        print(f"  {split_name:6s} → {n_i:>5} images  |  {n_l:>5} labelled images")
        total_imgs  += n_i
        total_labels += n_l

    print(f"\n✓ RUIE conversion complete → {out}")
    print(f"  Total: {total_imgs} images, {total_labels} images with detection labels")
    if total_labels == 0:
        print(
            "\n⚠  ZERO detection labels written.  This is expected if your RUIE subset only\n"
            "   contains marine fauna (holothurian, echinus, etc.) which are excluded from\n"
            "   the Trash-ICRA19 class mapping.  RUIE can still be used for turbidity\n"
            "   robustness visualisation — just not for detection mAP evaluation."
        )


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert RUIE-OD Pascal VOC annotations to YOLO format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", type=str, required=True, help="Path to RUIE-OD dataset root")
    parser.add_argument("--out",  type=str, default="./data/ruie", help="Output directory")
    parser.add_argument(
        "--split", type=str, default="test",
        choices=["test", "val", "auto"],
        help="Split to assign images to. 'auto' splits 80/20 val/test.",
    )
    args = parser.parse_args()
    convert(Path(args.root), Path(args.out), split=args.split)
