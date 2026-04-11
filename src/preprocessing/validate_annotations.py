"""
Annotation Validator for Trash-ICRA19

Checks for:
- Missing label files
- Malformed bounding boxes (coords outside [0,1])
- Empty label files
- Invalid class IDs
- ROV class filtering (class to ignore)

Run this BEFORE training. A single bad annotation silently degrades mAP.
"""

import argparse
from pathlib import Path

VALID_CLASSES = {0, 1, 2, 3, 4, 5, 6}  # Adjust based on your class mapping
# Exclude ROV class if present (check dataset README for class ID)
IGNORE_CLASSES = set()  # e.g., {7} for ROV


def validate_yolo_label(label_path: Path, fix: bool = False) -> dict:
    """Validate a single YOLO format label file."""
    issues = []
    fixed_lines = []

    if not label_path.exists():
        return {"path": str(label_path), "issues": ["FILE_MISSING"], "n_boxes": 0}

    with open(label_path) as f:
        lines = f.readlines()

    if not lines:
        return {"path": str(label_path), "issues": ["EMPTY_FILE"], "n_boxes": 0}

    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) != 5:
            issues.append(f"Line {i}: expected 5 values, got {len(parts)}")
            continue

        try:
            cls_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
        except ValueError:
            issues.append(f"Line {i}: non-numeric values")
            continue

        if cls_id in IGNORE_CLASSES:
            issues.append(f"Line {i}: ignored class {cls_id} (ROV)")
            continue

        if cls_id not in VALID_CLASSES:
            issues.append(f"Line {i}: invalid class {cls_id}")
            continue

        # Check bounds
        coords_valid = True
        for name, val in [("x", x_center), ("y", y_center), ("w", width), ("h", height)]:
            if val < 0 or val > 1:
                issues.append(f"Line {i}: {name}={val:.4f} outside [0,1]")
                coords_valid = False

        if width <= 0 or height <= 0:
            issues.append(f"Line {i}: zero/negative dimensions w={width}, h={height}")
            coords_valid = False

        if coords_valid:
            fixed_lines.append(line.strip())

    if fix and fixed_lines:
        with open(label_path, "w") as f:
            f.write("\n".join(fixed_lines) + "\n")

    return {
        "path": str(label_path),
        "issues": issues,
        "n_boxes": len(fixed_lines),
    }


def validate_dataset(image_dir: str, label_dir: str, fix: bool = False):
    """Validate all annotations in a dataset."""
    img_path = Path(image_dir)
    lbl_path = Path(label_dir)

    images = sorted(f for f in img_path.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"})

    total_issues = 0
    missing_labels = 0
    total_boxes = 0

    for img_file in images:
        label_file = lbl_path / f"{img_file.stem}.txt"
        result = validate_yolo_label(label_file, fix=fix)

        if result["issues"]:
            total_issues += len(result["issues"])
            if "FILE_MISSING" in result["issues"]:
                missing_labels += 1
            else:
                for issue in result["issues"]:
                    print(f"  {label_file.name}: {issue}")

        total_boxes += result["n_boxes"]

    print(f"\nValidation Summary:")
    print(f"  Images: {len(images)}")
    print(f"  Missing labels: {missing_labels}")
    print(f"  Total issues: {total_issues}")
    print(f"  Valid boxes: {total_boxes}")
    if fix:
        print(f"  (Issues were auto-fixed where possible)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--fix", action="store_true", help="Auto-fix recoverable issues")
    args = parser.parse_args()
    validate_dataset(args.images, args.labels, fix=args.fix)
