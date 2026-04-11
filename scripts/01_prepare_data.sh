#!/bin/bash
# Step 1: Prepare Trash-ICRA19 dataset
# Run ONCE. Commit data/splits/ to git immediately after.

set -e

echo "=== Validating annotations ==="
python src/preprocessing/validate_annotations.py \
    --images data/raw/trash-icra19/images \
    --labels data/raw/trash-icra19/labels \
    --fix

echo ""
echo "=== Creating stratified split ==="
python src/preprocessing/prepare_dataset.py \
    --images data/raw/trash-icra19/images \
    --labels data/raw/trash-icra19/labels \
    --output data/processed \
    --seed 42

echo ""
echo "=== Checking for data leaks ==="
python src/utils/data_leak_check.py \
    --train data/splits/train.txt \
    --val data/splits/val.txt \
    --test data/splits/test.txt

echo ""
echo "DONE. Now commit data/splits/ to git."
