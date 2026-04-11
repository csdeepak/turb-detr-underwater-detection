#!/bin/bash
# Step 4: CRITICAL — Prove turbidity breaks the baseline
# This is your FIRST publishable result.

set -e

BASELINE_WEIGHTS="results/baseline_rtdetr_clean/weights/best.pt"

echo "=== Track A: Clean test evaluation ==="
yolo detect val \
    model=$BASELINE_WEIGHTS \
    data=configs/datasets/trash_icra19.yaml \
    split=test

echo ""
echo "=== Track B: Turbid test evaluation ==="
for level in light medium heavy; do
    echo "--- Turbidity: $level ---"
    yolo detect val \
        model=$BASELINE_WEIGHTS \
        data=configs/datasets/trash_icra19_turbid_${level}.yaml \
        split=test
done

echo ""
echo "=== DONE ==="
echo "Record these mAP numbers. This is your motivation table."
echo "If drop < 10 mAP points: your problem statement needs rethinking."
echo "If drop > 15 mAP points: you have a publishable finding. Proceed."
