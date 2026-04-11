#!/bin/bash
# Step 2: Train vanilla RT-DETR baseline on clean data
# This gives you your Track A baseline number.

set -e

echo "=== Training Vanilla RT-DETR (Baseline) ==="
echo "Estimated time: 6-10 hours on T4 GPU"

yolo detect train \
    model=rtdetr-l.pt \
    data=configs/datasets/trash_icra19.yaml \
    epochs=50 \
    imgsz=640 \
    batch=16 \
    seed=42 \
    project=results \
    name=baseline_rtdetr_clean \
    device=0

echo ""
echo "=== Baseline training complete ==="
echo "Weights saved to: results/baseline_rtdetr_clean/weights/best.pt"
echo "Now run 03_generate_turbid.sh"
