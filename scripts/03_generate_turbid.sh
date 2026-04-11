#!/bin/bash
# Step 3: Generate turbid versions of the TEST set
# Labels stay the same — only images change.

set -e

echo "=== Generating turbid test images ==="

python src/augmentation/jaffe_mcglamery.py \
    --input data/processed/test/images \
    --output data/augmented \
    --labels data/processed/test/labels \
    --levels light medium heavy \
    --seed 42

echo ""
echo "=== Done. Turbid test images saved to data/augmented/ ==="
echo "Now run 04_evaluate_collapse.sh"
