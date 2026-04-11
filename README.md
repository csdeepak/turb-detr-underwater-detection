# Turb-DETR: Turbidity-Aware Real-Time Transformer Detector for Underwater Plastic Debris

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **SimAM-Injected RT-DETR for Plastic Debris Detection in Turbid AUV Environments**

## Overview

Autonomous Underwater Vehicles (AUVs) equipped with visual detection systems suffer a systematic, turbidity-induced accuracy collapse when deployed in real coastal and deep-water environments. Standard object detectors trained on clear-water imagery fail catastrophically when encountering murky conditions caused by suspended particles, phytoplankton blooms, and sediment disturbance.

**Turb-DETR** addresses this by injecting a parameter-free SimAM attention module between the CNN intra-scale encoder and the Transformer cross-scale encoder of RT-DETR, suppressing turbidity noise before the self-attention stage contaminates feature representations globally.

### Key Contributions

1. **First domain adaptation of RT-DETR** to underwater plastic debris detection
2. **SimAM injection between CNN and Transformer encoder stages** — parameter-free turbidity noise suppression
3. **Jaffe-McGlamery turbidity augmentation** — physically grounded simulation replacing naive Gaussian blur
4. **Multi-model benchmark** under controlled turbidity conditions (YOLOv9/v10, Vanilla RT-DETR, Turb-DETR)
5. **Edge deployment validation** on AUV-class hardware

## Project Structure

```
turb-detr/
├── configs/                    # Training & evaluation configurations
│   ├── train_baseline.yaml     # Vanilla RT-DETR training config
│   ├── train_turbdetr.yaml     # Turb-DETR training config
│   ├── train_yolo_baseline.yaml
│   ├── datasets/
│   │   ├── trash_icra19.yaml   # Dataset config for Trash-ICRA19
│   │   └── trash_icra19_turbid.yaml
│   └── augmentation.yaml       # Jaffe-McGlamery parameters
│
├── src/                        # Source code
│   ├── models/
│   │   ├── simam.py            # SimAM attention module
│   │   ├── turb_detr.py        # Turb-DETR integration with RT-DETR
│   │   └── __init__.py
│   ├── augmentation/
│   │   ├── jaffe_mcglamery.py  # Physically-grounded turbidity simulation
│   │   ├── calibrate.py        # PSNR/SSIM calibration against UFO-120
│   │   └── __init__.py
│   ├── preprocessing/
│   │   ├── prepare_dataset.py  # Download, validate, split Trash-ICRA19
│   │   ├── validate_annotations.py  # Check for broken bboxes, format issues
│   │   └── __init__.py
│   ├── evaluation/
│   │   ├── evaluate.py         # Unified evaluation across all tracks
│   │   ├── statistical_tests.py # McNemar's test, bootstrap CI
│   │   ├── visualize_results.py # Attention maps, qualitative comparisons
│   │   └── __init__.py
│   └── utils/
│       ├── seed.py             # Reproducibility utilities
│       ├── metrics.py          # Custom metric helpers
│       ├── data_leak_check.py  # Verify no test images in training set
│       └── __init__.py
│
├── scripts/                    # Executable scripts (entry points)
│   ├── 01_prepare_data.sh      # Download + validate + split
│   ├── 02_train_baseline.sh    # Train vanilla RT-DETR
│   ├── 03_generate_turbid.sh   # Apply Jaffe-McGlamery to test set
│   ├── 04_evaluate_collapse.sh # Evaluate baseline on turbid data
│   ├── 05_train_turbdetr.sh    # Train Turb-DETR
│   ├── 06_train_yolo.sh        # Train YOLO baselines
│   ├── 07_ablation.sh          # Run ablation studies
│   └── 08_full_benchmark.sh    # Final 3-seed full evaluation
│
├── data/                       # Data directory (NOT committed to git)
│   ├── raw/                    # Original downloaded datasets
│   ├── processed/              # Cleaned + formatted datasets
│   ├── splits/                 # train.txt, val.txt, test.txt (COMMITTED)
│   └── augmented/              # Turbidity-augmented images
│       ├── light/              # c=0.05
│       ├── medium/             # c=0.15
│       └── heavy/              # c=0.30
│
├── datasets/                   # Ultralytics dataset YAML configs (symlinked)
├── models/
│   ├── checkpoints/            # Saved model weights
│   └── exports/                # TensorRT / ONNX exports
│
├── results/                    # Experiment outputs
│   ├── figures/                # Generated plots and charts
│   ├── tables/                 # CSV result tables
│   └── qualitative/            # Detection visualizations
│
├── notebooks/                  # Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb
│   ├── 02_turbidity_visualization.ipynb
│   ├── 03_results_analysis.ipynb
│   └── 04_paper_figures.ipynb
│
├── paper/                      # Paper-related files
│   ├── latex/                  # LaTeX source
│   └── figures/                # Publication-quality figures
│
├── tests/                      # Unit tests
│   ├── test_simam.py
│   ├── test_augmentation.py
│   └── test_data_integrity.py
│
├── .gitignore
├── LICENSE
├── requirements.txt
├── setup.py
└── README.md
```

## Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/your-username/turb-detr.git
cd turb-detr

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Set reproducibility seeds
export TURB_DETR_SEED=42
```

### 2. Data Preparation

```bash
# Download and prepare Trash-ICRA19
python src/preprocessing/prepare_dataset.py \
    --dataset trash-icra19 \
    --output data/processed \
    --split-ratio 0.70 0.15 0.15 \
    --seed 42

# Validate annotations
python src/preprocessing/validate_annotations.py \
    --data-dir data/processed \
    --fix-errors
```

### 3. Train Baseline (Week 1)

```bash
# Train vanilla RT-DETR on clean Trash-ICRA19
python -m ultralytics detect train \
    model=rtdetr-l.pt \
    data=configs/datasets/trash_icra19.yaml \
    epochs=50 \
    imgsz=640 \
    batch=16 \
    seed=42 \
    project=results \
    name=baseline_rtdetr_clean
```

### 4. Prove Turbidity Collapse (Week 1 — Critical)

```bash
# Generate turbid test images at 3 severity levels
python src/augmentation/jaffe_mcglamery.py \
    --input data/processed/test/images \
    --output data/augmented \
    --levels light medium heavy

# Evaluate baseline model on all conditions
python src/evaluation/evaluate.py \
    --model results/baseline_rtdetr_clean/weights/best.pt \
    --clean-test data/processed/test \
    --turbid-test data/augmented \
    --output results/tables/turbidity_collapse.csv
```

### 5. Train Turb-DETR

```bash
# Train Turb-DETR (SimAM-injected RT-DETR) on augmented data
python -m ultralytics detect train \
    model=configs/train_turbdetr.yaml \
    data=configs/datasets/trash_icra19.yaml \
    epochs=150 \
    imgsz=640 \
    batch=16 \
    seed=42 \
    project=results \
    name=turbdetr_augmented
```

## Experimental Design

### Evaluation Tracks

| Track | Purpose | Train Data | Test Data | Metric |
|-------|---------|-----------|-----------|--------|
| **A — Clean** | Baseline control | Clean Trash-ICRA19 | Clean Trash-ICRA19 | mAP@0.5 |
| **B — Synthetic Turbid** | Turbidity collapse proof | Clean Trash-ICRA19 | Turbid Trash-ICRA19 (3 levels) | mAP@0.5 |
| **C — Real-World** | Generalization validation | Clean Trash-ICRA19 | UFO-120 (real turbid) | Qualitative |

### Models Compared

| Model | Description | Role |
|-------|-------------|------|
| YOLOv9/v10 | CNN-based detector | Baseline |
| Vanilla RT-DETR | Unmodified transformer detector | Baseline |
| **Turb-DETR** | SimAM-injected RT-DETR | **Proposed** |

### Ablation Study

| Variant | SimAM | Turbidity Aug | Purpose |
|---------|-------|---------------|---------|
| Vanilla RT-DETR | ✗ | ✗ | Control |
| RT-DETR + Aug | ✗ | ✓ | Isolate augmentation effect |
| RT-DETR + SimAM | ✓ | ✗ | Isolate SimAM effect |
| **Turb-DETR (Full)** | ✓ | ✓ | Combined effect |

## Reproducibility

Every experiment is reproducible:

- **Seeds**: All random seeds fixed (`torch`, `numpy`, `random`, `CUDA`)
- **Splits**: `data/splits/` contains frozen `train.txt`, `val.txt`, `test.txt`
- **Configs**: All hyperparameters in version-controlled YAML files
- **Tracking**: Weights & Biases integration for experiment logging
- **Data leak check**: Automated verification that no test image enters training

```bash
# Verify data integrity before any training run
python src/utils/data_leak_check.py \
    --train data/splits/train.txt \
    --val data/splits/val.txt \
    --test data/splits/test.txt
```

## Datasets

| Dataset | Size | Role | Source |
|---------|------|------|--------|
| [Trash-ICRA19](https://conservancy.umn.edu/handle/11299/214366) | 5,700+ images | Primary train/test | Fulton et al., 2019 |
| [UFO-120](http://irvlab.cs.umn.edu/resources/ufo-120-dataset) | 1,500+ pairs | Real turbidity validation | Islam et al., 2020 |
| [TrashCan 1.0](https://conservancy.umn.edu/) | 7,000+ images | Generalization testing | — |

## Hardware Requirements

| Stage | Minimum | Recommended |
|-------|---------|-------------|
| Training | NVIDIA T4 (16GB) | NVIDIA A100 (40GB) |
| Inference | NVIDIA Jetson Nano (4GB) | Jetson Orin Nano |
| Development | 16GB RAM, 50GB disk | 32GB RAM, 100GB disk |

## Citation

```bibtex
@article{turbdetr2026,
    title={Turb-DETR: A Turbidity-Aware Real-Time Transformer Detector
           for Plastic Debris Monitoring in Autonomous Underwater Systems},
    author={Deepak, C S and Gunadeep, CH and Anirudh and
            Reddy, Dareddy Devesh and Chandrashekar},
    journal={TBD},
    year={2026}
}
```

## Team

| Name | SRN | Role |
|------|-----|------|
| C S Deepak | PES1UG23CS907 | — |
| CH Gunadeep | PES1UG23CS160 | — |
| Anirudh | PES1UG23CS077 | — |
| Dareddy Devesh Reddy | PES1UG23CS171 | — |
| Chandrashekar | PES1UG23CS149 | — |

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for the RT-DETR implementation
- [Trash-ICRA19](https://conservancy.umn.edu/handle/11299/214366) dataset by Fulton et al.
- [UFO-120](http://irvlab.cs.umn.edu/resources/ufo-120-dataset) dataset by Islam et al.
- SimAM attention module by Yang et al. (2021)
