# Turb-DETR: Underwater Plastic Detection

Real-time underwater plastic debris detection using **RT-DETR** (Real-Time DEtection TRansformer) with turbidity-aware enhancements.

## Project Overview

This research project adapts the RT-DETR architecture for detecting plastic waste in underwater environments with varying turbidity levels. The system is designed to handle challenges unique to underwater imagery — color distortion, low contrast, light scattering, and suspended particulates.

## Repository Structure

```
turb-detr-underwater-detection/
├── augmentation/          # Underwater-specific data augmentation pipelines
│   ├── __init__.py
│   ├── underwater.py      # Turbidity, color-shift, caustic simulations
│   └── pipeline.py        # Composable augmentation pipeline
├── configs/               # YAML experiment & model configurations
│   ├── dataset.yaml       # Dataset paths and class definitions
│   ├── train_config.yaml  # Training hyperparameters
│   └── model_config.yaml  # Model architecture settings
├── data/                  # Dataset root (contents excluded from Git)
│   ├── raw/               # Original images & labels
│   ├── processed/         # Preprocessed / resized data
│   └── annotations/       # COCO-format annotation JSONs
├── evaluation/            # Metrics computation & result analysis
│   ├── __init__.py
│   ├── metrics.py         # mAP, precision, recall, F1
│   └── visualize.py       # Detection result visualization
├── models/                # Model definitions & custom modules
│   ├── backbones/         # Feature extractor variants
│   ├── heads/             # Detection head modules
│   ├── __init__.py
│   └── turb_detr.py       # Main Turb-DETR model wrapper
├── notebooks/             # Jupyter notebooks for Colab & exploration
│   └── 01_train_colab.ipynb
├── outputs/               # Runtime artifacts (excluded from Git)
│   ├── checkpoints/       # Saved model weights
│   ├── logs/              # TensorBoard / WandB logs
│   └── visualizations/    # Prediction overlays & plots
├── scripts/               # CLI entry-point scripts
│   ├── train.py           # Launch training
│   ├── evaluate.py        # Run evaluation on test set
│   └── infer.py           # Single-image / batch inference
├── training/              # Training engine & helpers
│   ├── __init__.py
│   ├── trainer.py         # Training loop orchestration
│   └── scheduler.py       # LR scheduler utilities
├── utils/                 # Shared helpers
│   ├── __init__.py
│   ├── logger.py          # Loguru-based structured logging
│   └── io_utils.py        # File I/O, path resolution, Colab checks
├── .gitignore
├── README.md
└── requirements.txt
```

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/csdeepak/turb-detr-underwater-detection.git
cd turb-detr-underwater-detection
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare Dataset

Place your dataset under `data/raw/` following YOLO or COCO format. Update paths in `configs/dataset.yaml`.

### 3. Train

```bash
python scripts/train.py --config configs/train_config.yaml
```

### 4. Evaluate

```bash
python scripts/evaluate.py --config configs/train_config.yaml --weights outputs/checkpoints/best.pt
```

### 5. Google Colab

Open `notebooks/01_train_colab.ipynb` in Colab and follow the cells. GPU runtime is auto-detected.

## Tech Stack

| Component | Library |
|---|---|
| Detection Model | RT-DETR via Ultralytics |
| Framework | PyTorch 2.1+ |
| Augmentation | Albumentations |
| Tracking | Weights & Biases / TensorBoard |
| Evaluation | pycocotools, scikit-learn |

## Requirements

- Python 3.10+
- CUDA 11.8+ (for GPU training)
- See `requirements.txt` for full dependency list

## License

This project is for academic research purposes.

## Authors

- CHENNUPATI GUNADEEP - PES University
- Deepak C S — PES University