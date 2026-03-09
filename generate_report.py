"""Generate a detailed PDF project report for Turb-DETR."""
from __future__ import annotations
import textwrap
from fpdf import FPDF

class ReportPDF(FPDF):
    MARGIN = 15
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 6, "Turb-DETR Underwater Plastic Detection - Project Report", align="C")
            self.ln(8)
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

def build_report(out_path: str = "Turb_DETR_Project_Report.pdf"):
    pdf = ReportPDF("P", "mm", "A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    W = 210 - 2 * ReportPDF.MARGIN

    def section(title: str):
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(25, 60, 120)
        pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        pdf.set_draw_color(25, 60, 120)
        pdf.line(ReportPDF.MARGIN, pdf.get_y(), 210 - ReportPDF.MARGIN, pdf.get_y())
        pdf.ln(3)

    def subsection(title: str):
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(40, 40, 40)
        pdf.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)

    def body(text: str):
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(30, 30, 30)
        pdf.multi_cell(W, 5, text)
        pdf.ln(2)

    def bullet(text: str):
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(5)
        pdf.cell(5, 5, chr(8226))
        pdf.multi_cell(W - 10, 5, text)
        pdf.ln(1)

    def code_block(text: str):
        pdf.set_font("Courier", "", 8)
        pdf.set_fill_color(240, 240, 245)
        pdf.set_text_color(30, 30, 30)
        for line in text.strip().split("\n"):
            pdf.cell(W, 4, "  " + line, fill=True, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

    def table(headers, rows, col_widths=None):
        if col_widths is None:
            col_widths = [W / len(headers)] * len(headers)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_fill_color(25, 60, 120)
        pdf.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            pdf.cell(col_widths[i], 7, h, border=1, fill=True, align="C")
        pdf.ln()
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(30, 30, 30)
        alt = False
        for row in rows:
            if alt:
                pdf.set_fill_color(245, 245, 250)
            else:
                pdf.set_fill_color(255, 255, 255)
            for i, val in enumerate(row):
                pdf.cell(col_widths[i], 6, str(val), border=1, fill=True, align="C")
            pdf.ln()
            alt = not alt
        pdf.ln(3)

    # ═══════════════════════════════════════════════════════════
    # COVER PAGE
    # ═══════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.ln(50)
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(25, 60, 120)
    pdf.cell(0, 15, "Turb-DETR", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 16)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 10, "Underwater Plastic Detection", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 10, "using Turbidity-Aware RT-DETR", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(15)
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 8, "Project Progress Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "Date: March 8, 2026", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "Course: AISD - PES University, Semester 6", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(20)
    pdf.set_draw_color(25, 60, 120)
    pdf.set_line_width(0.5)
    pdf.line(60, pdf.get_y(), 150, pdf.get_y())
    pdf.ln(8)
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, "GitHub: github.com/csdeepak/turb-detr-underwater-detection", align="C", new_x="LMARGIN", new_y="NEXT")

    # ═══════════════════════════════════════════════════════════
    # TABLE OF CONTENTS
    # ═══════════════════════════════════════════════════════════
    pdf.add_page()
    section("Table of Contents")
    toc_items = [
        ("1.", "Executive Summary"),
        ("2.", "Project Overview & Motivation"),
        ("3.", "Technology Stack"),
        ("4.", "Repository Structure"),
        ("5.", "Module-by-Module Detailed Explanation"),
        ("  5.1", "Configuration Files"),
        ("  5.2", "Models - Turb-DETR Architecture"),
        ("  5.3", "Models - SimAM Attention Module"),
        ("  5.4", "Augmentation Pipeline"),
        ("  5.5", "Turbidity Augmentation (Physics-Based)"),
        ("  5.6", "Training Scripts"),
        ("  5.7", "Evaluation & Benchmarking"),
        ("  5.8", "Data Validation"),
        ("  5.9", "Utility Modules"),
        ("  5.10", "CLI Entry Points"),
        ("  5.11", "Colab Training Notebook"),
        ("6.", "Architecture Deep Dive"),
        ("7.", "Dataset: Trash-ICRA19"),
        ("8.", "Current Project Status"),
        ("9.", "Future Work & Next Steps"),
        ("10.", "Appendix: File Inventory"),
    ]
    for num, title in toc_items:
        pdf.set_font("Helvetica", "B" if not num.startswith(" ") else "", 10)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(12, 6, num)
        pdf.cell(0, 6, title, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # ═══════════════════════════════════════════════════════════
    # 1. EXECUTIVE SUMMARY
    # ═══════════════════════════════════════════════════════════
    pdf.add_page()
    section("1. Executive Summary")
    body(
        "This report documents the current state of the Turb-DETR project -- a research-grade "
        "deep learning system for detecting plastic debris in underwater environments. The project "
        "adapts the RT-DETR (Real-Time DEtection TRansformer) architecture with a novel "
        "turbidity-aware attention mechanism (SimAM) to handle the unique visual challenges of "
        "underwater imagery: color distortion, light scattering, low contrast, and suspended "
        "particulates."
    )
    body(
        "As of March 8, 2026, the project has completed the full software scaffold, all core "
        "model architectures, data augmentation pipelines, training scripts, evaluation/benchmarking "
        "tools, dataset configuration, and a Google Colab training notebook. The codebase is "
        "production-structured, version-controlled on GitHub, and ready for the training & "
        "experimentation phase."
    )

    subsection("Key Accomplishments")
    accomplishments = [
        "Complete project repository with modular Python 3.10+ architecture",
        "Turb-DETR model: RT-DETR + SimAM turbidity suppression (zero extra parameters)",
        "Physics-based turbidity augmentation (Beer-Lambert law, backscatter, forward-scatter)",
        "Baseline RT-DETR training pipeline with Ultralytics API integration",
        "Multi-dataset evaluation framework (Trash-ICRA19, TrashCan, RUIE)",
        "Multi-model benchmarking suite (YOLOv10 vs RT-DETR vs Turb-DETR)",
        "Comprehensive dataset validation tool (6 automated checks)",
        "Google Colab notebook for end-to-end training on free GPU",
        "All code pushed to GitHub with clean commit history",
    ]
    for a in accomplishments:
        bullet(a)

    # ═══════════════════════════════════════════════════════════
    # 2. PROJECT OVERVIEW
    # ═══════════════════════════════════════════════════════════
    pdf.add_page()
    section("2. Project Overview & Motivation")
    body(
        "Ocean plastic pollution is one of the most pressing environmental challenges. Over 14 million "
        "tons of plastic enter the ocean annually. Autonomous underwater vehicles (AUVs) equipped with "
        "cameras can survey debris on the seafloor, but existing detection models struggle with "
        "turbidity -- reduced visibility caused by suspended sediment, algae, and particulate matter."
    )
    body(
        "Standard object detectors (YOLO, Faster R-CNN) are trained on clear-air imagery and perform "
        "poorly when visual quality degrades. Our hypothesis is that injecting a turbidity-aware "
        "attention mechanism between the CNN feature extractor and the transformer encoder can help "
        "the model learn to suppress degraded features and amplify salient object signals."
    )

    subsection("Research Objectives")
    objectives = [
        "Adapt RT-DETR for underwater plastic detection with turbidity robustness",
        "Integrate SimAM (parameter-free attention) for turbidity feature suppression",
        "Develop physics-based turbidity augmentation for training data diversity",
        "Benchmark against baseline detectors (YOLOv10, vanilla RT-DETR)",
        "Validate on multiple underwater datasets (Trash-ICRA19, TrashCan, RUIE)",
    ]
    for o in objectives:
        bullet(o)

    # ═══════════════════════════════════════════════════════════
    # 3. TECHNOLOGY STACK
    # ═══════════════════════════════════════════════════════════
    section("3. Technology Stack")
    table(
        ["Component", "Technology", "Version"],
        [
            ["Deep Learning", "PyTorch", ">= 2.2.0"],
            ["Detection API", "Ultralytics (RT-DETR/YOLO)", ">= 8.3.0"],
            ["Image Processing", "OpenCV (headless)", ">= 4.10.0"],
            ["Numerical", "NumPy", ">= 1.26.0"],
            ["Visualization", "Matplotlib", ">= 3.9.0"],
            ["Experiment Tracking", "Weights & Biases (wandb)", ">= 0.18.0"],
            ["Model Profiling", "thop", ">= 0.1.1"],
            ["Configuration", "PyYAML", ">= 6.0.2"],
            ["Progress Bars", "tqdm", ">= 4.67.0"],
            ["Logging", "Loguru", "latest"],
            ["Language", "Python", ">= 3.10"],
            ["GPU", "CUDA", "11.8+ / 12.x"],
        ],
        [55, 70, 55],
    )

    # ═══════════════════════════════════════════════════════════
    # 4. REPOSITORY STRUCTURE
    # ═══════════════════════════════════════════════════════════
    pdf.add_page()
    section("4. Repository Structure")
    body("The project follows a clean, modular layout designed for ML research reproducibility:")
    tree = """\
turb-detr-underwater-detection/
+-- augmentation/             # Data augmentation modules
|   +-- __init__.py
|   +-- pipeline.py           # Composable Albumentations pipeline
|   +-- turbidity_aug.py      # Physics-based turbidity simulation
|   +-- underwater.py         # Color shift, caustics, fog overlays
+-- configs/                  # YAML configuration files
|   +-- dataset.yaml          # Generic dataset config
|   +-- model_config.yaml     # Architecture hyperparameters
|   +-- train_config.yaml     # Training hyperparameters
|   +-- trash_icra19.yaml     # Trash-ICRA19 dataset definition
+-- data/                     # Dataset storage (git-ignored)
|   +-- validate_dataset.py   # Pre-training dataset health checks
|   +-- raw/ processed/ annotations/
+-- evaluation/               # Metrics & benchmarking
|   +-- benchmark_models.py   # Multi-model comparison suite
|   +-- evaluate.py           # Single-model multi-dataset eval
|   +-- metrics.py            # COCO-style mAP computation
|   +-- visualize.py          # Detection result visualization
+-- models/                   # Neural network architectures
|   +-- turb_detr.py          # Main Turb-DETR model (SimAM-enhanced)
|   +-- simam.py              # SimAM attention module (ICML 2021)
|   +-- backbones/ heads/     # Sub-module placeholders
+-- notebooks/                # Jupyter / Colab notebooks
|   +-- 01_train_colab.ipynb  # End-to-end Colab training
+-- outputs/                  # Runtime artifacts (git-ignored)
|   +-- checkpoints/ logs/ visualizations/
+-- scripts/                  # CLI entry points
|   +-- train.py              # Launch training
|   +-- evaluate.py           # Run evaluation
|   +-- infer.py              # Single-image inference
+-- training/                 # Training engine
|   +-- trainer.py            # Config-driven training orchestrator
|   +-- train_baseline.py     # Standalone baseline RT-DETR trainer
|   +-- scheduler.py          # LR scheduler (cosine + warmup)
+-- utils/                    # Shared utilities
|   +-- logger.py             # Loguru structured logging
|   +-- io_utils.py           # Path resolution, device detection
+-- .gitignore
+-- README.md
+-- requirements.txt"""
    code_block(tree)

    # ═══════════════════════════════════════════════════════════
    # 5. MODULE-BY-MODULE EXPLANATION
    # ═══════════════════════════════════════════════════════════
    pdf.add_page()
    section("5. Module-by-Module Detailed Explanation")

    # 5.1 Configs
    subsection("5.1 Configuration Files (configs/)")
    body(
        "All experiment settings are externalized to YAML files, enabling reproducible runs "
        "without code changes."
    )
    body(
        "train_config.yaml -- Central training configuration. Specifies the model variant "
        "(RT-DETR-L), 100 training epochs, batch size 16, AdamW optimizer with initial LR "
        "1e-4, cosine decay to 1% of initial LR, 3 warmup epochs, early stopping patience "
        "of 15 epochs, and mixed-precision (AMP) training. Also configures underwater-specific "
        "augmentation flags (color shift, turbidity simulation, caustic overlay)."
    )
    body(
        "model_config.yaml -- Architecture blueprint. Defines ResNet-50 backbone (pretrained, all "
        "stages trainable), hybrid encoder (256 hidden dim, 8 attention heads, GELU activation), "
        "decoder (6 layers, 300 object queries), and the turbidity attention module configuration "
        "(channel attention with reduction ratio 16)."
    )
    body(
        "trash_icra19.yaml -- Ultralytics-compatible dataset definition for the Trash-ICRA19 "
        "benchmark. Specifies dataset root path, train/val/test image directories, 5 object classes "
        "(plastic, bottle, can, bag, net), and supports path override for Colab environments."
    )

    # 5.2 Turb-DETR
    pdf.add_page()
    subsection("5.2 Models: Turb-DETR Architecture (models/turb_detr.py)")
    body(
        "The Turb-DETR model is the centerpiece of this project. It wraps an Ultralytics RT-DETR "
        "model and injects a SimAM turbidity suppression module between the CNN backbone and the "
        "transformer encoder. The architecture pipeline is:"
    )
    body("  Input Image -> CNN Backbone (ResNet/HGNetv2) -> SimAM Turbidity Suppression -> Transformer Encoder -> Object Query Decoder -> Detection Head")
    body(
        "Key design decisions:\n"
        "- Uses register_forward_hook() to inject SimAM after the backbone, avoiding any "
        "modification to Ultralytics source code.\n"
        "- SimAM is parameter-free, so pretrained RT-DETR weights load perfectly.\n"
        "- Setting use_simam=False reverts to a vanilla RT-DETR baseline for ablation studies.\n"
        "- The SimAMFeatureHook class processes multi-scale feature maps independently.\n"
        "- TurbiditySuppressionBlock (nn.Module) is provided as a standalone alternative."
    )
    body(
        "Public API preserved for downstream scripts:\n"
        "- TurbDETR(model_variant, weights, config_path, use_simam, simam_lambda)\n"
        "- .train(data_cfg, **kwargs)\n"
        "- .validate(data_cfg, **kwargs)\n"
        "- .predict(source, **kwargs)\n"
        "- .export(fmt, **kwargs)\n"
        "- .info() -- prints model summary with parameter counts\n"
        "- .remove_simam() -- dynamically remove SimAM hook at runtime"
    )

    # 5.3 SimAM
    subsection("5.3 Models: SimAM Attention Module (models/simam.py)")
    body(
        "SimAM (Simple, Parameter-Free Attention Module) from ICML 2021 is a lightweight "
        "attention mechanism that computes per-neuron importance weights using an energy-based "
        "formulation. Unlike SE-Net or CBAM, SimAM requires zero learnable parameters."
    )
    body(
        "Mathematical formulation:\n"
        "  For each neuron t in a feature map with spatial mean mu and variance sigma^2:\n"
        "  Energy e_t = 4 * (sigma^2 + lambda) / ((t - mu)^2 + 2*sigma^2 + 2*lambda)\n"
        "  Attention weight = sigmoid(1 / e_t)\n\n"
        "Low-energy neurons (those deviating from the mean) receive higher attention weights, "
        "effectively suppressing uniform/degraded regions and amplifying distinctive features -- "
        "exactly what is needed for turbidity suppression."
    )
    body(
        "Implementation: nn.Module with a single buffer (lambda_param = 1e-4 default). "
        "Input/output shape: (B, C, H, W) -> (B, C, H, W). Also provides a functional "
        "interface simam_attention() and includes a self-test function."
    )

    # 5.4 Augmentation Pipeline
    pdf.add_page()
    subsection("5.4 Augmentation Pipeline (augmentation/pipeline.py)")
    body(
        "Built on Albumentations for composable, bbox-aware augmentation. The training pipeline "
        "includes:"
    )
    aug_steps = [
        "LongestMaxSize(640) + padding to square",
        "UnderwaterAugmentation (custom): turbidity fog overlay, wavelength-dependent color shift, caustic light patterns",
        "HorizontalFlip (p=0.5), VerticalFlip (p=0.3)",
        "RandomBrightnessContrast, GaussNoise, GaussianBlur",
        "CLAHE (adaptive histogram equalization)",
        "ImageNet normalization + ToTensorV2",
    ]
    for s in aug_steps:
        bullet(s)
    body(
        "The validation pipeline applies only deterministic resize + normalize. All augmentations "
        "are YOLO bbox-format aware via Albumentations BboxParams."
    )

    # 5.5 Turbidity Augmentation
    subsection("5.5 Physics-Based Turbidity Augmentation (augmentation/turbidity_aug.py)")
    body(
        "A physically-inspired turbidity simulation module that models real underwater optical "
        "phenomena based on the Beer-Lambert law of light attenuation."
    )
    body("Four physics effects are modeled:")
    effects = [
        "Beer-Lambert Attenuation: Exponential decay of light intensity with distance, with wavelength-dependent absorption coefficients (red absorbed first, blue last).",
        "Backscatter Haze: Additive veiling light from particles scattering light back toward the camera. Modeled as a uniform color overlay blended by turbidity level.",
        "Forward-Scatter Blur: Gaussian blur simulating the angular spread of light through suspended particles. Kernel size scales with turbidity.",
        "Particle Noise: Salt-and-pepper noise simulating visible suspended sediment. Density proportional to turbidity level.",
    ]
    for e in effects:
        bullet(e)
    body(
        "Provides: apply_turbidity() function, TurbidityTransform (PyTorch-compatible nn.Module), "
        "turbidity presets (clear/moderate/heavy/extreme), and a visualize_turbidity() function "
        "for side-by-side comparison plots."
    )

    # 5.6 Training Scripts
    pdf.add_page()
    subsection("5.6 Training Scripts (training/)")
    body(
        "trainer.py -- Config-driven training orchestrator. Loads train_config.yaml, initializes "
        "TurbDETR with the specified variant and hyperparameters, and launches Ultralytics "
        "model.train() with all parameters mapped from YAML. Handles logging via Loguru."
    )
    body(
        "train_baseline.py -- Standalone CLI script for baseline RT-DETR training. Features "
        "full argparse CLI with configurable epochs, batch size, image size, optimizer, learning "
        "rate, early stopping, and output paths. Includes auto GPU detection, training time "
        "measurement, and post-training validation with metrics reporting. Can be run directly "
        "from terminal or with %run in Colab."
    )
    body(
        "scheduler.py -- Learning rate scheduler utility implementing cosine annealing with "
        "linear warmup. Wraps PyTorch LambdaLR with a custom lambda that linearly increases "
        "LR during warmup epochs, then cosine-decays to a configurable minimum factor."
    )

    # 5.7 Evaluation & Benchmarking
    subsection("5.7 Evaluation & Benchmarking (evaluation/)")
    body(
        "evaluate.py -- Comprehensive single-model evaluation across multiple datasets "
        "(Trash-ICRA19, TrashCan, RUIE). Metrics: mAP@0.5, mAP@0.5:0.95, precision, recall, "
        "and inference FPS (measured via timed dummy tensor inference with GPU synchronization). "
        "Outputs: formatted console table, JSON results file, CSV experiment log for tracking "
        "across runs. CLI interface with --weights, --datasets, --device flags."
    )
    body(
        "benchmark_models.py -- Multi-model comparison suite for benchmarking YOLOv10, RT-DETR, "
        "and Turb-DETR across all three datasets. Generates: comparison console table, grouped "
        "bar charts (mAP, precision, recall, FPS), FPS-vs-mAP scatter plot (speed-accuracy "
        "trade-off), parameter-efficiency bar chart, Markdown summary report with best-per-metric "
        "analysis, JSON + CSV result files. CLI interface with --models, --names, --datasets flags."
    )
    body(
        "metrics.py -- COCO-style evaluation using pycocotools. Computes AP, AP50, AP75, and "
        "size-stratified metrics (AP_small, AP_medium, AP_large). Includes per-class AP and F1 "
        "computation helpers."
    )
    body(
        "visualize.py -- Detection result visualization with color-coded bounding boxes and "
        "labels. Also provides training metric curve plotting (loss, mAP over epochs) with "
        "optional file saving."
    )

    # 5.8 Data Validation
    pdf.add_page()
    subsection("5.8 Dataset Validation (data/validate_dataset.py)")
    body(
        "A pre-training dataset health checker that runs 6 automated validation checks:"
    )
    checks = [
        "Directory structure: Verifies train/val/test image and label directories exist",
        "Image-label pairing: Ensures every image has a corresponding label file and vice versa",
        "YOLO bbox format: Validates all annotations follow [class_id x_center y_center w h] format with values in [0,1]",
        "Corrupted image detection: Attempts to open every image with OpenCV and flags failures",
        "Per-class distribution: Counts instances per class across all splits",
        "Summary statistics: Reports total images, labels, instances, and class balance",
    ]
    for c in checks:
        bullet(c)

    # 5.9 Utils
    subsection("5.9 Utility Modules (utils/)")
    body(
        "logger.py -- Structured logging via Loguru with dual sinks: colorized console output "
        "(INFO level) and rotating file logs (DEBUG level, 10MB rotation, 30-day retention) "
        "written to outputs/logs/. Provides get_logger(name) for module-specific loggers."
    )
    body(
        "io_utils.py -- Path resolution and environment detection. Key functions: is_colab() for "
        "Google Colab detection, get_project_root() that works on both local machines and Colab, "
        "load_yaml() for config file loading, ensure_dir() for directory creation, and "
        "get_device() for auto-detecting the best compute device (CUDA > MPS > CPU)."
    )

    # 5.10 CLI Scripts
    subsection("5.10 CLI Entry Points (scripts/)")
    body(
        "train.py -- Minimal CLI wrapper that loads a training config YAML and delegates to "
        "training.trainer.run_training(). Adds project root to sys.path for clean imports."
    )
    body(
        "evaluate.py -- CLI wrapper that loads a TurbDETR model from weights and runs validation "
        "against the dataset specified in the config YAML."
    )
    body(
        "infer.py -- CLI inference script accepting --weights, --source (image/directory/video), "
        "--imgsz, --conf, and --save-dir. Runs TurbDETR.predict() and saves annotated results."
    )

    # 5.11 Colab Notebook
    subsection("5.11 Google Colab Notebook (notebooks/01_train_colab.ipynb)")
    body(
        "A complete end-to-end training notebook with 18 cells covering:"
    )
    notebook_cells = [
        "Environment setup: clone repo, install requirements, import libraries",
        "GPU verification and CUDA status check",
        "Google Drive mounting for persistent weight storage",
        "Dataset download and symlink setup (Trash-ICRA19 YOLO format)",
        "Dataset structure verification and class statistics",
        "Baseline RT-DETR training via Ultralytics API (configurable epochs/batch/imgsz)",
        "Evaluation: mAP@0.5, mAP@0.5:0.95, precision, recall computation",
        "Inference visualization: annotated detection results on test images",
        "Weight saving to Google Drive with checkpoint management",
    ]
    for c in notebook_cells:
        bullet(c)

    # ═══════════════════════════════════════════════════════════
    # 6. ARCHITECTURE DEEP DIVE
    # ═══════════════════════════════════════════════════════════
    pdf.add_page()
    section("6. Architecture Deep Dive")
    subsection("6.1 RT-DETR Base Architecture")
    body(
        "RT-DETR (Real-Time DEtection TRansformer) is a hybrid architecture combining a CNN "
        "backbone with a transformer encoder-decoder, achieving real-time performance without "
        "NMS post-processing. Key properties:\n"
        "- Backbone: ResNet-50/101 or HGNetv2 for multi-scale feature extraction\n"
        "- Hybrid Encoder: Efficient intra-scale feature interaction via AIFI (Attention-based "
        "Intra-scale Feature Interaction) and cross-scale feature fusion via CCFM (CNN-based "
        "Cross-scale Feature-fusion Module)\n"
        "- Decoder: Standard transformer decoder with 300 learned object queries\n"
        "- No NMS: Uses bipartite matching (Hungarian algorithm) for set prediction"
    )

    subsection("6.2 SimAM Injection Strategy")
    body(
        "Our modification injects SimAM at the critical junction between CNN feature extraction "
        "and transformer encoding. This is achieved via PyTorch's register_forward_hook() "
        "mechanism -- no Ultralytics source code is modified."
    )
    body(
        "The hook intercepts the list of multi-scale feature maps output by the backbone "
        "(typically 3 scales: P3, P4, P5) and applies independent SimAM attention to each scale. "
        "This has three key benefits:\n"
        "1. Suppresses uniform/degraded activations from turbid regions\n"
        "2. Amplifies distinctive object features that survive turbidity\n"
        "3. Adds zero learnable parameters -- pretrained weights are 100% compatible"
    )

    subsection("6.3 Training Configuration")
    table(
        ["Parameter", "Value", "Notes"],
        [
            ["Model", "RT-DETR-L", "32M params, ResNet-50 backbone"],
            ["Image Size", "640x640", "Standard detection resolution"],
            ["Epochs", "100", "With early stopping (patience=15)"],
            ["Batch Size", "16", "Fits on T4/V100 GPU"],
            ["Optimizer", "AdamW", "Weight decay 1e-4"],
            ["Learning Rate", "1e-4", "Cosine decay to 1e-6"],
            ["Warmup", "3 epochs", "Linear warmup"],
            ["AMP", "Enabled", "FP16 mixed precision"],
            ["Augmentation", "Custom", "Underwater + standard"],
        ],
        [55, 40, 85],
    )

    # ═══════════════════════════════════════════════════════════
    # 7. DATASET
    # ═══════════════════════════════════════════════════════════
    pdf.add_page()
    section("7. Dataset: Trash-ICRA19")
    body(
        "The primary training and evaluation dataset is Trash-ICRA19, a benchmark for underwater "
        "trash detection derived from the J-EDI (JAMSTEC E-Library of Deep-sea Images) dataset. "
        "It contains annotated images of debris on the deep-sea floor captured by ROVs "
        "(Remotely Operated Vehicles)."
    )

    subsection("7.1 Class Definition")
    table(
        ["Class ID", "Name", "Description"],
        [
            ["0", "plastic", "Plastic fragments and sheets"],
            ["1", "bottle", "Plastic and glass bottles"],
            ["2", "can", "Metal cans and containers"],
            ["3", "bag", "Plastic bags and wrappers"],
            ["4", "net", "Fishing nets and rope"],
        ],
        [30, 40, 110],
    )

    subsection("7.2 Dataset Format")
    body(
        "Stored in YOLO format with the following structure:\n"
        "  data/trash_icra19/\n"
        "    images/train/  images/val/  images/test/\n"
        "    labels/train/  labels/val/  labels/test/\n\n"
        "Each label file contains one line per object:\n"
        "  <class_id> <x_center> <y_center> <width> <height>\n"
        "All coordinates are normalized to [0, 1]."
    )

    subsection("7.3 Additional Evaluation Datasets")
    table(
        ["Dataset", "Description", "Purpose"],
        [
            ["TrashCan", "Larger underwater/above-water debris dataset", "Cross-dataset generalization"],
            ["RUIE", "Real-world Underwater Image Enhancement", "Turbidity robustness testing"],
        ],
        [40, 80, 60],
    )

    # ═══════════════════════════════════════════════════════════
    # 8. CURRENT STATUS
    # ═══════════════════════════════════════════════════════════
    pdf.add_page()
    section("8. Current Project Status")
    body("Summary of all completed components as of March 8, 2026:")
    table(
        ["Component", "Status", "File(s)"],
        [
            ["Project scaffold", "Complete", "All directories + __init__.py"],
            ["Requirements", "Complete", "requirements.txt"],
            ["Training config", "Complete", "configs/train_config.yaml"],
            ["Model config", "Complete", "configs/model_config.yaml"],
            ["Dataset config", "Complete", "configs/trash_icra19.yaml"],
            ["Turb-DETR model", "Complete", "models/turb_detr.py"],
            ["SimAM attention", "Complete", "models/simam.py"],
            ["Augmentation pipeline", "Complete", "augmentation/pipeline.py"],
            ["Turbidity augmentation", "Complete", "augmentation/turbidity_aug.py"],
            ["Underwater effects", "Complete", "augmentation/underwater.py"],
            ["Baseline trainer", "Complete", "training/train_baseline.py"],
            ["Config trainer", "Complete", "training/trainer.py"],
            ["LR scheduler", "Complete", "training/scheduler.py"],
            ["Evaluation script", "Complete", "evaluation/evaluate.py"],
            ["Benchmark suite", "Complete", "evaluation/benchmark_models.py"],
            ["COCO metrics", "Complete", "evaluation/metrics.py"],
            ["Visualization", "Complete", "evaluation/visualize.py"],
            ["Dataset validation", "Complete", "data/validate_dataset.py"],
            ["Logger", "Complete", "utils/logger.py"],
            ["IO utilities", "Complete", "utils/io_utils.py"],
            ["CLI train script", "Complete", "scripts/train.py"],
            ["CLI eval script", "Complete", "scripts/evaluate.py"],
            ["CLI infer script", "Complete", "scripts/infer.py"],
            ["Colab notebook", "Complete", "notebooks/01_train_colab.ipynb"],
            ["Git + GitHub", "Complete", "All pushed to main branch"],
        ],
        [55, 30, 95],
    )
    body("Total files: 40+ | Total Python LOC: ~3,500+ | All code pushed to GitHub.")

    # ═══════════════════════════════════════════════════════════
    # 9. FUTURE WORK
    # ═══════════════════════════════════════════════════════════
    pdf.add_page()
    section("9. Future Work & Next Steps")

    subsection("Phase 1: Data Preparation (Immediate)")
    steps_p1 = [
        "Download and prepare Trash-ICRA19 dataset into data/trash_icra19/ with proper train/val/test splits",
        "Run data/validate_dataset.py to verify dataset integrity before training",
        "Download TrashCan and RUIE datasets for cross-dataset evaluation",
        "Create configs/trashcan.yaml and configs/ruie.yaml dataset definitions",
        "Analyze class distribution and potential class imbalance issues",
    ]
    for s in steps_p1:
        bullet(s)

    subsection("Phase 2: Training & Experimentation (Next 2-3 Weeks)")
    steps_p2 = [
        "Train baseline RT-DETR-L on Trash-ICRA19 (use_simam=False) to establish baseline metrics",
        "Train Turb-DETR (use_simam=True) under identical conditions for fair comparison",
        "Conduct hyperparameter search: SimAM lambda values (1e-3, 1e-4, 1e-5)",
        "Apply turbidity augmentation during training with varying intensity presets",
        "Run multi-epoch training on Google Colab or cloud GPU (T4/V100/A100)",
        "Track experiments in Weights & Biases (set wandb: true in config)",
        "Save best checkpoints to Google Drive for persistence",
    ]
    for s in steps_p2:
        bullet(s)

    subsection("Phase 3: Evaluation & Analysis (Week 3-4)")
    steps_p3 = [
        "Run evaluation/evaluate.py on all three datasets with best checkpoints",
        "Run evaluation/benchmark_models.py comparing YOLOv10 vs RT-DETR vs Turb-DETR",
        "Analyze per-class AP breakdown: which debris types benefit most from SimAM?",
        "Generate speed-accuracy trade-off plots (FPS vs mAP)",
        "Test robustness under different synthetic turbidity levels",
        "Perform ablation study: remove SimAM hook and compare performance drop",
    ]
    for s in steps_p3:
        bullet(s)

    subsection("Phase 4: Advanced Improvements (Optional / Extended)")
    steps_p4 = [
        "Experiment with RT-DETR-X (larger variant, ~67M params) for higher accuracy",
        "Try alternative attention mechanisms (CBAM, SE-Net) as SimAM replacements",
        "Implement multi-scale SimAM with learnable lambda per feature scale",
        "Add TTA (Test-Time Augmentation) for evaluation robustness",
        "Export models to ONNX/TensorRT for deployment on edge devices (Jetson Nano)",
        "Create inference demo with video input (underwater footage)",
        "Write research paper / technical report with experimental results",
    ]
    for s in steps_p4:
        bullet(s)

    subsection("Phase 5: Deployment & Documentation (Final)")
    steps_p5 = [
        "Package model as a pip-installable module",
        "Create Docker container for reproducible inference",
        "Build Gradio/Streamlit demo for interactive testing",
        "Write comprehensive API documentation",
        "Prepare presentation slides for project defense",
        "Final cleanup: code review, docstring audit, type-check with mypy",
    ]
    for s in steps_p5:
        bullet(s)

    # ═══════════════════════════════════════════════════════════
    # 10. APPENDIX: FILE INVENTORY
    # ═══════════════════════════════════════════════════════════
    pdf.add_page()
    section("10. Appendix: Complete File Inventory")
    all_files = [
        (".gitignore", "Git ignore rules for data, outputs, caches"),
        ("README.md", "Project overview, setup instructions, usage guide"),
        ("requirements.txt", "Python package dependencies (10 packages)"),
        ("augmentation/__init__.py", "Package init"),
        ("augmentation/pipeline.py", "Albumentations augmentation pipeline (84 LOC)"),
        ("augmentation/turbidity_aug.py", "Physics-based turbidity simulation (~250 LOC)"),
        ("augmentation/underwater.py", "Color shift, fog, caustic overlays (~80 LOC)"),
        ("configs/dataset.yaml", "Generic dataset configuration"),
        ("configs/model_config.yaml", "Architecture hyperparameters (37 lines)"),
        ("configs/train_config.yaml", "Training hyperparameters (62 lines)"),
        ("configs/trash_icra19.yaml", "Trash-ICRA19 dataset definition (35 lines)"),
        ("data/validate_dataset.py", "Dataset health checker (6 checks, ~200 LOC)"),
        ("evaluation/__init__.py", "Package init"),
        ("evaluation/benchmark_models.py", "Multi-model benchmark suite (~380 LOC)"),
        ("evaluation/evaluate.py", "Single-model evaluation (~310 LOC)"),
        ("evaluation/metrics.py", "COCO-style mAP computation (~90 LOC)"),
        ("evaluation/visualize.py", "Detection visualization (~100 LOC)"),
        ("models/__init__.py", "Package init"),
        ("models/simam.py", "SimAM attention module (~177 LOC)"),
        ("models/turb_detr.py", "Turb-DETR model with SimAM injection (~280 LOC)"),
        ("models/backbones/__init__.py", "Package init (placeholder)"),
        ("models/heads/__init__.py", "Package init (placeholder)"),
        ("notebooks/01_train_colab.ipynb", "Colab training notebook (18 cells)"),
        ("scripts/train.py", "CLI training entry point (~40 LOC)"),
        ("scripts/evaluate.py", "CLI evaluation entry point (~50 LOC)"),
        ("scripts/infer.py", "CLI inference entry point (~60 LOC)"),
        ("training/__init__.py", "Package init"),
        ("training/trainer.py", "Config-driven trainer (~70 LOC)"),
        ("training/train_baseline.py", "Baseline RT-DETR trainer (~227 LOC)"),
        ("training/scheduler.py", "Cosine warmup LR scheduler (~40 LOC)"),
        ("utils/__init__.py", "Package init"),
        ("utils/logger.py", "Loguru structured logging (~40 LOC)"),
        ("utils/io_utils.py", "Path resolution & device detection (~55 LOC)"),
    ]
    table(
        ["File Path", "Description"],
        [[f, d] for f, d in all_files],
        [70, 110],
    )

    # ═══════════════════════════════════════════════════════════
    # SAVE
    # ═══════════════════════════════════════════════════════════
    pdf.output(out_path)
    print(f"Report saved: {out_path}")

if __name__ == "__main__":
    import os
    from pathlib import Path
    # Resolve project root relative to this script — works on any OS
    os.chdir(Path(__file__).parent)
    build_report("Turb_DETR_Project_Report.pdf")
