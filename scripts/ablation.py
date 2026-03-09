"""Ablation Study Orchestrator — Turb-DETR.

Runs 4 training configurations to isolate the contribution of each component:

  Variant A: Baseline RT-DETR        (no SimAM, standard augmentation)
  Variant B: RT-DETR + Turb-Aug      (no SimAM, turbidity augmentation)
  Variant C: RT-DETR + SimAM         (SimAM, standard augmentation)
  Variant D: Turb-DETR (full)        (SimAM + turbidity augmentation)

This tells you whether the mAP gain (if any) comes from the architecture
change, the data augmentation, or the combination.

Usage
-----
    python scripts/ablation.py \\
        --data configs/trash_icra19.yaml \\
        --epochs 50 \\
        --batch 8 \\
        --output-dir outputs/ablation

After all runs complete, the script generates:
    outputs/ablation/ablation_results.json
    outputs/ablation/ablation_table.txt
    outputs/ablation/ablation_barplot.png
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────────────────────
# Ablation variant definitions
# ─────────────────────────────────────────────────────────────
@dataclass
class AblationVariant:
    name: str
    description: str
    use_simam: bool
    use_turbidity_aug: bool


VARIANTS: list[AblationVariant] = [
    AblationVariant(
        name="A_baseline",
        description="Baseline RT-DETR\n(no SimAM, standard aug)",
        use_simam=False,
        use_turbidity_aug=False,
    ),
    AblationVariant(
        name="B_turb_aug",
        description="RT-DETR + Turb-Aug\n(no SimAM)",
        use_simam=False,
        use_turbidity_aug=True,
    ),
    AblationVariant(
        name="C_simam",
        description="RT-DETR + SimAM\n(no turbidity aug)",
        use_simam=True,
        use_turbidity_aug=False,
    ),
    AblationVariant(
        name="D_turb_detr",
        description="Turb-DETR (full)\nSimAM + Turb-Aug",
        use_simam=True,
        use_turbidity_aug=True,
    ),
]


# ─────────────────────────────────────────────────────────────
# Train a single variant using the existing training script
# ─────────────────────────────────────────────────────────────
def train_variant(
    variant: AblationVariant,
    data_cfg: str,
    epochs: int,
    batch: int,
    output_dir: Path,
    imgsz: int = 640,
) -> dict:
    """Launch training for one ablation variant and return metrics."""
    run_name   = variant.name
    project    = str(output_dir)

    print(f"\n{'='*60}")
    print(f"  Ablation Variant: {run_name}")
    print(f"  SimAM: {variant.use_simam}  |  Turbidity Aug: {variant.use_turbidity_aug}")
    print(f"{'='*60}")

    # Build train command using train_baseline.py
    # Override augmentation flag by writing a temp config
    train_script = Path(__file__).parent / "train_variant.py"
    if not train_script.exists():
        # Fall back to inline training
        return _train_inline(variant, data_cfg, epochs, batch, output_dir, imgsz)

    cmd = [
        sys.executable, str(train_script),
        "--data", data_cfg,
        "--epochs", str(epochs),
        "--batch",  str(batch),
        "--imgsz",  str(imgsz),
        "--project", project,
        "--name",    run_name,
        "--use-simam",   "1" if variant.use_simam          else "0",
        "--use-turb-aug","1" if variant.use_turbidity_aug  else "0",
    ]
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"[WARNING] Variant {run_name} training exited with code {result.returncode}")

    # Look for best.pt and metrics
    best_pt = output_dir / run_name / "weights" / "best.pt"
    metrics = _extract_metrics(output_dir / run_name)
    metrics["weights"] = str(best_pt) if best_pt.exists() else None
    return metrics


def _train_inline(
    variant: AblationVariant,
    data_cfg: str,
    epochs: int,
    batch: int,
    output_dir: Path,
    imgsz: int,
) -> dict:
    """Inline training without a subprocess (imports TurbDETR directly)."""
    from models.turb_detr import TurbDETR

    model = TurbDETR(model_variant="rtdetr-l", use_simam=variant.use_simam)
    extra_augment = {
        "mixup":  0.1,
        "mosaic": 1.0,
        "degrees": 5.0,
    }
    if not variant.use_turbidity_aug:
        # Disable underwater augmentation flags via Ultralytics kwargs
        extra_augment["hsv_h"] = 0.0
        extra_augment["hsv_s"] = 0.0
        extra_augment["hsv_v"] = 0.0

    results = model.train(
        data_cfg=data_cfg,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        project=str(output_dir),
        name=variant.name,
        exist_ok=True,
        **extra_augment,
    )
    metrics = _results_to_dict(results)
    best_pt = output_dir / variant.name / "weights" / "best.pt"
    metrics["weights"] = str(best_pt) if best_pt.exists() else None
    return metrics


def _results_to_dict(results) -> dict:
    """Extract key metrics from Ultralytics training results."""
    try:
        return {
            "map50":   float(results.results_dict.get("metrics/mAP50(B)",   0.0)),
            "map5095": float(results.results_dict.get("metrics/mAP50-95(B)",0.0)),
            "precision": float(results.results_dict.get("metrics/precision(B)", 0.0)),
            "recall":    float(results.results_dict.get("metrics/recall(B)",    0.0)),
        }
    except Exception:
        return {"map50": 0.0, "map5095": 0.0, "precision": 0.0, "recall": 0.0}


def _extract_metrics(run_dir: Path) -> dict:
    """Try to read metrics from results.csv if it exists."""
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return {"map50": 0.0, "map5095": 0.0, "precision": 0.0, "recall": 0.0}
    try:
        import csv
        rows = list(csv.DictReader(open(csv_path)))
        if not rows:
            return {"map50": 0.0, "map5095": 0.0, "precision": 0.0, "recall": 0.0}
        last = rows[-1]
        return {
            "map50":     float(last.get("metrics/mAP50(B)",    0.0)),
            "map5095":   float(last.get("metrics/mAP50-95(B)", 0.0)),
            "precision": float(last.get("metrics/precision(B)",0.0)),
            "recall":    float(last.get("metrics/recall(B)",   0.0)),
        }
    except Exception:
        return {"map50": 0.0, "map5095": 0.0, "precision": 0.0, "recall": 0.0}


# ─────────────────────────────────────────────────────────────
# Generate results table and plot
# ─────────────────────────────────────────────────────────────
def generate_outputs(
    variant_names: list[str],
    descriptions: list[str],
    all_metrics: list[dict],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Plain text table ─────────────────────────────────────
    table_lines = [
        "ABLATION STUDY RESULTS",
        "=" * 72,
        f"{'Variant':<22} {'mAP@0.5':>10} {'mAP@0.5:0.95':>14} {'Precision':>11} {'Recall':>9}",
        "-" * 72,
    ]
    for name, m in zip(variant_names, all_metrics):
        table_lines.append(
            f"{name:<22} "
            f"{m.get('map50', 0.0):>10.4f} "
            f"{m.get('map5095', 0.0):>14.4f} "
            f"{m.get('precision', 0.0):>11.4f} "
            f"{m.get('recall', 0.0):>9.4f}"
        )
    table_lines.append("=" * 72)
    table_text = "\n".join(table_lines)
    print("\n" + table_text)

    table_path = output_dir / "ablation_table.txt"
    table_path.write_text(table_text)
    print(f"\nTable saved to {table_path}")

    # ── JSON ─────────────────────────────────────────────────
    output_json = [
        {"variant": n, "description": d, **m}
        for n, d, m in zip(variant_names, descriptions, all_metrics)
    ]
    json_path = output_dir / "ablation_results.json"
    with open(json_path, "w") as f:
        json.dump(output_json, f, indent=2)
    print(f"JSON saved to {json_path}")

    # ── Grouped bar chart ─────────────────────────────────────
    metrics_to_plot = ["map50", "map5095", "precision", "recall"]
    metric_labels   = ["mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall"]
    x     = np.arange(len(variant_names))
    width = 0.18
    colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]

    fig, ax = plt.subplots(figsize=(13, 7))
    for i, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
        values = [m.get(metric, 0.0) for m in all_metrics]
        bars = ax.bar(x + i * width, values, width, label=label, color=colors[i], alpha=0.88)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=7.5,
            )

    short_names = [n.split("_")[0] + "\n" + " + ".join(n.split("_")[1:]) for n in variant_names]
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(short_names, fontsize=10)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Ablation Study: Component Contribution", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plot_path = output_dir / "ablation_barplot.png"
    fig.tight_layout()
    fig.savefig(str(plot_path), dpi=150)
    plt.close(fig)
    print(f"Bar plot saved to {plot_path}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def run_ablation(
    data_cfg: str,
    epochs: int,
    batch: int,
    output_dir: str | Path,
    imgsz: int = 640,
    variants: list[AblationVariant] | None = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if variants is None:
        variants = VARIANTS

    all_metrics: list[dict] = []
    for variant in variants:
        metrics = train_variant(variant, data_cfg, epochs, batch, output_dir, imgsz)
        all_metrics.append(metrics)

    generate_outputs(
        variant_names=[v.name for v in variants],
        descriptions=[v.description for v in variants],
        all_metrics=all_metrics,
        output_dir=output_dir,
    )


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Turb-DETR ablation study")
    parser.add_argument("--data",       required=True, help="Dataset YAML path")
    parser.add_argument("--epochs",     type=int, default=50,
                        help="Epochs per variant (reduce for quick experiments)")
    parser.add_argument("--batch",      type=int, default=8)
    parser.add_argument("--imgsz",      type=int, default=640)
    parser.add_argument("--output-dir", default="outputs/ablation")
    parser.add_argument(
        "--variants", nargs="+",
        choices=["A_baseline", "B_turb_aug", "C_simam", "D_turb_detr"],
        help="Run only specific variants (default: all four)"
    )
    args = parser.parse_args()

    selected = None
    if args.variants:
        name_map = {v.name: v for v in VARIANTS}
        selected = [name_map[n] for n in args.variants if n in name_map]

    run_ablation(
        data_cfg=args.data,
        epochs=args.epochs,
        batch=args.batch,
        output_dir=args.output_dir,
        imgsz=args.imgsz,
        variants=selected,
    )
