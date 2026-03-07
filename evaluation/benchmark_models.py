"""Benchmark multiple detection models under turbidity conditions.

Compares YOLOv10, RT-DETR, and Turb-DETR across Trash-ICRA19, TrashCan,
and RUIE datasets.  Generates comparison tables, performance plots, and
a consolidated result summary.

Usage
─────
  python -m evaluation.benchmark_models \
      --models yolov10n.pt rtdetr-l.pt runs/turb_detr/best.pt \
      --names  YOLOv10 RT-DETR Turb-DETR \
      --datasets trash_icra19 trashcan ruie \
      --imgsz 640 \
      --device 0
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics import YOLO

matplotlib.use("Agg")  # non-interactive backend for server / Colab


# ─────────────────────────────────────────────────────────────
# Dataset registry (mirrors evaluate.py)
# ─────────────────────────────────────────────────────────────
DATASET_CONFIGS: dict[str, dict[str, Any]] = {
    "trash_icra19": {
        "data_yaml": "configs/trash_icra19.yaml",
        "description": "Trash-ICRA19",
        "split": "test",
    },
    "trashcan": {
        "data_yaml": "configs/trashcan.yaml",
        "description": "TrashCan",
        "split": "test",
    },
    "ruie": {
        "data_yaml": "configs/ruie.yaml",
        "description": "RUIE Turbidity",
        "split": "test",
    },
}


# ─────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────
@dataclass
class BenchmarkEntry:
    """Single (model × dataset) evaluation entry."""

    model_name: str = ""
    model_path: str = ""
    dataset: str = ""
    map50: float = 0.0
    map50_95: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    fps: float = 0.0
    params_total: int = 0
    params_trainable: int = 0
    imgsz: int = 640
    device: str = "cpu"


# ─────────────────────────────────────────────────────────────
# Model utilities
# ─────────────────────────────────────────────────────────────
def load_model(weights: str | Path) -> YOLO:
    """Load an Ultralytics-compatible model checkpoint."""
    weights = Path(weights)
    if not weights.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")
    return YOLO(str(weights))


def count_parameters(model: YOLO) -> tuple[int, int]:
    """Return (total, trainable) parameter counts."""
    try:
        total = sum(p.numel() for p in model.model.parameters())
        trainable = sum(
            p.numel() for p in model.model.parameters() if p.requires_grad
        )
        return total, trainable
    except Exception:
        return 0, 0


def measure_fps(
    model: YOLO,
    imgsz: int = 640,
    warmup: int = 10,
    iterations: int = 100,
    device: str = "cpu",
) -> float:
    """Measure inference FPS with a synthetic tensor."""
    dev = torch.device("cuda:0" if device not in ("cpu",) else "cpu")
    dummy = torch.randn(1, 3, imgsz, imgsz, device=dev)

    for _ in range(warmup):
        model.predict(source=dummy, verbose=False, device=device)

    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iterations):
        model.predict(source=dummy, verbose=False, device=device)
    if dev.type == "cuda":
        torch.cuda.synchronize()

    return iterations / (time.perf_counter() - t0)


# ─────────────────────────────────────────────────────────────
# Single evaluation run
# ─────────────────────────────────────────────────────────────
def evaluate_single(
    model: YOLO,
    model_name: str,
    dataset_key: str,
    imgsz: int,
    batch: int,
    device: str,
) -> BenchmarkEntry:
    """Run validation for one model × one dataset pair."""
    cfg = DATASET_CONFIGS[dataset_key]
    print(f"\n  ▸ {model_name} × {cfg['description']}")

    metrics = model.val(
        data=cfg["data_yaml"],
        split=cfg.get("split", "test"),
        imgsz=imgsz,
        batch=batch,
        device=device,
        verbose=False,
    )

    total, trainable = count_parameters(model)
    fps = measure_fps(model, imgsz=imgsz, device=device)

    return BenchmarkEntry(
        model_name=model_name,
        model_path=str(model.ckpt_path) if hasattr(model, "ckpt_path") else "",
        dataset=dataset_key,
        map50=float(metrics.box.map50),
        map50_95=float(metrics.box.map),
        precision=float(metrics.box.mp),
        recall=float(metrics.box.mr),
        fps=fps,
        params_total=total,
        params_trainable=trainable,
        imgsz=imgsz,
        device=device,
    )


# ─────────────────────────────────────────────────────────────
# Console table
# ─────────────────────────────────────────────────────────────
def print_comparison_table(entries: list[BenchmarkEntry]) -> None:
    """Print a rich console comparison table."""
    header = (
        f"{'Model':<14} {'Dataset':<14} {'mAP@.5':>7} {'mAP@.5:.95':>10} "
        f"{'Prec':>6} {'Rec':>6} {'FPS':>7} {'Params(M)':>10}"
    )
    sep = "─" * len(header)

    print(f"\n{'═' * len(header)}")
    print("  BENCHMARK COMPARISON")
    print(f"{'═' * len(header)}")
    print(header)
    print(sep)

    for e in entries:
        params_m = e.params_total / 1e6 if e.params_total else 0
        print(
            f"{e.model_name:<14} {e.dataset:<14} {e.map50:>7.4f} "
            f"{e.map50_95:>10.4f} {e.precision:>6.3f} {e.recall:>6.3f} "
            f"{e.fps:>7.1f} {params_m:>10.2f}"
        )

    print(sep)


# ─────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────
def _plot_grouped_bar(
    entries: list[BenchmarkEntry],
    metric_key: str,
    ylabel: str,
    title: str,
    save_path: Path,
) -> None:
    """Create a grouped bar chart: one group per dataset, one bar per model."""
    model_names = sorted({e.model_name for e in entries})
    dataset_names = sorted({e.dataset for e in entries})

    # Build value matrix  (models × datasets)
    values: dict[str, dict[str, float]] = {
        m: {d: 0.0 for d in dataset_names} for m in model_names
    }
    for e in entries:
        values[e.model_name][e.dataset] = getattr(e, metric_key, 0.0)

    x = np.arange(len(dataset_names))
    n_models = len(model_names)
    width = 0.75 / max(n_models, 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, m in enumerate(model_names):
        offsets = x - 0.375 + (i + 0.5) * width
        vals = [values[m][d] for d in dataset_names]
        ax.bar(offsets, vals, width, label=m, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[benchmark] Plot saved → {save_path}")


def generate_plots(entries: list[BenchmarkEntry], out_dir: Path) -> None:
    """Generate all comparison plots."""
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_specs = [
        ("map50", "mAP@0.5", "mAP@0.5 Comparison", "map50_comparison.png"),
        ("map50_95", "mAP@0.5:0.95", "mAP@0.5:0.95 Comparison", "map50_95_comparison.png"),
        ("precision", "Precision", "Precision Comparison", "precision_comparison.png"),
        ("recall", "Recall", "Recall Comparison", "recall_comparison.png"),
        ("fps", "FPS", "Inference Speed (FPS)", "fps_comparison.png"),
    ]

    for key, ylabel, title, fname in plot_specs:
        _plot_grouped_bar(entries, key, ylabel, title, out_dir / fname)

    # ── Scatter: FPS vs mAP ─────────────────────────────────
    _plot_fps_vs_map(entries, out_dir / "fps_vs_map.png")

    # ── Parameter-efficiency radar ───────────────────────────
    _plot_param_efficiency(entries, out_dir / "param_efficiency.png")


def _plot_fps_vs_map(entries: list[BenchmarkEntry], save_path: Path) -> None:
    """Scatter plot — inference speed vs accuracy trade-off."""
    fig, ax = plt.subplots(figsize=(8, 5))

    model_names = sorted({e.model_name for e in entries})
    markers = ["o", "s", "D", "^", "v", "P"]

    for i, m in enumerate(model_names):
        sub = [e for e in entries if e.model_name == m]
        fps_vals = [e.fps for e in sub]
        map_vals = [e.map50_95 for e in sub]
        marker = markers[i % len(markers)]
        ax.scatter(fps_vals, map_vals, s=100, marker=marker, label=m, zorder=3)
        for e, xv, yv in zip(sub, fps_vals, map_vals):
            ax.annotate(
                e.dataset, (xv, yv),
                textcoords="offset points", xytext=(5, 5),
                fontsize=7, alpha=0.7,
            )

    ax.set_xlabel("FPS")
    ax.set_ylabel("mAP@0.5:0.95")
    ax.set_title("Speed–Accuracy Trade-off")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[benchmark] Plot saved → {save_path}")


def _plot_param_efficiency(entries: list[BenchmarkEntry], save_path: Path) -> None:
    """Bar chart — parameter count per model (log-scale)."""
    model_names: list[str] = []
    param_counts: list[float] = []
    seen: set[str] = set()

    for e in entries:
        if e.model_name not in seen:
            seen.add(e.model_name)
            model_names.append(e.model_name)
            param_counts.append(e.params_total / 1e6)

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(model_names, param_counts, color=plt.cm.Set2.colors[:len(model_names)])
    ax.set_ylabel("Parameters (M)")
    ax.set_title("Model Size Comparison")
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, param_counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{val:.1f}M", ha="center", va="bottom", fontsize=9,
        )

    plt.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[benchmark] Plot saved → {save_path}")


# ─────────────────────────────────────────────────────────────
# Output writers
# ─────────────────────────────────────────────────────────────
def save_results_json(entries: list[BenchmarkEntry], path: Path) -> None:
    """Dump all benchmark results to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(e) for e in entries]
    path.write_text(json.dumps(data, indent=2))
    print(f"[benchmark] JSON saved → {path}")


def save_results_csv(entries: list[BenchmarkEntry], path: Path) -> None:
    """Write benchmark results to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "model_name", "dataset", "map50", "map50_95",
        "precision", "recall", "fps",
        "params_total", "params_trainable", "imgsz", "device",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for e in entries:
            writer.writerow({k: getattr(e, k) for k in fieldnames})

    print(f"[benchmark] CSV saved  → {path}")


def save_summary(entries: list[BenchmarkEntry], path: Path) -> None:
    """Write a readable Markdown summary."""
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Benchmark Summary\n",
        f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        "## Results\n",
        "| Model | Dataset | mAP@.5 | mAP@.5:.95 | Precision | Recall | FPS | Params(M) |",
        "|-------|---------|--------|-----------|-----------|--------|-----|-----------|",
    ]

    for e in entries:
        params_m = e.params_total / 1e6 if e.params_total else 0
        lines.append(
            f"| {e.model_name} | {e.dataset} | {e.map50:.4f} | "
            f"{e.map50_95:.4f} | {e.precision:.3f} | {e.recall:.3f} | "
            f"{e.fps:.1f} | {params_m:.2f} |"
        )

    # Best-per-metric table
    lines.append("\n## Best Models per Metric\n")
    for metric, label in [
        ("map50", "mAP@0.5"),
        ("map50_95", "mAP@0.5:0.95"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("fps", "FPS"),
    ]:
        best = max(entries, key=lambda e: getattr(e, metric))
        lines.append(
            f"- **{label}**: {best.model_name} on {best.dataset} "
            f"({getattr(best, metric):.4f})"
        )

    path.write_text("\n".join(lines))
    print(f"[benchmark] Summary   → {path}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark detection models under turbidity conditions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help="Paths to model checkpoints (.pt).",
    )
    p.add_argument(
        "--names",
        type=str,
        nargs="+",
        default=None,
        help="Human-readable names for each model (must match --models length).",
    )
    p.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=list(DATASET_CONFIGS),
        choices=list(DATASET_CONFIGS),
        help="Datasets to benchmark on.",
    )
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    p.add_argument("--batch", type=int, default=16, help="Validation batch size.")
    p.add_argument(
        "--device",
        type=str,
        default="0" if torch.cuda.is_available() else "cpu",
        help="Device for inference.",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="results/benchmark",
        help="Output directory for results and plots.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    model_paths = args.models
    model_names = args.names or [Path(p).stem for p in model_paths]

    if len(model_names) != len(model_paths):
        raise ValueError("--names must have the same length as --models")

    print("═" * 60)
    print("  TURB-DETR BENCHMARK")
    print("═" * 60)
    print(f"  Models   : {', '.join(model_names)}")
    print(f"  Datasets : {', '.join(args.datasets)}")
    print(f"  Device   : {args.device}")
    print(f"  Image sz : {args.imgsz}")
    print("═" * 60)

    # ── Run all evaluations ──────────────────────────────────
    entries: list[BenchmarkEntry] = []

    for weights_path, name in zip(model_paths, model_names):
        print(f"\n{'─' * 60}")
        print(f"  Loading: {name} ({weights_path})")
        print(f"{'─' * 60}")
        model = load_model(weights_path)

        for ds_key in args.datasets:
            entry = evaluate_single(
                model=model,
                model_name=name,
                dataset_key=ds_key,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
            )
            entries.append(entry)

    # ── Outputs ──────────────────────────────────────────────
    out_dir = Path(args.out_dir)

    print_comparison_table(entries)
    save_results_json(entries, out_dir / "benchmark_results.json")
    save_results_csv(entries, out_dir / "benchmark_results.csv")
    save_summary(entries, out_dir / "benchmark_summary.md")
    generate_plots(entries, out_dir / "plots")

    print(f"\n[benchmark] All outputs saved to {out_dir}/")
    print("[benchmark] Done.")


if __name__ == "__main__":
    main()
