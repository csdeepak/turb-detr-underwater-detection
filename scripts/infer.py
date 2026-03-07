"""CLI entry point — run inference with a trained Turb-DETR model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.turb_detr import TurbDETR
from utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Turb-DETR Inference")
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to trained model weights (.pt)",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Image file, directory, or video path",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="outputs/visualizations",
        help="Directory to save results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info(f"Running inference: source={args.source}, weights={args.weights}")

    model = TurbDETR(weights=args.weights)
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        save=True,
        project=str(Path(args.save_dir).parent),
        name=Path(args.save_dir).name,
    )

    logger.info(f"Inference done — {len(results)} result(s) saved to {args.save_dir}")


if __name__ == "__main__":
    main()
