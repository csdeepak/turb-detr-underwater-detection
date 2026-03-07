"""CLI entry point — evaluate a trained Turb-DETR model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.turb_detr import TurbDETR
from utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Turb-DETR model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to training config YAML (for data paths)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to trained model weights (.pt)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info(f"Loading weights from {args.weights}")

    model = TurbDETR(weights=args.weights)

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    results = model.validate(data_cfg=cfg["data"]["config"])
    logger.info(f"Validation results: {results}")


if __name__ == "__main__":
    main()
