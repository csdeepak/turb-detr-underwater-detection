"""CLI entry point — train Turb-DETR model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.trainer import run_training
from utils.io_utils import get_device
from utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Turb-DETR model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to training config YAML",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info(f"Device: {get_device()}")
    logger.info(f"Config: {args.config}")
    run_training(args.config)


if __name__ == "__main__":
    main()
