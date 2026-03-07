"""Trainer — orchestrates the full training pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from models.turb_detr import TurbDETR
from utils.logger import get_logger

logger = get_logger(__name__)


def load_train_config(config_path: str | Path) -> dict[str, Any]:
    """Load training configuration from YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_training(config_path: str | Path) -> None:
    """End-to-end training from a single config file.

    Parameters
    ----------
    config_path : str | Path
        Path to ``train_config.yaml``.
    """
    cfg = load_train_config(config_path)
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]["config"]
    output_cfg = cfg["output"]

    logger.info(f"Initializing {model_cfg['name']} (pretrained={model_cfg['pretrained']})")

    model = TurbDETR(
        model_variant=model_cfg["name"],
        weights=None,  # use COCO-pretrained default
    )

    train_kwargs: dict[str, Any] = {
        "epochs": train_cfg["epochs"],
        "batch": train_cfg["batch_size"],
        "imgsz": model_cfg["imgsz"],
        "optimizer": train_cfg["optimizer"],
        "lr0": train_cfg["lr0"],
        "lrf": train_cfg["lrf"],
        "weight_decay": train_cfg["weight_decay"],
        "warmup_epochs": train_cfg["warmup_epochs"],
        "patience": train_cfg["patience"],
        "amp": train_cfg["amp"],
        "workers": cfg["data"]["workers"],
        "project": output_cfg["project"],
        "name": output_cfg["name"],
        "save_period": output_cfg["save_period"],
        "exist_ok": output_cfg["exist_ok"],
    }

    logger.info("Starting training …")
    results = model.train(data_cfg=data_cfg, **train_kwargs)
    logger.info("Training complete.")
    return results
