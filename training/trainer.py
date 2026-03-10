"""Trainer — orchestrates the full training pipeline."""

from __future__ import annotations

import random
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


# ─────────────────────────────────────────────────────────────
# Turbidity augmentation callback
# ─────────────────────────────────────────────────────────────
class _TurbidityTransform:
    """Ultralytics-compatible transform that applies turbidity simulation.

    Ultralytics datasets pass a ``labels`` dict through each transform in
    ``dataset.transforms.transforms``.  The image is stored as a numpy
    BGR ``uint8`` array in ``labels["img"]`` at this stage of the pipeline
    (before Ultralytics normalises / converts to tensor).
    """

    def __init__(
        self,
        prob: float = 0.5,
        level_range: tuple[float, float] = (0.1, 0.7),
        color_shift: bool = False,
        caustic: bool = False,
    ) -> None:
        self.prob = prob
        self.level_range = level_range
        self.color_shift = color_shift
        self.caustic = caustic
        # Import lazily so the module is only required when turbidity aug is enabled.
        from augmentation.turbidity_aug import apply_turbidity
        self._apply = apply_turbidity
        if color_shift:
            from augmentation.underwater import underwater_color_shift
            self._color_shift = underwater_color_shift
        if caustic:
            from augmentation.underwater import add_caustic_pattern
            self._caustic = add_caustic_pattern

    def __call__(self, labels: dict) -> dict:
        if random.random() < self.prob:
            img = labels.get("img")
            if img is not None:
                level = random.uniform(*self.level_range)
                img = self._apply(img, level=level)
                if self.color_shift:
                    img = self._color_shift(
                        img,
                        blue_gain=random.uniform(1.0, 1.3),
                        red_loss=random.uniform(0.6, 0.9),
                    )
                if self.caustic:
                    img = self._caustic(img, strength=random.uniform(0.05, 0.15))
                labels["img"] = img
        return labels


def _make_turbidity_callback(
    prob: float,
    level_range: tuple[float, float],
    color_shift: bool = False,
    caustic: bool = False,
):
    """Return an Ultralytics ``on_train_start`` callback that injects turbidity Aug.

    The callback is called once the Ultralytics trainer has built the training
    dataloader, so ``trainer.train_loader.dataset`` is already initialised.
    It prepends ``_TurbidityTransform`` to the dataset's ``transforms.transforms``
    list so that turbidity simulation runs on every training image *before*
    Ultralytics' own normalisation and tensor conversion.
    """
    transform = _TurbidityTransform(
        prob=prob,
        level_range=level_range,
        color_shift=color_shift,
        caustic=caustic,
    )

    def on_train_start(trainer):
        ds = getattr(trainer, "train_loader", None)
        if ds is None:
            logger.warning("turbidity_aug: train_loader not found on trainer — skipping.")
            return
        ds = trainer.train_loader.dataset
        transforms_obj = getattr(ds, "transforms", None)
        if transforms_obj is None or not hasattr(transforms_obj, "transforms"):
            logger.warning(
                "turbidity_aug: dataset.transforms has unexpected structure — "
                "turbidity augmentation was NOT applied. "
                "Ensure Ultralytics ≥ 8.3 is installed."
            )
            return
        # Prepend so turbidity runs on the raw BGR image before any Ultralytics op.
        transforms_obj.transforms.insert(0, transform)
        logger.info(
            f"turbidity_aug: injected into dataset transforms "
            f"(prob={prob}, level={level_range[0]:.1f}–{level_range[1]:.1f})"
        )

    return on_train_start


# ─────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────

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

    use_simam = model_cfg.get("use_simam", True)
    logger.info(f"SimAM injection: {use_simam}")

    model = TurbDETR(
        model_variant=model_cfg["name"],
        weights=None,  # use COCO-pretrained default
        use_simam=use_simam,
    )

    # ── Turbidity augmentation ────────────────────────────────
    # The custom flags (turbidity_simulation, underwater_color_shift) are NOT
    # native Ultralytics keys and were previously silently ignored.  We now
    # wire them via a trainer callback that patches the dataset transform list.
    aug_cfg = cfg.get("augmentation", {})
    if aug_cfg.get("turbidity_simulation", False):
        callback = _make_turbidity_callback(
            prob=aug_cfg.get("turbidity_prob", 0.5),
            level_range=(
                aug_cfg.get("turbidity_level_min", 0.1),
                aug_cfg.get("turbidity_level_max", 0.7),
            ),
            color_shift=aug_cfg.get("underwater_color_shift", False),
            caustic=aug_cfg.get("caustic_overlay", False),
        )
        model.model.add_callback("on_train_start", callback)
        logger.info("turbidity_simulation: callback registered — will inject on train start.")
    else:
        logger.info("turbidity_simulation: disabled in config.")

    # ── Pass only native Ultralytics augmentation keys ─────────
    _ULTRALYTICS_AUG_KEYS = {
        "hsv_h", "hsv_s", "hsv_v", "degrees", "translate",
        "scale", "flipud", "fliplr", "mosaic", "mixup",
    }
    aug_kwargs = {k: v for k, v in aug_cfg.items() if k in _ULTRALYTICS_AUG_KEYS}

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
        **aug_kwargs,
    }

    logger.info("Starting training …")
    results = model.train(data_cfg=data_cfg, **train_kwargs)
    logger.info("Training complete.")
    return results

