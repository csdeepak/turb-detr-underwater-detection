"""Composable augmentation pipeline using Albumentations + custom transforms."""

from __future__ import annotations

from typing import Any

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2

from augmentation.turbidity_aug import apply_turbidity  # physics-accurate (Beer-Lambert)
from augmentation.underwater import (
    add_caustic_pattern,
    underwater_color_shift,
)


class UnderwaterAugmentation(A.ImageOnlyTransform):
    """Albumentations-compatible wrapper for underwater-specific transforms.

    Uses ``apply_turbidity`` from ``turbidity_aug`` (Beer-Lambert + backscatter
    + forward-scatter) for turbidity simulation — the same model used during
    turbidity robustness evaluation — so train/eval augmentation is consistent.
    """

    def __init__(
        self,
        turbidity_prob: float = 0.5,
        color_shift_prob: float = 0.5,
        caustic_prob: float = 0.2,
        turbidity_level_min: float = 0.1,
        turbidity_level_max: float = 0.7,
        always_apply: bool = False,
        p: float = 1.0,
    ) -> None:
        super().__init__(always_apply=always_apply, p=p)
        self.turbidity_prob = turbidity_prob
        self.color_shift_prob = color_shift_prob
        self.caustic_prob = caustic_prob
        self.turbidity_level_min = turbidity_level_min
        self.turbidity_level_max = turbidity_level_max

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        rng = np.random.default_rng()
        if rng.random() < self.turbidity_prob:
            level = float(rng.uniform(self.turbidity_level_min, self.turbidity_level_max))
            # apply_turbidity expects BGR uint8; img from Albumentations is RGB uint8 —
            # convert, apply, convert back.
            import cv2
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            bgr = apply_turbidity(bgr, level=level)
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rng.random() < self.color_shift_prob:
            img = underwater_color_shift(
                img,
                blue_gain=rng.uniform(1.0, 1.4),
                red_loss=rng.uniform(0.5, 0.9),
            )
        if rng.random() < self.caustic_prob:
            img = add_caustic_pattern(img, strength=rng.uniform(0.05, 0.2))
        return img

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return (
            "turbidity_prob",
            "color_shift_prob",
            "caustic_prob",
            "turbidity_level_min",
            "turbidity_level_max",
        )


def get_train_transforms(imgsz: int = 640) -> A.Compose:
    """Standard training augmentation pipeline with underwater effects.

    .. note::
        **NOT called in the main Turb-DETR training path.**
        Ultralytics manages its own transforms internally.  Turbidity
        augmentation is injected via the ``on_train_start`` callback in
        ``training/trainer.py`` instead.  This function is provided for
        external / custom DataLoader pipelines that need a standalone
        Albumentations pipeline.
    """
    return A.Compose(
        [
            A.LongestMaxSize(max_size=imgsz),
            A.PadIfNeeded(min_height=imgsz, min_width=imgsz, border_mode=0),
            UnderwaterAugmentation(p=0.7),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(var_limit=(10, 50), p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.CLAHE(clip_limit=4.0, p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
    )


def get_val_transforms(imgsz: int = 640) -> A.Compose:
    """Validation / test augmentation pipeline (deterministic).

    .. note::
        **NOT called in the main Turb-DETR evaluation path.**
        Provided for external / custom DataLoader pipelines only.
    """
    return A.Compose(
        [
            A.LongestMaxSize(max_size=imgsz),
            A.PadIfNeeded(min_height=imgsz, min_width=imgsz, border_mode=0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
    )
