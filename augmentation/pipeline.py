"""Composable augmentation pipeline using Albumentations + custom transforms."""

from __future__ import annotations

from typing import Any

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2

from augmentation.underwater import (
    add_caustic_pattern,
    simulate_turbidity,
    underwater_color_shift,
)


class UnderwaterAugmentation(A.ImageOnlyTransform):
    """Albumentations-compatible wrapper for underwater-specific transforms."""

    def __init__(
        self,
        turbidity_prob: float = 0.5,
        color_shift_prob: float = 0.5,
        caustic_prob: float = 0.2,
        always_apply: bool = False,
        p: float = 1.0,
    ) -> None:
        super().__init__(always_apply=always_apply, p=p)
        self.turbidity_prob = turbidity_prob
        self.color_shift_prob = color_shift_prob
        self.caustic_prob = caustic_prob

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        rng = np.random.default_rng()
        if rng.random() < self.turbidity_prob:
            intensity = rng.uniform(0.1, 0.4)
            img = simulate_turbidity(img, intensity=intensity)
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
        return ("turbidity_prob", "color_shift_prob", "caustic_prob")


def get_train_transforms(imgsz: int = 640) -> A.Compose:
    """Standard training augmentation pipeline with underwater effects."""
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
    """Validation / test augmentation pipeline (deterministic)."""
    return A.Compose(
        [
            A.LongestMaxSize(max_size=imgsz),
            A.PadIfNeeded(min_height=imgsz, min_width=imgsz, border_mode=0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
    )
