"""Tests for Jaffe-McGlamery turbidity augmentation."""

import numpy as np
import pytest
from src.augmentation.jaffe_mcglamery import jaffe_mcglamery_turbidity, TURBIDITY_LEVELS


def _make_test_image(h=100, w=100):
    """Create a simple test image."""
    return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)


def test_output_shape():
    """Output shape must match input."""
    img = _make_test_image()
    out = jaffe_mcglamery_turbidity(img, c=0.15)
    assert out.shape == img.shape


def test_output_dtype():
    """Output must be uint8."""
    img = _make_test_image()
    out = jaffe_mcglamery_turbidity(img, c=0.15)
    assert out.dtype == np.uint8


def test_output_range():
    """Output values must be in [0, 255]."""
    img = _make_test_image()
    out = jaffe_mcglamery_turbidity(img, c=0.30)
    assert out.min() >= 0
    assert out.max() <= 255


def test_higher_turbidity_lower_contrast():
    """Higher attenuation should reduce image contrast."""
    img = _make_test_image(200, 200)
    light = jaffe_mcglamery_turbidity(img, c=0.05, seed=42)
    heavy = jaffe_mcglamery_turbidity(img, c=0.30, seed=42)
    assert np.std(heavy.astype(float)) < np.std(light.astype(float))


def test_reproducibility():
    """Same seed must produce identical output."""
    img = _make_test_image()
    out1 = jaffe_mcglamery_turbidity(img, c=0.15, seed=42)
    out2 = jaffe_mcglamery_turbidity(img, c=0.15, seed=42)
    np.testing.assert_array_equal(out1, out2)


def test_all_preset_levels():
    """All preset turbidity levels must work."""
    img = _make_test_image()
    for name, params in TURBIDITY_LEVELS.items():
        out = jaffe_mcglamery_turbidity(img, c=params["c"], depth=params["depth"])
        assert out.shape == img.shape
