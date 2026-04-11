"""Tests for data integrity — run before every experiment."""

import pytest
from pathlib import Path
from src.utils.data_leak_check import load_filenames


@pytest.fixture
def split_files():
    """Check if split files exist."""
    splits_dir = Path("data/splits")
    if not splits_dir.exists():
        pytest.skip("Split files not yet generated")
    return {
        "train": splits_dir / "train.txt",
        "val": splits_dir / "val.txt",
        "test": splits_dir / "test.txt",
    }


def test_no_train_test_overlap(split_files):
    train = load_filenames(str(split_files["train"]))
    test = load_filenames(str(split_files["test"]))
    overlap = train & test
    assert len(overlap) == 0, f"LEAK: {len(overlap)} images in both train and test"


def test_no_train_val_overlap(split_files):
    train = load_filenames(str(split_files["train"]))
    val = load_filenames(str(split_files["val"]))
    overlap = train & val
    assert len(overlap) == 0, f"LEAK: {len(overlap)} images in both train and val"


def test_splits_not_empty(split_files):
    for name, path in split_files.items():
        data = load_filenames(str(path))
        assert len(data) > 0, f"{name} split is empty"
