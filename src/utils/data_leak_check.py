"""
Data Leak Checker

CRITICAL: Run this BEFORE every training session.
Verifies that no test/val images appear in the training set.
A single leaked image invalidates all results.
"""

import argparse
from pathlib import Path


def load_filenames(filepath: str) -> set:
    """Load image filenames from a split file (one filename per line)."""
    with open(filepath) as f:
        return {line.strip() for line in f if line.strip()}


def check_leaks(train_file: str, val_file: str, test_file: str) -> bool:
    """Check for overlap between train, val, and test splits."""
    train = load_filenames(train_file)
    val = load_filenames(val_file)
    test = load_filenames(test_file)

    train_val_leak = train & val
    train_test_leak = train & test
    val_test_leak = val & test

    clean = True

    if train_val_leak:
        print(f"LEAK: {len(train_val_leak)} images in BOTH train and val")
        for f in list(train_val_leak)[:5]:
            print(f"  - {f}")
        clean = False

    if train_test_leak:
        print(f"LEAK: {len(train_test_leak)} images in BOTH train and test")
        for f in list(train_test_leak)[:5]:
            print(f"  - {f}")
        clean = False

    if val_test_leak:
        print(f"LEAK: {len(val_test_leak)} images in BOTH val and test")
        for f in list(val_test_leak)[:5]:
            print(f"  - {f}")
        clean = False

    if clean:
        print(f"NO LEAKS DETECTED")
        print(f"  Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
        print(f"  Total: {len(train) + len(val) + len(test)}")

    return clean


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--test", required=True)
    args = parser.parse_args()

    is_clean = check_leaks(args.train, args.val, args.test)
    if not is_clean:
        exit(1)
