"""Custom metric helpers for evaluation."""

import numpy as np


def compute_map_drop(clean_map: float, turbid_map: float) -> dict:
    """Compute mAP drop metrics for reporting."""
    absolute_drop = clean_map - turbid_map
    relative_drop = (absolute_drop / clean_map) * 100 if clean_map > 0 else 0
    return {
        "clean_map": clean_map,
        "turbid_map": turbid_map,
        "absolute_drop": absolute_drop,
        "relative_drop_pct": relative_drop,
    }
