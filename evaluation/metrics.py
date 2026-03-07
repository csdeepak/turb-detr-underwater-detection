"""Evaluation metrics — mAP, precision, recall, F1 using pycocotools."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def evaluate_coco(
    gt_json: str | Path,
    pred_json: str | Path,
    iou_type: str = "bbox",
) -> dict[str, float]:
    """Run COCO-style evaluation and return summary metrics.

    Parameters
    ----------
    gt_json : str | Path
        Path to ground-truth COCO annotation JSON.
    pred_json : str | Path
        Path to predictions JSON (COCO results format).
    iou_type : str
        ``"bbox"`` for detection, ``"segm"`` for instance segmentation.

    Returns
    -------
    dict
        Dictionary with AP, AP50, AP75, AR keys.
    """
    coco_gt = COCO(str(gt_json))
    coco_dt = coco_gt.loadRes(str(pred_json))

    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats
    return {
        "AP": float(stats[0]),
        "AP50": float(stats[1]),
        "AP75": float(stats[2]),
        "AP_small": float(stats[3]),
        "AP_medium": float(stats[4]),
        "AP_large": float(stats[5]),
        "AR_1": float(stats[6]),
        "AR_10": float(stats[7]),
        "AR_100": float(stats[8]),
    }


def compute_f1(precision: float, recall: float) -> float:
    """Compute F1 score from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def per_class_metrics(
    coco_gt: COCO,
    coco_dt: COCO,
    iou_thresh: float = 0.5,
) -> dict[int, dict[str, float]]:
    """Compute per-class AP at a given IoU threshold.

    Returns mapping of category_id → {AP, precision, recall}.
    """
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.iouThrs = [iou_thresh]
    coco_eval.evaluate()
    coco_eval.accumulate()

    results: dict[int, dict[str, float]] = {}
    for idx, cat_id in enumerate(coco_eval.params.catIds):
        precision = coco_eval.eval["precision"][0, :, idx, 0, -1]
        valid = precision[precision > -1]
        ap = float(np.mean(valid)) if len(valid) > 0 else 0.0
        results[cat_id] = {"AP": ap}

    return results
