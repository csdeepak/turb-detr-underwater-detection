"""
Unified Evaluation Script

Evaluates a model across all three tracks:
- Track A: Clean test set
- Track B: Synthetic turbid test set (3 levels)
- Track C: Real-world turbid (UFO-120, qualitative only)

Outputs a CSV table ready for paper inclusion.
"""

import argparse
import csv
from pathlib import Path

from ultralytics import RTDETR


def evaluate_track_a(model_path: str, clean_test_yaml: str) -> dict:
    """Track A: Evaluate on clean Trash-ICRA19 test set."""
    model = RTDETR(model_path)
    results = model.val(data=clean_test_yaml)
    return {
        "condition": "clean",
        "map50": results.box.map50,
        "map50_95": results.box.map,
        "precision": results.box.mp,
        "recall": results.box.mr,
    }


def evaluate_track_b(model_path: str, turbid_test_dir: str, levels: list[str] = None) -> list[dict]:
    """Track B: Evaluate on synthetic turbid test sets."""
    if levels is None:
        levels = ["light", "medium", "heavy"]

    model = RTDETR(model_path)
    results_list = []

    for level in levels:
        yaml_path = Path(turbid_test_dir) / f"trash_icra19_turbid_{level}.yaml"
        if not yaml_path.exists():
            print(f"  WARNING: {yaml_path} not found, skipping {level}")
            continue

        results = model.val(data=str(yaml_path))
        results_list.append({
            "condition": f"turbid_{level}",
            "map50": results.box.map50,
            "map50_95": results.box.map,
            "precision": results.box.mp,
            "recall": results.box.mr,
        })

    return results_list


def save_results(all_results: list[dict], output_path: str):
    """Save results to CSV."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["condition", "map50", "map50_95", "precision", "recall"])
        writer.writeheader()
        writer.writerows(all_results)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model weights")
    parser.add_argument("--clean-yaml", required=True, help="Clean test dataset YAML")
    parser.add_argument("--turbid-dir", default=None, help="Directory with turbid YAML configs")
    parser.add_argument("--output", default="results/tables/evaluation.csv")
    args = parser.parse_args()

    all_results = []

    print("Track A: Clean evaluation...")
    clean_result = evaluate_track_a(args.model, args.clean_yaml)
    all_results.append(clean_result)
    print(f"  mAP@0.5: {clean_result['map50']:.4f}")

    if args.turbid_dir:
        print("\nTrack B: Synthetic turbid evaluation...")
        turbid_results = evaluate_track_b(args.model, args.turbid_dir)
        for r in turbid_results:
            all_results.append(r)
            print(f"  {r['condition']}: mAP@0.5 = {r['map50']:.4f}")

    save_results(all_results, args.output)
