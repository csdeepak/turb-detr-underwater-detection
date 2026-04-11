"""
Statistical Significance Tests

Required for publication. Reports without significance tests
are treated as noise by reviewers.

Implements:
- McNemar's test (paired prediction comparison)
- Bootstrap confidence intervals for mAP
- Multi-run mean ± std aggregation
"""

import numpy as np
from scipy import stats


def mcnemar_test(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """
    McNemar's test for comparing two models' predictions.

    Args:
        correct_a: Boolean array — model A correct predictions
        correct_b: Boolean array — model B correct predictions

    Returns:
        dict with chi2 statistic and p-value
    """
    # Contingency: b01 = A wrong & B right, b10 = A right & B wrong
    b01 = np.sum(~correct_a & correct_b)
    b10 = np.sum(correct_a & ~correct_b)

    if b01 + b10 == 0:
        return {"chi2": 0.0, "p_value": 1.0, "significant": False}

    chi2 = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)

    return {
        "chi2": chi2,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "b01": int(b01),
        "b10": int(b10),
    }


def bootstrap_ci(values: np.ndarray, n_bootstrap: int = 1000, ci: float = 0.95, seed: int = 42) -> dict:
    """
    Bootstrap confidence interval for a metric (e.g., mAP).

    Args:
        values: Array of per-image metric values
        n_bootstrap: Number of bootstrap resamples
        ci: Confidence level

    Returns:
        dict with mean, lower, upper bounds
    """
    rng = np.random.RandomState(seed)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        boot_means.append(np.mean(sample))

    boot_means = np.array(boot_means)
    alpha = (1 - ci) / 2

    return {
        "mean": np.mean(values),
        "ci_lower": np.percentile(boot_means, alpha * 100),
        "ci_upper": np.percentile(boot_means, (1 - alpha) * 100),
        "std": np.std(boot_means),
    }


def multi_run_summary(run_results: list[float]) -> str:
    """Format multi-run results as mean ± std for paper."""
    mean = np.mean(run_results)
    std = np.std(run_results)
    return f"{mean:.1f} ± {std:.1f}"
