"""Bias-corrected accelerated (BCa) bootstrap confidence intervals.

Implements rigorous bootstrap methods for statistical inference as described
in the paper's methodology section.
"""

from __future__ import annotations

import logging
import math
from typing import Callable, Optional, Sequence

__all__ = ["bca_bootstrap_ci", "bootstrap_ci", "cohens_d"]

logger = logging.getLogger(__name__)

try:
    import numpy as np
    from scipy import stats as scipy_stats

    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False
    logger.warning("scipy not available, bootstrap CI will use simpler method")


def bca_bootstrap_ci(
    data: Sequence[float],
    statistic_fn: Optional[Callable] = None,
    alpha: float = 0.05,
    n_bootstrap: int = 10000,
    seed: Optional[int] = None,
) -> tuple[float, float, dict]:
    """Compute bias-corrected accelerated (BCa) bootstrap confidence interval.

    This implements the BCa method described in Efron & Tibshirani (1993),
    which corrects for bias and skewness in the bootstrap distribution.

    Args:
        data: Sample data (1D array-like).
        statistic_fn: Function to compute statistic. If None, uses mean.
        alpha: Significance level (default 0.05 for 95% CI).
        n_bootstrap: Number of bootstrap samples.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (ci_low, ci_high, metadata_dict)
    """
    if not _SCIPY_AVAILABLE:
        logger.warning("scipy not available, falling back to percentile bootstrap")
        return percentile_bootstrap_ci(data, statistic_fn, alpha, n_bootstrap, seed)

    if statistic_fn is None:
        statistic_fn = np.mean

    data_array = np.asarray(data)
    n = len(data_array)

    if n < 2:
        raise ValueError("Need at least 2 samples for bootstrap CI")

    # Original statistic
    theta_hat = statistic_fn(data_array)

    # Generate bootstrap distribution
    rng = np.random.default_rng(seed=seed)
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data_array, size=n, replace=True)
        bootstrap_stats.append(statistic_fn(sample))
    bootstrap_stats = np.array(bootstrap_stats)

    # Bias correction factor (z0)
    # Proportion of bootstrap stats less than original
    p = np.mean(bootstrap_stats < theta_hat)
    if p == 0:
        z0 = -np.inf
    elif p == 1:
        z0 = np.inf
    else:
        z0 = scipy_stats.norm.ppf(p)

    # Acceleration factor (a) via jackknife
    jackknife_stats = []
    for i in range(n):
        sample = np.delete(data_array, i)
        jackknife_stats.append(statistic_fn(sample))
    jackknife_stats = np.array(jackknife_stats)

    jackknife_mean = np.mean(jackknife_stats)
    numerator = np.sum((jackknife_mean - jackknife_stats) ** 3)
    denominator = 6 * (np.sum((jackknife_mean - jackknife_stats) ** 2) ** (3 / 2))

    if denominator == 0 or not np.isfinite(denominator):
        a = 0.0
    else:
        a = numerator / denominator

    # BCa percentiles
    z_alpha = scipy_stats.norm.ppf(alpha / 2)
    z_1_minus_alpha = scipy_stats.norm.ppf(1 - alpha / 2)

    # Correct percentiles
    if not np.isfinite(z0):
        # Fallback to percentile method
        p_low = alpha / 2
        p_high = 1 - alpha / 2
    elif abs(a) < 1e-10:
        # No acceleration needed
        p_low = scipy_stats.norm.cdf(z0 + z_alpha)
        p_high = scipy_stats.norm.cdf(z0 + z_1_minus_alpha)
    else:
        # Full BCa correction
        denom_low = 1 - a * (z0 + z_alpha)
        denom_high = 1 - a * (z0 + z_1_minus_alpha)

        if abs(denom_low) < 1e-10 or abs(denom_high) < 1e-10:
            # Numerical instability, fallback to percentile
            p_low = alpha / 2
            p_high = 1 - alpha / 2
        else:
            p_low = scipy_stats.norm.cdf(z0 + (z0 + z_alpha) / denom_low)
            p_high = scipy_stats.norm.cdf(z0 + (z0 + z_1_minus_alpha) / denom_high)

    # Ensure percentiles are in valid range
    p_low = np.clip(p_low, 0, 1)
    p_high = np.clip(p_high, 0, 1)

    # Compute confidence interval
    ci_low = np.percentile(bootstrap_stats, p_low * 100)
    ci_high = np.percentile(bootstrap_stats, p_high * 100)

    metadata = {
        "method": "bca_bootstrap",
        "n_bootstrap": n_bootstrap,
        "alpha": alpha,
        "z0": float(z0) if np.isfinite(z0) else None,
        "acceleration": float(a),
        "p_low": float(p_low),
        "p_high": float(p_high),
        "theta_hat": float(theta_hat),
    }

    return ci_low, ci_high, metadata


def percentile_bootstrap_ci(
    data: Sequence[float],
    statistic_fn: Optional[Callable] = None,
    alpha: float = 0.05,
    n_bootstrap: int = 10000,
    seed: Optional[int] = None,
) -> tuple[float, float, dict]:
    """Compute percentile bootstrap confidence interval (simpler method).

    Args:
        data: Sample data.
        statistic_fn: Function to compute statistic. If None, uses mean.
        alpha: Significance level.
        n_bootstrap: Number of bootstrap samples.
        seed: Random seed.

    Returns:
        Tuple of (ci_low, ci_high, metadata_dict)
    """
    if not _SCIPY_AVAILABLE:
        # Fallback without scipy
        import random

        if statistic_fn is None:
            statistic_fn = lambda x: sum(x) / len(x)

        data_list = list(data)
        n = len(data_list)
        random.seed(seed)

        bootstrap_stats = []
        for _ in range(n_bootstrap):
            sample = [random.choice(data_list) for _ in range(n)]
            bootstrap_stats.append(statistic_fn(sample))

        bootstrap_stats.sort()
        low_idx = int((alpha / 2) * n_bootstrap)
        high_idx = int((1 - alpha / 2) * n_bootstrap)

        ci_low = bootstrap_stats[low_idx]
        ci_high = bootstrap_stats[high_idx]

        metadata = {
            "method": "percentile_bootstrap_no_scipy",
            "n_bootstrap": n_bootstrap,
            "alpha": alpha,
        }

        return ci_low, ci_high, metadata

    # With scipy
    if statistic_fn is None:
        statistic_fn = np.mean

    data_array = np.asarray(data)
    n = len(data_array)

    rng = np.random.default_rng(seed=seed)
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data_array, size=n, replace=True)
        bootstrap_stats.append(statistic_fn(sample))
    bootstrap_stats = np.array(bootstrap_stats)

    ci_low = np.percentile(bootstrap_stats, (alpha / 2) * 100)
    ci_high = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)

    metadata = {
        "method": "percentile_bootstrap",
        "n_bootstrap": n_bootstrap,
        "alpha": alpha,
    }

    return ci_low, ci_high, metadata


def bootstrap_ci(
    data: Sequence[float],
    statistic_fn: Optional[Callable] = None,
    alpha: float = 0.05,
    n_bootstrap: int = 10000,
    method: str = "bca",
    seed: Optional[int] = None,
) -> tuple[float, float, dict]:
    """Compute bootstrap confidence interval.

    Args:
        data: Sample data.
        statistic_fn: Function to compute statistic.
        alpha: Significance level.
        n_bootstrap: Number of bootstrap samples.
        method: "bca" or "percentile".
        seed: Random seed.

    Returns:
        Tuple of (ci_low, ci_high, metadata_dict)
    """
    if method == "bca":
        return bca_bootstrap_ci(data, statistic_fn, alpha, n_bootstrap, seed)
    elif method == "percentile":
        return percentile_bootstrap_ci(data, statistic_fn, alpha, n_bootstrap, seed)
    else:
        raise ValueError(f"Unknown bootstrap method: {method}")


def cohens_d(group1: Sequence[float], group2: Sequence[float]) -> float:
    """Compute Cohen's d effect size for two groups.

    Args:
        group1: First group samples.
        group2: Second group samples.

    Returns:
        Cohen's d effect size.
    """
    if _SCIPY_AVAILABLE:
        g1 = np.asarray(group1)
        g2 = np.asarray(group2)

        n1, n2 = len(g1), len(g2)
        var1, var2 = np.var(g1, ddof=1), np.var(g2, ddof=1)

        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        if pooled_std == 0:
            return 0.0

        return (np.mean(g1) - np.mean(g2)) / pooled_std
    else:
        # Fallback without numpy
        import math

        n1, n2 = len(group1), len(group2)
        mean1 = sum(group1) / n1
        mean2 = sum(group2) / n2

        var1 = sum((x - mean1) ** 2 for x in group1) / (n1 - 1)
        var2 = sum((x - mean2) ** 2 for x in group2) / (n2 - 1)

        pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        if pooled_std == 0:
            return 0.0

        return (mean1 - mean2) / pooled_std