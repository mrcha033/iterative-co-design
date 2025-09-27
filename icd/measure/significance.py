"""Utilities for paired significance tests on measurement metrics."""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Iterable, Mapping, MutableMapping, Sequence

try:  # pragma: no cover - optional dependency
    from scipy import stats as _scipy_stats  # type: ignore
except Exception:  # pragma: no cover - absence is expected in CI
    _scipy_stats = None

__all__ = [
    "paired_statistics",
    "compute_prd_significance",
    "merge_significance",
]


def _finite_floats(values: Iterable[float | int | None]) -> list[float]:
    result: list[float] = []
    for value in values:
        if value is None:
            continue
        try:
            fval = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(fval):
            result.append(fval)
    return result


def _mean(data: Sequence[float]) -> float | None:
    if not data:
        return None
    return float(sum(data) / len(data))


def _scalar_mean(metrics: Mapping[str, object], key: str) -> float | None:
    if key == "latency_ms":
        alt = metrics.get("latency_ms")
        if isinstance(alt, Mapping):
            maybe = alt.get("mean")
            if isinstance(maybe, (int, float)) and math.isfinite(float(maybe)):
                return float(maybe)
        maybe = metrics.get("latency_ms_mean")
        if isinstance(maybe, (int, float)) and math.isfinite(float(maybe)):
            return float(maybe)
        return None
    value = metrics.get(key)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _samples_for_metric(metrics: Mapping[str, object], key: str) -> list[float]:
    if key == "latency_ms":
        latency = metrics.get("latency_ms")
        if isinstance(latency, Mapping):
            samples = latency.get("samples")
            if isinstance(samples, Iterable):
                return _finite_floats(samples)  # type: ignore[arg-type]
        legacy = metrics.get("latency_samples")
        if isinstance(legacy, Iterable):
            return _finite_floats(legacy)  # type: ignore[arg-type]
        return []
    samples_obj = metrics.get(f"{key}_samples")
    if isinstance(samples_obj, Iterable):
        return _finite_floats(samples_obj)  # type: ignore[arg-type]
    value = metrics.get(key)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return [float(value)]
    return []


def paired_statistics(
    baseline_samples: Sequence[float | int | None],
    trial_samples: Sequence[float | int | None],
    *,
    baseline_mean: float | None = None,
    trial_mean: float | None = None,
    alpha: float = 0.05,
) -> dict[str, float | int | str | None]:
    """Compute paired delta, CI, effect size and p-value between two sample sets.

    Args:
        baseline_samples: Iterable of baseline samples (e.g., latency measurements).
        trial_samples: Iterable of trial samples.
        baseline_mean: Optional fallback mean if samples are unavailable.
        trial_mean: Optional fallback mean if samples are unavailable.
        alpha: Significance level for confidence intervals (default 0.05 ⇒ 95% CI).

    Returns:
        A JSON-serialisable dictionary describing the comparison. All statistics
        are ``None`` when they cannot be computed deterministically.
    """

    base = _finite_floats(baseline_samples)
    trial = _finite_floats(trial_samples)
    n = min(len(base), len(trial))
    base_used = base[:n]
    trial_used = trial[:n]

    base_mean = _mean(base_used) if base_used else baseline_mean
    trial_mean = _mean(trial_used) if trial_used else trial_mean
    mean_diff = None
    if base_mean is not None and trial_mean is not None:
        mean_diff = trial_mean - base_mean

    result: dict[str, float | int | str | None] = {
        "sample_size": int(n),
        "mean_baseline": base_mean,
        "mean_trial": trial_mean,
        "mean_diff": mean_diff,
        "ci_low": None,
        "ci_high": None,
        "statistic": None,
        "p_value": None,
        "effect_size": None,
        "method": "no_samples" if n == 0 else "insufficient_samples",
    }

    if n == 0:
        return result

    diffs = [trial_used[i] - base_used[i] for i in range(n)]
    mean_diff = _mean(diffs)
    if mean_diff is not None:
        result["mean_diff"] = mean_diff
    if base_used:
        result["mean_baseline"] = _mean(base_used)
    if trial_used:
        result["mean_trial"] = _mean(trial_used)

    if n == 1:
        result["method"] = "insufficient_samples"
        return result

    # Compute dispersion of paired differences
    try:
        import statistics

        stdev_diff = statistics.stdev(diffs)
    except Exception:
        stdev_diff = 0.0

    if not math.isfinite(stdev_diff) or stdev_diff == 0.0:
        # Zero variance ⇒ deterministic delta; CI collapses to mean diff
        result["method"] = "zero_variance"
        if mean_diff is not None:
            result["ci_low"] = mean_diff
            result["ci_high"] = mean_diff
            result["p_value"] = 0.0 if mean_diff != 0.0 else 1.0
        else:
            result["p_value"] = 1.0
        result["statistic"] = None
        result["effect_size"] = None
        return result

    se = stdev_diff / math.sqrt(n)
    if se == 0.0 or not math.isfinite(se):
        result["method"] = "zero_variance"
        return result

    statistic = None
    p_value = None
    ci_low = None
    ci_high = None
    effect_size = mean_diff / stdev_diff if mean_diff is not None else None

    if _scipy_stats is not None:
        try:  # pragma: no cover - SciPy branch not exercised in CI
            res = _scipy_stats.ttest_rel(trial_used, base_used, nan_policy="omit")
            statistic = float(res.statistic)
            p_value = float(res.pvalue)
            critical = float(_scipy_stats.t.ppf(1.0 - alpha / 2.0, n - 1))
            ci_low = mean_diff - critical * se if mean_diff is not None else None
            ci_high = mean_diff + critical * se if mean_diff is not None else None
            method = "scipy.ttest_rel"
        except Exception:  # pragma: no cover - defensive guard
            method = "normal_approx"
            statistic = mean_diff / se if mean_diff is not None else None
            normal = NormalDist()
            p_value = (
                2.0 * (1.0 - normal.cdf(abs(statistic))) if statistic is not None else None
            )
            z = normal.inv_cdf(1.0 - alpha / 2.0)
            ci_low = mean_diff - z * se if mean_diff is not None else None
            ci_high = mean_diff + z * se if mean_diff is not None else None
    else:
        method = "normal_approx"
        statistic = mean_diff / se if mean_diff is not None else None
        normal = NormalDist()
        p_value = 2.0 * (1.0 - normal.cdf(abs(statistic))) if statistic is not None else None
        z = normal.inv_cdf(1.0 - alpha / 2.0)
        ci_low = mean_diff - z * se if mean_diff is not None else None
        ci_high = mean_diff + z * se if mean_diff is not None else None

    result.update(
        {
            "method": method,
            "statistic": statistic,
            "p_value": p_value,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "effect_size": effect_size,
        }
    )
    return result


def compute_prd_significance(
    baseline_metrics: Mapping[str, object],
    trial_metrics: Mapping[str, object],
    *,
    alpha: float = 0.05,
) -> dict[str, dict[str, float | int | str | None]]:
    """Compute significance metadata for latency/L2/EpT metrics."""

    keys = {
        "latency_ms": "latency_ms",
        "l2_hit_pct": "l2_hit_pct",
        "ept_j_per_tok": "ept_j_per_tok",
    }

    results: dict[str, dict[str, float | int | str | None]] = {}
    for out_key, metric_key in keys.items():
        baseline_samples = _samples_for_metric(baseline_metrics, metric_key)
        trial_samples = _samples_for_metric(trial_metrics, metric_key)
        stats = paired_statistics(
            baseline_samples,
            trial_samples,
            baseline_mean=_scalar_mean(baseline_metrics, metric_key),
            trial_mean=_scalar_mean(trial_metrics, metric_key),
            alpha=alpha,
        )
        results[out_key] = stats
    return results


def merge_significance(
    target: MutableMapping[str, object],
    *,
    baseline_metrics: Mapping[str, object],
    trial_metrics: Mapping[str, object],
    alpha: float = 0.05,
) -> None:
    """Attach significance results into ``target`` under ``significance`` key."""

    significance = compute_prd_significance(baseline_metrics, trial_metrics, alpha=alpha)
    if significance:
        target["significance"] = significance

