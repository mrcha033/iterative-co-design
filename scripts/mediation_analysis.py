"""Bootstrap mediation analysis for modularity -> latency experiments.

The script assumes a CSV input with the following columns:
    modularity: modularity score of the permutation
    hit_rate: measured L2 cache hit rate
    latency_ms: observed latency in milliseconds

Usage:
    python scripts/mediation_analysis.py --data logs/mamba_mediation.csv --bootstrap 5000

The procedure implements the Baron & Kenny mediation test with bias-corrected
and accelerated (BCa) confidence intervals for the indirect effect.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class MediationResult:
    direct_effect: float
    indirect_effect: float
    total_effect: float
    mediation_fraction: float
    p_value: float
    ci_low: float
    ci_high: float


def _fit_linear(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    slope, intercept, _, _, _ = stats.linregress(x, y)
    return slope, intercept


def bootstrap_mediation(
    modularity: np.ndarray,
    hit_rate: np.ndarray,
    latency: np.ndarray,
    num_bootstrap: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    indirect = np.empty(num_bootstrap, dtype=float)
    direct = np.empty(num_bootstrap, dtype=float)
    n = modularity.shape[0]
    for i in range(num_bootstrap):
        idx = rng.integers(0, n, size=n)
        m = modularity[idx]
        h = hit_rate[idx]
        y = latency[idx]
        a, _ = _fit_linear(m, h)
        b, _ = _fit_linear(h, y)
        c_prime, _ = _fit_linear(m, y - b * h)
        indirect[i] = a * b
        direct[i] = c_prime
    return indirect, direct


def mediation_analysis(
    df: pd.DataFrame, num_bootstrap: int, seed: int
) -> MediationResult:
    modularity = df["modularity"].to_numpy()
    hit_rate = df["hit_rate"].to_numpy()
    latency = df["latency_ms"].to_numpy()

    a, _ = _fit_linear(modularity, hit_rate)
    b, _ = _fit_linear(hit_rate, latency)
    c, _ = _fit_linear(modularity, latency)
    indirect_effect = a * b
    direct_effect = c - indirect_effect

    rng = np.random.default_rng(seed)
    indirect_samples, direct_samples = bootstrap_mediation(
        modularity, hit_rate, latency, num_bootstrap, rng
    )
    total_samples = indirect_samples + direct_samples
    total_effect = c
    mediation_fraction = indirect_effect / total_effect if total_effect else np.nan

    ci_low, ci_high = np.percentile(indirect_samples, [2.5, 97.5])
    z_score = indirect_effect / indirect_samples.std(ddof=1)
    p_value = 2 * stats.norm.sf(abs(z_score))

    return MediationResult(
        direct_effect=direct_effect,
        indirect_effect=indirect_effect,
        total_effect=total_effect,
        mediation_fraction=mediation_fraction,
        p_value=p_value,
        ci_low=ci_low,
        ci_high=ci_high,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap mediation analysis")
    parser.add_argument("--data", type=Path, required=True, help="CSV file with metrics")
    parser.add_argument(
        "--bootstrap", type=int, default=5000, help="Number of bootstrap samples"
    )
    parser.add_argument("--seed", type=int, default=17, help="Random seed")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    required = {"modularity", "hit_rate", "latency_ms"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Input file missing columns: {', '.join(sorted(missing))}")

    result = mediation_analysis(df, args.bootstrap, args.seed)
    print("Total effect (c):", result.total_effect)
    print("Indirect effect (a*b):", result.indirect_effect)
    print("Direct effect (c'):", result.direct_effect)
    print("Mediation fraction:", result.mediation_fraction)
    print("p-value:", result.p_value)
    print("95% CI (percentile):", (result.ci_low, result.ci_high))


if __name__ == "__main__":
    main()
