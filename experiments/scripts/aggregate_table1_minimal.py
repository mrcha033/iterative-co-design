#!/usr/bin/env python3
"""Aggregate minimal Table 1 results (Mamba only)."""

import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats


def load_metrics(run_dir: Path):
    metrics_file = run_dir / "metrics.json"
    if not metrics_file.exists():
        return None
    with open(metrics_file) as f:
        return json.load(f)


def compute_cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("table1_dir", type=Path)
    args = parser.parse_args()

    baselines = ["dense", "algo_only", "linear", "iterative"]
    results = {b: [] for b in baselines}

    # Collect runs
    for baseline in baselines:
        baseline_dir = args.table1_dir / baseline
        if not baseline_dir.exists():
            continue
        for run_dir in sorted(baseline_dir.glob("run_*")):
            metrics = load_metrics(run_dir)
            if metrics:
                latency = metrics.get("latency_ms")
                if latency is not None:
                    results[baseline].append(latency)

    # Statistics
    print("\n" + "="*60)
    print("Table 1 (Minimal) - Mamba-2.8B Results")
    print("="*60)

    stats_dict = {}
    for baseline in baselines:
        data = results[baseline]
        if data:
            mean = np.mean(data)
            std = np.std(data, ddof=1)
            stats_dict[baseline] = {"mean": mean, "std": std, "n": len(data)}
            print(f"{baseline:12s}: {mean:.1f} ± {std:.2f} ms (n={len(data)})")

    # Critical comparison
    if results["iterative"] and results["linear"]:
        iterative_data = results["iterative"]
        linear_data = results["linear"]

        t_stat, p_value = stats.ttest_rel(linear_data, iterative_data)
        cohens_d = compute_cohens_d(linear_data, iterative_data)
        improvement = (stats_dict["linear"]["mean"] - stats_dict["iterative"]["mean"]) / stats_dict["linear"]["mean"] * 100

        print("\n" + "-"*60)
        print("Critical Comparison: Iterative vs Linear")
        print(f"  Improvement: {improvement:.1f}%")
        print(f"  p-value: {p_value:.6f} {'✓ significant' if p_value < 0.01 else '✗ n.s.'}")
        print(f"  Cohen's d: {cohens_d:.2f}")
        print("="*60)

        # Check if meets paper claims (15-25% improvement, p<0.001, d>1.2)
        print("\nValidation against paper claims:")
        print(f"  [{'✓' if 15 <= improvement <= 25 else '✗'}] Improvement in 15-25% range: {improvement:.1f}%")
        print(f"  [{'✓' if p_value < 0.001 else '✗'}] p < 0.001: {p_value:.6f}")
        print(f"  [{'✓' if cohens_d > 1.2 else '✗'}] Cohen's d > 1.2: {cohens_d:.2f}")


if __name__ == "__main__":
    main()