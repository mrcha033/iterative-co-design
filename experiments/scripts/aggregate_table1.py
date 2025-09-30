#!/usr/bin/env python3
"""Aggregate Table 1 results with statistical analysis (paired t-tests, Cohen's d)."""

import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple


def load_metrics(run_dir: Path) -> Dict:
    """Load metrics.json from a run directory."""
    metrics_file = run_dir / "metrics.json"
    if not metrics_file.exists():
        return None

    with open(metrics_file) as f:
        return json.load(f)


def compute_cohens_d(group1: List[float], group2: List[float]) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def analyze_architecture(arch_dir: Path, arch_name: str) -> Dict:
    """Analyze results for a single architecture."""
    baselines = ["dense", "algo_only", "linear", "iterative"]

    results = {baseline: [] for baseline in baselines}

    # Collect all runs
    for baseline in baselines:
        baseline_dir = arch_dir / baseline
        if not baseline_dir.exists():
            continue

        for run_dir in sorted(baseline_dir.glob("run_*")):
            metrics = load_metrics(run_dir)
            if metrics:
                latency = metrics.get("latency_ms")
                if latency is not None:
                    results[baseline].append(latency)

    # Check if we have data
    if not all(results.values()):
        return None

    # Compute statistics
    stats_dict = {}

    for baseline in baselines:
        data = results[baseline]
        stats_dict[f"{baseline}_mean"] = np.mean(data)
        stats_dict[f"{baseline}_std"] = np.std(data, ddof=1)
        stats_dict[f"{baseline}_n"] = len(data)

    # Critical comparison: Iterative vs Linear Pipeline
    iterative_data = results["iterative"]
    linear_data = results["linear"]

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(linear_data, iterative_data)

    # Cohen's d
    cohens_d = compute_cohens_d(linear_data, iterative_data)

    # Improvement percentage
    improvement = (stats_dict["linear_mean"] - stats_dict["iterative_mean"]) / stats_dict["linear_mean"] * 100

    stats_dict["improvement_pct"] = improvement
    stats_dict["p_value"] = p_value
    stats_dict["cohens_d"] = cohens_d
    stats_dict["significant"] = p_value < 0.001  # Bonferroni corrected α = 0.004

    return stats_dict


def main():
    parser = argparse.ArgumentParser(description="Aggregate Table 1 results")
    parser.add_argument("table1_dir", type=Path, help="Path to table1 experiment directory")
    parser.add_argument("--output", type=Path, default=Path("results_summary.csv"), help="Output CSV file")
    args = parser.parse_args()

    architectures = ["mamba", "bert", "resnet50", "gcn_arxiv"]

    all_results = []

    for arch in architectures:
        arch_dir = args.table1_dir / arch
        if not arch_dir.exists():
            print(f"Warning: {arch} directory not found, skipping...")
            continue

        print(f"\nAnalyzing {arch}...")
        stats = analyze_architecture(arch_dir, arch)

        if stats is None:
            print(f"  ⚠ Incomplete data for {arch}")
            continue

        stats["architecture"] = arch
        all_results.append(stats)

        # Print summary
        print(f"  Dense:     {stats['dense_mean']:.1f} ± {stats['dense_std']:.2f} ms")
        print(f"  Algo-Only: {stats['algo_only_mean']:.1f} ± {stats['algo_only_std']:.2f} ms")
        print(f"  Linear:    {stats['linear_mean']:.1f} ± {stats['linear_std']:.2f} ms")
        print(f"  Iterative: {stats['iterative_mean']:.1f} ± {stats['iterative_std']:.2f} ms")
        print(f"  → Improvement: {stats['improvement_pct']:.1f}%")
        print(f"  → p-value: {stats['p_value']:.6f} {'✓ sig' if stats['significant'] else '✗ n.s.'}")
        print(f"  → Cohen's d: {stats['cohens_d']:.2f}")

    # Create DataFrame and save
    df = pd.DataFrame(all_results)

    # Reorder columns
    cols = ["architecture", "dense_mean", "dense_std", "algo_only_mean", "algo_only_std",
            "linear_mean", "linear_std", "iterative_mean", "iterative_std",
            "improvement_pct", "p_value", "cohens_d", "significant"]
    df = df[cols]

    df.to_csv(args.output, index=False)
    print(f"\n✓ Results saved to: {args.output}")

    # Print LaTeX table snippet
    print("\n" + "="*60)
    print("LaTeX Table Snippet:")
    print("="*60)
    for _, row in df.iterrows():
        arch_name = {"mamba": "Mamba-2.8B", "bert": "BERT-large",
                     "resnet50": "ResNet-50", "gcn_arxiv": "GCN (ogbn-arxiv)"}[row["architecture"]]

        print(f"{row['dense_mean']:.1f} ± {row['dense_std']:.1f} & "
              f"{row['algo_only_mean']:.1f} ± {row['algo_only_std']:.1f} & "
              f"{row['linear_mean']:.1f} ± {row['linear_std']:.1f} & "
              f"\\textbf{{{row['iterative_mean']:.1f} ± {row['iterative_std']:.1f}}} & "
              f"\\textbf{{{row['improvement_pct']:.1f}\\%}} (d={row['cohens_d']:.1f}) \\\\")
    print("="*60)


if __name__ == "__main__":
    main()