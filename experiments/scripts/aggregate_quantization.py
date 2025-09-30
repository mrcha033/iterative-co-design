#!/usr/bin/env python3
"""Aggregate quantization experiment results."""

import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats


def load_metrics(run_dir: Path):
    """Load metrics.json from a run directory."""
    metrics_file = run_dir / "metrics.json"
    if not metrics_file.exists():
        return None
    with open(metrics_file) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Aggregate quantization results")
    parser.add_argument("quant_dir", type=Path, help="Path to quantization experiment directory")
    parser.add_argument("--output", type=Path, default=Path("summary.csv"))
    args = parser.parse_args()

    strategies = ["quant_perm", "perm_quant", "iterative"]
    strategy_names = {
        "quant_perm": "Quant→Permute",
        "perm_quant": "Permute→Quant",
        "iterative": "Iterative (Permute→Quant→RePermute)"
    }

    results = {s: [] for s in strategies}

    # Collect all runs
    for strategy in strategies:
        strategy_dir = args.quant_dir / strategy
        if not strategy_dir.exists():
            continue

        for run_dir in sorted(strategy_dir.glob("run_*")):
            metrics = load_metrics(run_dir)
            if metrics:
                latency = metrics.get("latency_ms")
                if latency is not None:
                    results[strategy].append(latency)

    # Compute statistics
    stats_list = []

    for strategy in strategies:
        data = results[strategy]
        if not data:
            continue

        stats_dict = {
            "strategy": strategy_names[strategy],
            "mean": np.mean(data),
            "std": np.std(data, ddof=1),
            "n": len(data)
        }
        stats_list.append(stats_dict)

    # Critical comparison: Iterative vs best linear pipeline
    # Compare iterative against "perm_quant" (Permute-then-Quant)
    if results["iterative"] and results["perm_quant"]:
        iterative_data = results["iterative"]
        perm_quant_data = results["perm_quant"]

        # Paired t-test
        t_stat, p_value = stats.ttest_rel(perm_quant_data, iterative_data)

        # Improvement
        perm_quant_mean = np.mean(perm_quant_data)
        iterative_mean = np.mean(iterative_data)
        improvement = (perm_quant_mean - iterative_mean) / perm_quant_mean * 100

        print("\n" + "="*60)
        print("Quantization Experiment Results")
        print("="*60)

        for s in stats_list:
            print(f"{s['strategy']:40s}: {s['mean']:.1f} ± {s['std']:.2f} ms (n={s['n']})")

        print("\n" + "-"*60)
        print(f"Critical comparison: Iterative vs Permute→Quant")
        print(f"  Improvement: {improvement:.1f}%")
        print(f"  p-value: {p_value:.6f} {'✓ significant' if p_value < 0.01 else '✗ n.s.'}")
        print("="*60)

    # Save to CSV
    df = pd.DataFrame(stats_list)
    df.to_csv(args.output, index=False)
    print(f"\n✓ Results saved to: {args.output}")


if __name__ == "__main__":
    main()