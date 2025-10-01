#!/usr/bin/env python3
"""Aggregate baseline experiment results into Table 1 (paper format).

Compiles results from all baseline runs and computes:
- Mean ± std latency for each configuration
- Percentage improvements
- Statistical significance (paired t-tests with Bonferroni correction)
- Effect sizes (Cohen's d)

Generates LaTeX table matching paper format.

Usage:
    python scripts/aggregate_table1.py \\
        --input results/table1 \\
        --output results/paper_tables/table1_main_results.tex
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def compute_cohens_d_paired(baseline: List[float], treatment: List[float]) -> float:
    """Compute Cohen's d for paired samples.

    Args:
        baseline: Baseline measurements
        treatment: Treatment measurements

    Returns:
        Cohen's d effect size
    """
    differences = np.array(baseline) - np.array(treatment)
    return float(np.mean(differences) / np.std(differences, ddof=1))


def load_experimental_data(results_dir: Path, model: str, baseline: str) -> Dict[str, Any]:
    """Load experimental results for model/baseline combination.

    Args:
        results_dir: Root results directory
        model: Model name (mamba/bert/resnet/gcn)
        baseline: Baseline name (dense/algo/linear/iterative)

    Returns:
        Dictionary containing latency statistics
    """
    result_file = results_dir / model / f"{baseline}_metrics.json"

    if not result_file.exists():
        logger.warning(f"Missing: {result_file}")
        return {}

    with open(result_file) as f:
        data = json.load(f)

    return data


def aggregate_table1(results_dir: Path, output_file: Path, alpha: float = 0.001) -> None:
    """Generate Table 1 from experimental results.

    Args:
        results_dir: Directory containing all experimental results
        output_file: Output LaTeX file path
        alpha: Significance threshold (default: 0.001)
    """
    logger.info("=" * 80)
    logger.info("AGGREGATING TABLE 1: MAIN RESULTS")
    logger.info("=" * 80)

    models = ["mamba", "bert", "resnet", "gcn"]
    baselines = ["dense", "algo", "linear", "iterative"]

    # Bonferroni correction: k = 4 models × 3 metrics = 12 comparisons
    k_comparisons = 12
    alpha_corrected = alpha / k_comparisons
    logger.info(f"Significance threshold: α = {alpha} (Bonferroni-corrected: {alpha_corrected:.6f})")

    # Collect data
    table_data = []

    for model in models:
        logger.info(f"\n--- Processing {model} ---")

        row = {"model": model}

        # Load all baselines
        baseline_results = {}
        for baseline in baselines:
            data = load_experimental_data(results_dir, model, baseline)
            if not data:
                logger.error(f"Missing data for {model}/{baseline}")
                continue

            latency_stats = data.get("latency_stats", {})
            row[f"{baseline}_mean"] = latency_stats.get("mean", float("nan"))
            row[f"{baseline}_std"] = latency_stats.get("std", float("nan"))

            # Store samples for statistical tests
            baseline_results[baseline] = data

        # Compute improvement: Iterative vs Linear (our primary comparison)
        if "linear_mean" in row and "iterative_mean" in row:
            linear_mean = row["linear_mean"]
            iter_mean = row["iterative_mean"]

            if not np.isnan(linear_mean) and not np.isnan(iter_mean):
                improvement_pct = (linear_mean - iter_mean) / linear_mean * 100
                row["improvement_pct"] = improvement_pct
                logger.info(f"  Improvement: {improvement_pct:.1f}%")
            else:
                row["improvement_pct"] = float("nan")
                logger.warning(f"  Cannot compute improvement (NaN values)")

        # Statistical significance test
        if "linear" in baseline_results and "iterative" in baseline_results:
            linear_samples = baseline_results["linear"].get("latency_stats", {}).get("samples", [])
            iter_samples = baseline_results["iterative"].get("latency_stats", {}).get("samples", [])

            if linear_samples and iter_samples:
                # Paired t-test
                t_stat, p_value = stats.ttest_rel(linear_samples, iter_samples)

                # Cohen's d
                cohen_d = compute_cohens_d_paired(linear_samples, iter_samples)

                # Significance after Bonferroni correction
                significant = p_value < alpha_corrected

                row["p_value"] = p_value
                row["cohen_d"] = cohen_d
                row["significant"] = significant

                logger.info(f"  p-value: {p_value:.6f} (threshold: {alpha_corrected:.6f})")
                logger.info(f"  Cohen's d: {cohen_d:.2f}")
                logger.info(f"  Significant: {significant}")
            else:
                logger.warning(f"  Missing sample data for statistical tests")
                row["p_value"] = float("nan")
                row["cohen_d"] = float("nan")
                row["significant"] = False

        table_data.append(row)

    # Create DataFrame
    df = pd.DataFrame(table_data)

    # Generate LaTeX table
    latex_table = generate_latex_table(df, alpha_corrected)

    # Save LaTeX
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        f.write(latex_table)

    # Also save CSV for review
    csv_file = output_file.with_suffix(".csv")
    df.to_csv(csv_file, index=False)

    logger.info(f"\nTable saved to: {output_file}")
    logger.info(f"CSV saved to: {csv_file}")

    # Summary statistics
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY STATISTICS")
    logger.info("=" * 80)

    valid_improvements = df[~df["improvement_pct"].isna()]["improvement_pct"]
    if len(valid_improvements) > 0:
        logger.info(f"Improvements: {valid_improvements.min():.1f}% to {valid_improvements.max():.1f}%")
        logger.info(f"Mean improvement: {valid_improvements.mean():.1f}%")

    valid_cohens_d = df[~df["cohen_d"].isna()]["cohen_d"]
    if len(valid_cohens_d) > 0:
        logger.info(f"Cohen's d range: {valid_cohens_d.min():.2f} to {valid_cohens_d.max():.2f}")

    significant_count = df["significant"].sum()
    logger.info(f"Significant results (α={alpha_corrected:.6f}): {significant_count}/{len(df)}")


def generate_latex_table(df: pd.DataFrame, alpha_corrected: float) -> str:
    """Generate LaTeX table matching paper format.

    Args:
        df: DataFrame containing aggregated results
        alpha_corrected: Bonferroni-corrected significance threshold

    Returns:
        LaTeX table string
    """
    latex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Main Results: Latency improvements across 4 architectures and 4 baselines}",
        r"\label{tab:main_results}",
        r"\small",
        r"\begin{tabular}{l c c c c c c}",
        r"\toprule",
        r"Model & Dense & Algo-Only & Linear & Iterative & $\Delta$ (\%) & Sig. \\",
        r"\midrule",
    ]

    model_names = {
        "mamba": "Mamba-2.8B",
        "bert": "BERT-large",
        "resnet": "ResNet-50",
        "gcn": "GCN (arxiv)",
    }

    for _, row in df.iterrows():
        model = model_names.get(row["model"], row["model"])

        # Format latency values (mean ± std)
        dense = format_latency(row.get("dense_mean"), row.get("dense_std"))
        algo = format_latency(row.get("algo_mean"), row.get("algo_std"))
        linear = format_latency(row.get("linear_mean"), row.get("linear_std"))
        iterative = format_latency(row.get("iterative_mean"), row.get("iterative_std"))

        # Format improvement
        improvement = row.get("improvement_pct", float("nan"))
        if not np.isnan(improvement):
            improvement_str = f"\\textbf{{{improvement:.1f}\\%}}"
        else:
            improvement_str = "---"

        # Format significance
        p_value = row.get("p_value", float("nan"))
        cohen_d = row.get("cohen_d", float("nan"))
        significant = row.get("significant", False)

        if not np.isnan(p_value) and not np.isnan(cohen_d):
            if significant:
                # Bold + star for significant results
                sig_str = f"$p<{alpha_corrected:.0e}^*$, $d={cohen_d:.1f}$"
            else:
                sig_str = f"$p={p_value:.3f}$, $d={cohen_d:.1f}$"
        else:
            sig_str = "---"

        latex_lines.append(
            f"{model} & {dense} & {algo} & {linear} & {iterative} & {improvement_str} & {sig_str} \\\\"
        )

    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{0.5em}",
        r"\begin{minipage}{\linewidth}",
        r"\footnotesize",
        r"\textbf{Dense}: No optimization. ",
        r"\textbf{Algo-Only}: HDS sparsity without permutation. ",
        r"\textbf{Linear}: Permute then sparsify. ",
        r"\textbf{Iterative}: Permute, sparsify, then re-permute (our method). ",
        f"All comparisons use paired t-tests with Bonferroni correction ($\\alpha={alpha_corrected:.0e}$). ",
        r"$^*$ indicates statistical significance. ",
        r"Cohen's $d > 1.0$ indicates large effect size.",
        r"\end{minipage}",
        r"\end{table}",
    ])

    return "\n".join(latex_lines)


def format_latency(mean: float, std: float) -> str:
    """Format latency as mean ± std with appropriate precision.

    Args:
        mean: Mean latency (ms)
        std: Standard deviation (ms)

    Returns:
        Formatted string (e.g., "12.3 ± 0.4")
    """
    if np.isnan(mean) or np.isnan(std):
        return "---"

    # Determine precision based on magnitude
    if mean < 1.0:
        return f"{mean:.3f} $\\pm$ {std:.3f}"
    elif mean < 10.0:
        return f"{mean:.2f} $\\pm$ {std:.2f}"
    else:
        return f"{mean:.1f} $\\pm$ {std:.1f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate baseline results into Table 1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Generate Table 1 from experimental results
  python scripts/aggregate_table1.py \\
      --input results/table1 \\
      --output results/paper_tables/table1_main_results.tex

  # Use different significance threshold
  python scripts/aggregate_table1.py \\
      --input results/table1 \\
      --output results/paper_tables/table1_main_results.tex \\
      --alpha 0.01
        """,
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input directory containing experimental results",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output LaTeX file path",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.001,
        help="Significance threshold before Bonferroni correction (default: 0.001)",
    )

    args = parser.parse_args(argv)

    aggregate_table1(args.input, args.output, args.alpha)

    return 0


if __name__ == "__main__":
    sys.exit(main())
