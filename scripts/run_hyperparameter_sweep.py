#!/usr/bin/env python3
"""Hyperparameter Sensitivity Analysis (Paper Appendix C.2)

Tests sensitivity to the number of clusters (k) in the clustering algorithm.

This experiment sweeps k values [4, 8, 16, 24, 32, 48, 64] and measures:
  - Latency improvement
  - Modularity (Q)
  - Number of clusters found
  - Solver runtime

Expected result: Latency improvement should plateau for k in [8, 32].

Usage:
    python scripts/run_hyperparameter_sweep.py \\
        --model mamba \\
        --config configs/mamba.json \\
        --output results/sensitivity/mamba_hyperparameter_sweep.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch

from icd.core.cost import CostConfig
from icd.core.graph_instrumented import build_instrumented_graph
from icd.core.graph_pytorch import build_csr_from_fx_trace
from icd.core.solver import fit_permutation
from icd.experiments.hf import load_mamba_model, load_hf_sequence_classifier
from icd.measure.cuda_latency import measure_latency_with_stats
from icd.runtime.apply_pi import apply_pi_to_mamba_hf, apply_pi_to_bert

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_model_and_config(model_name: str, config_path: Path) -> tuple[Any, Dict]:
    """Load model and configuration."""
    with open(config_path) as f:
        config = json.load(f)

    if model_name == "mamba":
        model = load_mamba_model(config.get("model_name", "state-spaces/mamba-2.8b-hf"))
    elif model_name == "bert":
        model = load_hf_sequence_classifier(
            config.get("model_name", "bert-large-uncased"),
            num_labels=2
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = model.to("cuda")
    model.eval()

    return model, config


def build_graph_from_model(model: Any, config: Dict) -> Any:
    """Build co-access graph."""
    graph_config = config.get("graph", {})
    source = graph_config.get("source", "instrumented")

    if source == "instrumented":
        logger.info("Building graph via instrumented profiling...")
        W = build_instrumented_graph(
            model=model,
            temporal_window_ns=graph_config.get("instrumented", {}).get("temporal_window_ns", 100),
            num_samples=graph_config.get("instrumented", {}).get("num_samples", 10),
            cache_line_bytes=graph_config.get("instrumented", {}).get("cache_line_bytes", 64),
        )
    else:
        logger.warning("Using heuristic graph construction")
        W = build_csr_from_fx_trace(model)

    return W


def prepare_inputs(model_name: str) -> Dict[str, torch.Tensor]:
    """Prepare test inputs."""
    if model_name in ["mamba", "bert"]:
        batch_size = 1
        seq_len = 512
        return {
            "input_ids": torch.randint(0, 1000, (batch_size, seq_len), device="cuda"),
            "attention_mask": torch.ones(batch_size, seq_len, device="cuda"),
        }
    else:
        raise ValueError(f"Unknown model: {model_name}")


def measure_hyperparameter_sensitivity(
    model_name: str,
    config_path: Path,
    output_path: Path,
    k_values: List[int] = [4, 8, 16, 24, 32, 48, 64],
) -> None:
    """Run hyperparameter sensitivity sweep.

    Args:
        model_name: Model architecture name
        config_path: Path to config JSON
        output_path: Output JSON path
        k_values: List of k (num_clusters) values to test
    """
    logger.info("=" * 80)
    logger.info("HYPERPARAMETER SENSITIVITY ANALYSIS (k = num_clusters)")
    logger.info("=" * 80)

    # Load config and measure baseline once
    with open(config_path) as f:
        config = json.load(f)

    logger.info("Measuring baseline (no IASP)...")
    model_baseline, _ = load_model_and_config(model_name, config_path)
    inputs = prepare_inputs(model_name)
    pipeline_config = config.get("pipeline", {})

    stats_baseline = measure_latency_with_stats(
        model=model_baseline,
        inputs=inputs,
        num_repeats=pipeline_config.get("repeats", 1000),
        warmup=pipeline_config.get("warmup_iter", 50),
        device="cuda",
    )

    baseline_latency = stats_baseline['mean']
    logger.info(f"Baseline latency: {baseline_latency:.3f} ± {stats_baseline['std']:.3f} ms")

    del model_baseline
    torch.cuda.empty_cache()

    results = []

    for k in k_values:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Testing k = {k} clusters")
        logger.info(f"{'=' * 80}")

        # Load fresh model
        model, _ = load_model_and_config(model_name, config_path)

        # Build graph
        W = build_graph_from_model(model, config)

        # Update solver config with target k
        solver_config = config.get("solver", {})
        cost_config = CostConfig(
            alpha=solver_config.get("alpha", 1.0),
            beta=solver_config.get("beta", 0.2),
        )

        # For Louvain, k is not a direct parameter, but we can control it
        # via resolution parameter. For spectral clustering, we set k directly.
        # Here we use spectral clustering to have direct control over k.
        logger.info(f"Finding permutation with k={k} clusters (spectral clustering)...")

        import time
        start_time = time.time()

        pi, solver_stats = fit_permutation(
            W=W,
            cfg=cost_config,
            time_budget_s=solver_config.get("time_budget_s", 60.0),
            refine_steps=solver_config.get("refine_steps", 500),
            seed=solver_config.get("rng_seed", 0),
            method="spectral",  # Use spectral to control k directly
            k=k,  # Pass k parameter
        )

        solver_time = time.time() - start_time

        # Apply permutation
        if model_name == "mamba":
            apply_pi_to_mamba_hf(model, pi)
        elif model_name == "bert":
            apply_pi_to_bert(model, pi)

        # Measure latency
        stats_iasp = measure_latency_with_stats(
            model=model,
            inputs=inputs,
            num_repeats=pipeline_config.get("repeats", 1000),
            warmup=pipeline_config.get("warmup_iter", 50),
            device="cuda",
        )

        improvement_pct = (baseline_latency - stats_iasp['mean']) / baseline_latency * 100

        logger.info(f"IASP latency: {stats_iasp['mean']:.3f} ± {stats_iasp['std']:.3f} ms")
        logger.info(f"Improvement: {improvement_pct:.1f}%")
        logger.info(f"Modularity: {solver_stats.get('Q_louvain', 'N/A')}")
        logger.info(f"Solver time: {solver_time:.2f}s")

        results.append({
            "k": k,
            "latency_ms": stats_iasp['mean'],
            "latency_std_ms": stats_iasp['std'],
            "improvement_pct": improvement_pct,
            "modularity": solver_stats.get("Q_louvain", None),
            "solver_time_s": solver_time,
        })

        del model
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Analysis and Visualization
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 80)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 80)

    for r in results:
        logger.info(
            f"k={r['k']:2d}: "
            f"Latency={r['latency_ms']:.2f}ms, "
            f"Improvement={r['improvement_pct']:.1f}%, "
            f"Q={r['modularity']:.3f if r['modularity'] else 'N/A'}, "
            f"Time={r['solver_time_s']:.1f}s"
        )

    # Find optimal k range
    improvements = [r['improvement_pct'] for r in results]
    max_improvement = max(improvements)
    plateau_threshold = max_improvement * 0.95  # Within 5% of max

    optimal_k_values = [r['k'] for r in results if r['improvement_pct'] >= plateau_threshold]

    logger.info("")
    logger.info(f"Max improvement: {max_improvement:.1f}%")
    logger.info(f"Optimal k range (≥95% of max): {min(optimal_k_values)} - {max(optimal_k_values)}")

    # Generate plot
    plot_path = output_path.with_suffix(".png")
    generate_plot(results, baseline_latency, plot_path)
    logger.info(f"\nPlot saved to: {plot_path}")

    # Save results
    output = {
        "experiment": "hyperparameter_sensitivity",
        "model": model_name,
        "parameter": "k (num_clusters)",
        "baseline_latency_ms": baseline_latency,
        "k_values": k_values,
        "results": results,
        "summary": {
            "max_improvement_pct": max_improvement,
            "optimal_k_range": [min(optimal_k_values), max(optimal_k_values)],
            "plateau_threshold_pct": plateau_threshold,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Results saved to: {output_path}")


def generate_plot(results: List[Dict], baseline_latency: float, output_path: Path) -> None:
    """Generate hyperparameter sensitivity plot."""
    k_values = [r['k'] for r in results]
    improvements = [r['improvement_pct'] for r in results]
    modularities = [r['modularity'] if r['modularity'] else 0 for r in results]
    solver_times = [r['solver_time_s'] for r in results]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

    # Plot 1: Improvement % vs k
    ax1.plot(k_values, improvements, 'o-', linewidth=2, markersize=8)
    ax1.axhline(y=max(improvements) * 0.95, color='red', linestyle='--',
                label='95% of max', alpha=0.5)
    ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax1.set_ylabel('Improvement (%)', fontsize=12)
    ax1.set_title('Latency Improvement vs Cluster Count', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)

    # Plot 2: Modularity vs k
    ax2.plot(k_values, modularities, 's-', color='orange', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax2.set_ylabel('Modularity (Q)', fontsize=12)
    ax2.set_title('Graph Modularity vs Cluster Count', fontsize=14)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Solver time vs k
    ax3.plot(k_values, solver_times, 'D-', color='green', linewidth=2, markersize=8)
    ax3.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax3.set_ylabel('Solver Time (s)', fontsize=12)
    ax3.set_title('Solver Runtime vs Cluster Count', fontsize=14)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Hyperparameter Sensitivity Analysis (k = num_clusters)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Example:
  python scripts/run_hyperparameter_sweep.py \\
      --model mamba \\
      --config configs/mamba.json \\
      --output results/sensitivity/mamba_hyperparameter_sweep.json
        """,
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=["mamba", "bert"],
        help="Model architecture",
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to config JSON",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output JSON path",
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=[4, 8, 16, 24, 32, 48, 64],
        help="List of k values to test (default: 4 8 16 24 32 48 64)",
    )

    args = parser.parse_args(argv)

    measure_hyperparameter_sensitivity(
        model_name=args.model,
        config_path=args.config,
        output_path=args.output,
        k_values=args.k_values,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
