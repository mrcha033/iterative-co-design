#!/usr/bin/env python3
"""Batch Size Sensitivity Analysis (Paper Figure: batch_size_sensitivity)

Tests whether IASP improvements are consistent across different batch sizes.

This experiment sweeps batch sizes [1, 2, 4, 8, 16, 32, 64, 128, 256] and
measures latency for both:
  - Baseline (no IASP)
  - IASP-optimized

Expected result: Improvement % should be relatively stable (~15-20%) across
batch sizes, with slight decrease at batch_size >= 128.

Usage:
    python scripts/run_batch_size_sweep.py \\
        --model mamba \\
        --config configs/mamba.json \\
        --output results/sensitivity/mamba_batch_size_sweep.json
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


def prepare_inputs(model_name: str, batch_size: int, seq_len: int = 512) -> Dict[str, torch.Tensor]:
    """Prepare test inputs with specified batch size."""
    if model_name in ["mamba", "bert"]:
        return {
            "input_ids": torch.randint(0, 1000, (batch_size, seq_len), device="cuda"),
            "attention_mask": torch.ones(batch_size, seq_len, device="cuda"),
        }
    else:
        raise ValueError(f"Unknown model: {model_name}")


def measure_batch_size_sensitivity(
    model_name: str,
    config_path: Path,
    output_path: Path,
    batch_sizes: List[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256],
) -> None:
    """Run batch size sensitivity sweep.

    Args:
        model_name: Model architecture name
        config_path: Path to config JSON
        output_path: Output JSON path
        batch_sizes: List of batch sizes to test
    """
    logger.info("=" * 80)
    logger.info("BATCH SIZE SENSITIVITY ANALYSIS")
    logger.info("=" * 80)

    # Load config
    with open(config_path) as f:
        config = json.load(f)

    pipeline_config = config.get("pipeline", {})
    solver_config = config.get("solver", {})

    results = []

    for batch_size in batch_sizes:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Testing batch_size = {batch_size}")
        logger.info(f"{'=' * 80}")

        # -----------------------------------------------------------------------
        # Baseline (no IASP)
        # -----------------------------------------------------------------------
        logger.info("[1/2] Measuring baseline (no IASP)...")
        model_baseline, _ = load_model_and_config(model_name, config_path)

        inputs = prepare_inputs(model_name, batch_size)

        stats_baseline = measure_latency_with_stats(
            model=model_baseline,
            inputs=inputs,
            num_repeats=pipeline_config.get("repeats", 1000),
            warmup=pipeline_config.get("warmup_iter", 50),
            device="cuda",
        )

        logger.info(f"Baseline latency: {stats_baseline['mean']:.3f} ± {stats_baseline['std']:.3f} ms")

        # -----------------------------------------------------------------------
        # IASP-optimized
        # -----------------------------------------------------------------------
        logger.info("[2/2] Measuring IASP-optimized...")
        model_iasp, _ = load_model_and_config(model_name, config_path)

        # Build graph (use batch_size=1 for graph construction to save time)
        inputs_graph = prepare_inputs(model_name, batch_size=1)
        W = build_graph_from_model(model_iasp, config)

        # Find permutation
        cost_config = CostConfig(
            alpha=solver_config.get("alpha", 1.0),
            beta=solver_config.get("beta", 0.2),
        )

        pi, solver_stats = fit_permutation(
            W=W,
            cfg=cost_config,
            time_budget_s=solver_config.get("time_budget_s", 60.0),
            refine_steps=solver_config.get("refine_steps", 500),
            seed=solver_config.get("rng_seed", 0),
            method=solver_config.get("method", "louvain"),
        )

        # Apply permutation
        if model_name == "mamba":
            apply_pi_to_mamba_hf(model_iasp, pi)
        elif model_name == "bert":
            apply_pi_to_bert(model_iasp, pi)

        # Measure with target batch size
        stats_iasp = measure_latency_with_stats(
            model=model_iasp,
            inputs=inputs,
            num_repeats=pipeline_config.get("repeats", 1000),
            warmup=pipeline_config.get("warmup_iter", 50),
            device="cuda",
        )

        logger.info(f"IASP latency: {stats_iasp['mean']:.3f} ± {stats_iasp['std']:.3f} ms")

        # Compute improvement
        improvement_pct = (stats_baseline['mean'] - stats_iasp['mean']) / stats_baseline['mean'] * 100

        logger.info(f"Improvement: {improvement_pct:.1f}%")

        results.append({
            "batch_size": batch_size,
            "baseline_latency_ms": stats_baseline['mean'],
            "baseline_std_ms": stats_baseline['std'],
            "iasp_latency_ms": stats_iasp['mean'],
            "iasp_std_ms": stats_iasp['std'],
            "improvement_pct": improvement_pct,
            "modularity": solver_stats.get("Q_louvain", None),
        })

        # Clean up models to save memory
        del model_baseline, model_iasp
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Analysis and Visualization
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 80)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 80)

    for r in results:
        logger.info(
            f"Batch {r['batch_size']:3d}: "
            f"Baseline={r['baseline_latency_ms']:.2f}ms, "
            f"IASP={r['iasp_latency_ms']:.2f}ms, "
            f"Improvement={r['improvement_pct']:.1f}%"
        )

    # Check consistency claim
    improvements = [r['improvement_pct'] for r in results]
    mean_improvement = np.mean(improvements)
    std_improvement = np.std(improvements)

    logger.info("")
    logger.info(f"Mean improvement: {mean_improvement:.1f}% ± {std_improvement:.1f}%")

    if std_improvement < 3.0:
        logger.info("✓ Improvements are CONSISTENT across batch sizes")
    else:
        logger.info("⚠ Improvements vary significantly across batch sizes")

    # Generate plot
    plot_path = output_path.with_suffix(".png")
    generate_plot(results, plot_path)
    logger.info(f"\nPlot saved to: {plot_path}")

    # Save results
    output = {
        "experiment": "batch_size_sensitivity",
        "model": model_name,
        "batch_sizes": batch_sizes,
        "results": results,
        "summary": {
            "mean_improvement_pct": mean_improvement,
            "std_improvement_pct": std_improvement,
            "consistent": std_improvement < 3.0,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Results saved to: {output_path}")


def generate_plot(results: List[Dict], output_path: Path) -> None:
    """Generate batch size sensitivity plot."""
    batch_sizes = [r['batch_size'] for r in results]
    baseline_latencies = [r['baseline_latency_ms'] for r in results]
    iasp_latencies = [r['iasp_latency_ms'] for r in results]
    improvements = [r['improvement_pct'] for r in results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot 1: Latency vs batch size
    ax1.plot(batch_sizes, baseline_latencies, 'o-', label='Baseline', linewidth=2)
    ax1.plot(batch_sizes, iasp_latencies, 's-', label='IASP', linewidth=2)
    ax1.set_xlabel('Batch Size', fontsize=12)
    ax1.set_ylabel('Latency (ms)', fontsize=12)
    ax1.set_title('Latency vs Batch Size', fontsize=14)
    ax1.set_xscale('log', base=2)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)

    # Plot 2: Improvement % vs batch size
    ax2.plot(batch_sizes, improvements, 'D-', color='green', linewidth=2)
    ax2.axhline(y=np.mean(improvements), color='red', linestyle='--', label=f'Mean: {np.mean(improvements):.1f}%')
    ax2.set_xlabel('Batch Size', fontsize=12)
    ax2.set_ylabel('Improvement (%)', fontsize=12)
    ax2.set_title('IASP Improvement vs Batch Size', fontsize=14)
    ax2.set_xscale('log', base=2)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Batch Size Sensitivity Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Example:
  python scripts/run_batch_size_sweep.py \\
      --model mamba \\
      --config configs/mamba.json \\
      --output results/sensitivity/mamba_batch_size_sweep.json
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
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256],
        help="List of batch sizes to test (default: 1 2 4 8 16 32 64 128 256)",
    )

    args = parser.parse_args(argv)

    measure_batch_size_sensitivity(
        model_name=args.model,
        config_path=args.config,
        output_path=args.output,
        batch_sizes=args.batch_sizes,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
