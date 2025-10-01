#!/usr/bin/env python3
"""Bandwidth Saturation Analysis (Paper Figure: bandwidth_saturation)

Tests when the optimization transitions from memory-bound to compute-bound.

This experiment sweeps (batch_size, sequence_length) pairs and profiles:
  - Memory bandwidth utilization (%)
  - Latency for baseline vs IASP
  - Speedup as function of bandwidth

Expected result: IASP advantage persists until workload becomes compute-bound.

Usage:
    python scripts/run_bandwidth_saturation.py \\
        --model mamba \\
        --config configs/mamba.json \\
        --output results/sensitivity/mamba_bandwidth_saturation.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

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


def measure_bandwidth_utilization(model: Any, inputs: Dict, device: str = "cuda") -> float:
    """Measure memory bandwidth utilization using PyTorch profiler.

    Returns:
        Bandwidth utilization as percentage (0-100)
    """
    try:
        # Use torch.cuda to measure memory traffic
        torch.cuda.reset_peak_memory_stats(device)

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(**inputs)

        # Measure
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        with torch.no_grad():
            for _ in range(100):
                _ = model(**inputs)
        end_event.record()

        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)

        # Get memory stats
        max_memory_mb = torch.cuda.max_memory_allocated(device) / 1e6

        # Estimate bandwidth utilization
        # Peak A100 bandwidth: ~1555 GB/s
        peak_bandwidth_gbs = 1555.0

        # Rough estimate: bytes transferred / time
        bytes_per_iteration = max_memory_mb * 1e6 * 2  # Read + write
        total_bytes = bytes_per_iteration * 100
        elapsed_s = elapsed_ms / 1000.0

        actual_bandwidth_gbs = (total_bytes / 1e9) / elapsed_s
        utilization_pct = (actual_bandwidth_gbs / peak_bandwidth_gbs) * 100

        return min(100.0, utilization_pct)

    except Exception as e:
        logger.warning(f"Could not measure bandwidth: {e}")
        return 0.0


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


def prepare_inputs(model_name: str, batch_size: int, seq_len: int) -> Dict[str, torch.Tensor]:
    """Prepare test inputs."""
    if model_name in ["mamba", "bert"]:
        return {
            "input_ids": torch.randint(0, 1000, (batch_size, seq_len), device="cuda"),
            "attention_mask": torch.ones(batch_size, seq_len, device="cuda"),
        }
    else:
        raise ValueError(f"Unknown model: {model_name}")


def run_bandwidth_saturation_analysis(
    model_name: str,
    config_path: Path,
    output_path: Path,
    workload_configs: List[Tuple[int, int]] = None,
) -> None:
    """Run bandwidth saturation analysis.

    Args:
        model_name: Model architecture name
        config_path: Path to config JSON
        output_path: Output JSON path
        workload_configs: List of (batch_size, seq_len) tuples to test
    """
    logger.info("=" * 80)
    logger.info("BANDWIDTH SATURATION ANALYSIS")
    logger.info("=" * 80)

    # Default workload sweep if not provided
    if workload_configs is None:
        # Sweep from small (memory-bound) to large (compute-bound)
        workload_configs = [
            (1, 128),
            (1, 256),
            (1, 512),
            (2, 512),
            (4, 512),
            (8, 512),
            (1, 1024),
            (2, 1024),
            (4, 1024),
            (1, 2048),
            (2, 2048),
            (8, 1024),
            (16, 512),
        ]

    with open(config_path) as f:
        config = json.load(f)

    pipeline_config = config.get("pipeline", {})
    solver_config = config.get("solver", {})

    results = []

    for batch_size, seq_len in workload_configs:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Testing workload: batch_size={batch_size}, seq_len={seq_len}")
        logger.info(f"{'=' * 80}")

        # -----------------------------------------------------------------------
        # Baseline (no IASP)
        # -----------------------------------------------------------------------
        logger.info("[1/2] Measuring baseline...")
        model_baseline, _ = load_model_and_config(model_name, config_path)
        inputs = prepare_inputs(model_name, batch_size, seq_len)

        # Measure bandwidth
        bandwidth_baseline = measure_bandwidth_utilization(model_baseline, inputs)

        # Measure latency
        stats_baseline = measure_latency_with_stats(
            model=model_baseline,
            inputs=inputs,
            num_repeats=pipeline_config.get("repeats", 1000),
            warmup=pipeline_config.get("warmup_iter", 50),
            device="cuda",
        )

        logger.info(f"Baseline: {stats_baseline['mean']:.3f} ms, BW util: {bandwidth_baseline:.1f}%")

        # -----------------------------------------------------------------------
        # IASP-optimized
        # -----------------------------------------------------------------------
        logger.info("[2/2] Measuring IASP-optimized...")
        model_iasp, _ = load_model_and_config(model_name, config_path)

        # Build graph (use smaller workload for graph construction)
        inputs_graph = prepare_inputs(model_name, batch_size=1, seq_len=512)
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

        # Measure bandwidth
        bandwidth_iasp = measure_bandwidth_utilization(model_iasp, inputs)

        # Measure latency
        stats_iasp = measure_latency_with_stats(
            model=model_iasp,
            inputs=inputs,
            num_repeats=pipeline_config.get("repeats", 1000),
            warmup=pipeline_config.get("warmup_iter", 50),
            device="cuda",
        )

        logger.info(f"IASP: {stats_iasp['mean']:.3f} ms, BW util: {bandwidth_iasp:.1f}%")

        # Compute speedup
        speedup_pct = (stats_baseline['mean'] - stats_iasp['mean']) / stats_baseline['mean'] * 100

        # Determine regime
        is_memory_bound = bandwidth_baseline > 60.0
        regime = "memory-bound" if is_memory_bound else "compute-bound"

        logger.info(f"Speedup: {speedup_pct:.1f}%, Regime: {regime}")

        results.append({
            "batch_size": batch_size,
            "seq_len": seq_len,
            "baseline_latency_ms": stats_baseline['mean'],
            "baseline_std_ms": stats_baseline['std'],
            "baseline_bandwidth_pct": bandwidth_baseline,
            "iasp_latency_ms": stats_iasp['mean'],
            "iasp_std_ms": stats_iasp['std'],
            "iasp_bandwidth_pct": bandwidth_iasp,
            "speedup_pct": speedup_pct,
            "regime": regime,
            "modularity": solver_stats.get("Q_louvain", None),
        })

        # Clean up
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
            f"BS={r['batch_size']:2d}, SeqLen={r['seq_len']:4d}: "
            f"BW={r['baseline_bandwidth_pct']:5.1f}%, "
            f"Speedup={r['speedup_pct']:5.1f}%, "
            f"Regime={r['regime']}"
        )

    # Find saturation point
    memory_bound_results = [r for r in results if r['regime'] == 'memory-bound']
    compute_bound_results = [r for r in results if r['regime'] == 'compute-bound']

    logger.info("")
    logger.info(f"Memory-bound workloads: {len(memory_bound_results)}")
    logger.info(f"Compute-bound workloads: {len(compute_bound_results)}")

    if memory_bound_results:
        avg_speedup_memory = np.mean([r['speedup_pct'] for r in memory_bound_results])
        logger.info(f"Average speedup (memory-bound): {avg_speedup_memory:.1f}%")

    if compute_bound_results:
        avg_speedup_compute = np.mean([r['speedup_pct'] for r in compute_bound_results])
        logger.info(f"Average speedup (compute-bound): {avg_speedup_compute:.1f}%")

    # Generate plot
    plot_path = output_path.with_suffix(".pdf")
    generate_plot(results, plot_path)
    logger.info(f"\nPlot saved to: {plot_path}")

    # Save results
    output = {
        "experiment": "bandwidth_saturation",
        "model": model_name,
        "workload_configs": workload_configs,
        "results": results,
        "summary": {
            "num_memory_bound": len(memory_bound_results),
            "num_compute_bound": len(compute_bound_results),
            "avg_speedup_memory_bound": avg_speedup_memory if memory_bound_results else 0.0,
            "avg_speedup_compute_bound": avg_speedup_compute if compute_bound_results else 0.0,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Results saved to: {output_path}")


def generate_plot(results: List[Dict], output_path: Path) -> None:
    """Generate bandwidth saturation plot."""
    bandwidths = [r['baseline_bandwidth_pct'] for r in results]
    speedups = [r['speedup_pct'] for r in results]
    regimes = [r['regime'] for r in results]

    # Separate by regime
    memory_bound_idx = [i for i, r in enumerate(regimes) if r == 'memory-bound']
    compute_bound_idx = [i for i, r in enumerate(regimes) if r == 'compute-bound']

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot memory-bound workloads
    if memory_bound_idx:
        ax.scatter(
            [bandwidths[i] for i in memory_bound_idx],
            [speedups[i] for i in memory_bound_idx],
            marker='o',
            s=100,
            label='Memory-bound',
            color='blue',
            alpha=0.7
        )

    # Plot compute-bound workloads
    if compute_bound_idx:
        ax.scatter(
            [bandwidths[i] for i in compute_bound_idx],
            [speedups[i] for i in compute_bound_idx],
            marker='s',
            s=100,
            label='Compute-bound',
            color='red',
            alpha=0.7
        )

    # Add threshold line at 60%
    ax.axvline(x=60, color='gray', linestyle='--', linewidth=2, alpha=0.5, label='60% threshold')

    ax.set_xlabel('Memory Bandwidth Utilization (%)', fontsize=12)
    ax.set_ylabel('IASP Speedup (%)', fontsize=12)
    ax.set_title('Co-Design Speedup vs Memory Bandwidth Utilization', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bandwidth Saturation Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Example:
  python scripts/run_bandwidth_saturation.py \\
      --model mamba \\
      --config configs/mamba.json \\
      --output results/sensitivity/mamba_bandwidth_saturation.json
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

    args = parser.parse_args(argv)

    run_bandwidth_saturation_analysis(
        model_name=args.model,
        config_path=args.config,
        output_path=args.output,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
