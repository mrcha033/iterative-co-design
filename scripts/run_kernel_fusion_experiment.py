#!/usr/bin/env python3
"""Kernel Fusion Composability Experiment (Paper Table 5)

Tests whether IASP improvements compose with kernel fusion optimizations.

Paper Table 5 structure:
    Configuration         | Latency (ms) | Improvement
    ----------------------|--------------|-------------
    Baseline (no fusion)  |     X        |     —
    Fusion only           |     Y        |    ΔY%
    IASP only             |     Z        |    ΔZ%
    IASP + Fusion         |     W        |    ΔW%

Key question: Does (IASP + Fusion) improvement ≈ ΔY + ΔZ (additive)?
Or do they interfere/synergize?

Usage:
    python scripts/run_kernel_fusion_experiment.py \\
        --model mamba \\
        --config configs/mamba.json \\
        --output results/table5/mamba_fusion_composability.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from icd.core.cost import CostConfig
from icd.core.graph_instrumented import build_instrumented_graph
from icd.core.graph_pytorch import build_csr_from_fx_trace
from icd.core.solver import fit_permutation
from icd.experiments.hf import load_mamba_model, load_hf_sequence_classifier
from icd.measure.cuda_latency import measure_latency_with_stats, compare_latencies
from icd.runtime.apply_pi import apply_pi_to_mamba_hf, apply_pi_to_bert

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def apply_kernel_fusion(model: nn.Module, model_name: str) -> nn.Module:
    """Apply kernel fusion optimizations (torch.compile or TorchScript).

    For this experiment, we use torch.compile with fusion-friendly settings.
    This fuses element-wise operations and reduces kernel launch overhead.

    Args:
        model: PyTorch model
        model_name: Model architecture name

    Returns:
        Compiled model with fused kernels
    """
    logger.info("Applying kernel fusion via torch.compile...")

    # Use torch.compile with max-autotune mode for aggressive fusion
    try:
        compiled_model = torch.compile(
            model,
            mode="max-autotune",  # Aggressive optimization
            fullgraph=False,      # Allow partial graphs
        )
        logger.info("✓ Kernel fusion applied via torch.compile")
        return compiled_model
    except Exception as e:
        logger.warning(f"torch.compile failed: {e}")
        logger.warning("Falling back to TorchScript fusion")

        # Fallback: TorchScript fusion
        try:
            scripted = torch.jit.script(model)
            scripted = torch.jit.freeze(scripted)
            logger.info("✓ Kernel fusion applied via TorchScript")
            return scripted
        except Exception as e2:
            logger.error(f"TorchScript fusion also failed: {e2}")
            logger.warning("Returning model without fusion")
            return model


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


def run_kernel_fusion_experiment(
    model_name: str,
    config_path: Path,
    output_path: Path,
) -> None:
    """Run kernel fusion composability experiment.

    Tests 4 configurations:
    1. Baseline: No fusion, no IASP
    2. Fusion only: Kernel fusion, no IASP
    3. IASP only: Data layout optimization, no fusion
    4. IASP + Fusion: Both optimizations

    Args:
        model_name: Model architecture name
        config_path: Path to config JSON
        output_path: Output JSON path
    """
    logger.info("=" * 80)
    logger.info("KERNEL FUSION COMPOSABILITY EXPERIMENT (Table 5)")
    logger.info("=" * 80)

    # Prepare inputs (same for all configs)
    inputs = prepare_inputs(model_name)

    # -------------------------------------------------------------------------
    # Config 1: Baseline (no fusion, no IASP)
    # -------------------------------------------------------------------------
    logger.info("\n[1/4] Baseline (no fusion, no IASP)...")
    model1, config = load_model_and_config(model_name, config_path)

    pipeline_config = config.get("pipeline", {})
    stats_baseline = measure_latency_with_stats(
        model=model1,
        inputs=inputs,
        num_repeats=pipeline_config.get("repeats", 1000),
        warmup=pipeline_config.get("warmup_iter", 50),
        device="cuda",
    )
    logger.info(f"Baseline latency: {stats_baseline['mean']:.3f} ± {stats_baseline['std']:.3f} ms")

    # -------------------------------------------------------------------------
    # Config 2: Fusion only (no IASP)
    # -------------------------------------------------------------------------
    logger.info("\n[2/4] Fusion only (no IASP)...")
    model2, _ = load_model_and_config(model_name, config_path)
    model2 = apply_kernel_fusion(model2, model_name)

    stats_fusion = measure_latency_with_stats(
        model=model2,
        inputs=inputs,
        num_repeats=pipeline_config.get("repeats", 1000),
        warmup=pipeline_config.get("warmup_iter", 50),
        device="cuda",
    )
    logger.info(f"Fusion-only latency: {stats_fusion['mean']:.3f} ± {stats_fusion['std']:.3f} ms")

    # -------------------------------------------------------------------------
    # Config 3: IASP only (no fusion)
    # -------------------------------------------------------------------------
    logger.info("\n[3/4] IASP only (no fusion)...")
    model3, _ = load_model_and_config(model_name, config_path)

    # Build graph
    W = build_graph_from_model(model3, config)

    # Find permutation
    solver_config = config.get("solver", {})
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
        apply_pi_to_mamba_hf(model3, pi)
    elif model_name == "bert":
        apply_pi_to_bert(model3, pi)

    stats_iasp = measure_latency_with_stats(
        model=model3,
        inputs=inputs,
        num_repeats=pipeline_config.get("repeats", 1000),
        warmup=pipeline_config.get("warmup_iter", 50),
        device="cuda",
    )
    logger.info(f"IASP-only latency: {stats_iasp['mean']:.3f} ± {stats_iasp['std']:.3f} ms")

    # -------------------------------------------------------------------------
    # Config 4: IASP + Fusion (both optimizations)
    # -------------------------------------------------------------------------
    logger.info("\n[4/4] IASP + Fusion (both)...")
    model4, _ = load_model_and_config(model_name, config_path)

    # Apply IASP first
    W = build_graph_from_model(model4, config)
    pi, _ = fit_permutation(
        W=W,
        cfg=cost_config,
        time_budget_s=solver_config.get("time_budget_s", 60.0),
        refine_steps=solver_config.get("refine_steps", 500),
        seed=solver_config.get("rng_seed", 0),
        method=solver_config.get("method", "louvain"),
    )

    if model_name == "mamba":
        apply_pi_to_mamba_hf(model4, pi)
    elif model_name == "bert":
        apply_pi_to_bert(model4, pi)

    # Then apply fusion
    model4 = apply_kernel_fusion(model4, model_name)

    stats_both = measure_latency_with_stats(
        model=model4,
        inputs=inputs,
        num_repeats=pipeline_config.get("repeats", 1000),
        warmup=pipeline_config.get("warmup_iter", 50),
        device="cuda",
    )
    logger.info(f"IASP+Fusion latency: {stats_both['mean']:.3f} ± {stats_both['std']:.3f} ms")

    # -------------------------------------------------------------------------
    # Analysis
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 80)
    logger.info("RESULTS SUMMARY (Table 5)")
    logger.info("=" * 80)

    baseline_lat = stats_baseline['mean']
    fusion_lat = stats_fusion['mean']
    iasp_lat = stats_iasp['mean']
    both_lat = stats_both['mean']

    fusion_improvement = (baseline_lat - fusion_lat) / baseline_lat * 100
    iasp_improvement = (baseline_lat - iasp_lat) / baseline_lat * 100
    both_improvement = (baseline_lat - both_lat) / baseline_lat * 100

    logger.info(f"Baseline (no fusion, no IASP):  {baseline_lat:.3f} ms  (—)")
    logger.info(f"Fusion only:                     {fusion_lat:.3f} ms  ({fusion_improvement:+.1f}%)")
    logger.info(f"IASP only:                       {iasp_lat:.3f} ms  ({iasp_improvement:+.1f}%)")
    logger.info(f"IASP + Fusion:                   {both_lat:.3f} ms  ({both_improvement:+.1f}%)")
    logger.info("")

    # Composability analysis
    expected_combined = fusion_improvement + iasp_improvement
    actual_combined = both_improvement
    composability_ratio = actual_combined / expected_combined if expected_combined > 0 else 0

    logger.info(f"Expected combined improvement (additive): {expected_combined:.1f}%")
    logger.info(f"Actual combined improvement:              {actual_combined:.1f}%")
    logger.info(f"Composability ratio:                      {composability_ratio:.2f}")

    if 0.9 <= composability_ratio <= 1.1:
        logger.info("✓ Improvements compose ADDITIVELY (orthogonal optimizations)")
    elif composability_ratio > 1.1:
        logger.info("✓ Improvements SYNERGIZE (superadditive)")
    else:
        logger.info("⚠ Improvements INTERFERE (subadditive)")

    # Statistical tests
    comparison_iasp_vs_both = compare_latencies(
        stats_iasp['samples'],
        stats_both['samples'],
    )

    logger.info("")
    logger.info(f"Statistical significance (IASP vs IASP+Fusion):")
    logger.info(f"  p-value: {comparison_iasp_vs_both['p_value']:.6f}")
    logger.info(f"  Cohen's d: {comparison_iasp_vs_both['cohen_d']:.3f}")
    logger.info(f"  Significant: {comparison_iasp_vs_both['significant']}")

    # Save results
    result = {
        "experiment": "kernel_fusion_composability",
        "model": model_name,
        "configurations": {
            "baseline": {
                "latency_stats": stats_baseline,
                "improvement_pct": 0.0,
            },
            "fusion_only": {
                "latency_stats": stats_fusion,
                "improvement_pct": fusion_improvement,
            },
            "iasp_only": {
                "latency_stats": stats_iasp,
                "improvement_pct": iasp_improvement,
                "solver_stats": solver_stats,
                "permutation": pi.tolist(),
            },
            "iasp_fusion": {
                "latency_stats": stats_both,
                "improvement_pct": both_improvement,
            },
        },
        "composability_analysis": {
            "expected_combined_improvement_pct": expected_combined,
            "actual_combined_improvement_pct": actual_combined,
            "composability_ratio": composability_ratio,
            "interpretation": (
                "additive" if 0.9 <= composability_ratio <= 1.1
                else "synergistic" if composability_ratio > 1.1
                else "interfering"
            ),
        },
        "statistical_comparison": comparison_iasp_vs_both,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Kernel Fusion Composability Experiment (Table 5)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Example:
  python scripts/run_kernel_fusion_experiment.py \\
      --model mamba \\
      --config configs/mamba.json \\
      --output results/table5/mamba_fusion_composability.json
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

    run_kernel_fusion_experiment(
        model_name=args.model,
        config_path=args.config,
        output_path=args.output,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
