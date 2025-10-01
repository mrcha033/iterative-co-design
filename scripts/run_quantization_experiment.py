#!/usr/bin/env python3
"""Quantization Co-Design Experiment (Paper Figure: quant_results)

Tests the critical claim: Does iteration provide value beyond just applying
optimizations sequentially?

Three strategies compared:
  1. Quant-then-Permute: Apply PTQ, then find optimal permutation
  2. Permute-then-Quant: Find permutation, then apply PTQ
  3. Iterative (Ours): Permute → Quant → RE-PERMUTE

Usage:
    python scripts/run_quantization_experiment.py \\
        --model mamba \\
        --config configs/mamba_3b.json \\
        --output results/quantization/mamba_quant_experiment.json
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

from icd.adapters.quant import apply_post_training_quantization
from icd.core.cost import CostConfig
from icd.core.solver import fit_permutation
from icd.measure.cuda_latency import measure_latency_with_stats, compare_latencies
from icd.runtime.apply_pi import apply_pi_to_mamba_hf, apply_pi_to_bert
from icd.experiments.hf import load_mamba_model, load_hf_sequence_classifier

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
    """Build co-access graph (use instrumented for real patterns)."""
    graph_config = config.get("graph", {})
    source = graph_config.get("source", "instrumented")

    if source == "instrumented":
        from icd.core.graph_instrumented import build_instrumented_graph

        logger.info("Building graph via instrumented profiling...")
        W = build_instrumented_graph(
            model=model,
            temporal_window_ns=graph_config.get("instrumented", {}).get("temporal_window_ns", 100),
            num_samples=graph_config.get("instrumented", {}).get("num_samples", 10),
            cache_line_bytes=graph_config.get("instrumented", {}).get("cache_line_bytes", 64),
        )
    else:
        from icd.core.graph_pytorch import build_csr_from_fx_trace
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


def strategy_1_quant_then_permute(
    model: Any,
    config: Dict,
    model_name: str,
) -> tuple[Any, Dict]:
    """Strategy 1: Apply quantization first, THEN find permutation.

    This tests: Can we find a good layout for an already-quantized model?
    """
    logger.info("=" * 80)
    logger.info("STRATEGY 1: QUANT-THEN-PERMUTE")
    logger.info("=" * 80)

    # 1. Apply quantization to FP32 model
    logger.info("Step 1: Applying PTQ to FP32 model...")
    quant_config = config.get("transform", {}).get("quant", {})
    model = apply_post_training_quantization(
        model,
        dtype=quant_config.get("dtype", "int8"),
        calibration_samples=quant_config.get("calibration_samples", 100),
    )

    # 2. Build graph from quantized model
    logger.info("Step 2: Building graph from quantized model...")
    W = build_graph_from_model(model, config)

    # 3. Find permutation for quantized model
    logger.info("Step 3: Finding permutation for quantized model...")
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

    # 4. Apply permutation
    logger.info("Step 4: Applying permutation to quantized model...")
    if model_name == "mamba":
        apply_pi_to_mamba_hf(model, pi)
    elif model_name == "bert":
        apply_pi_to_bert(model, pi)

    # 5. Measure final latency
    logger.info("Step 5: Measuring final latency...")
    inputs = prepare_inputs(model_name)
    pipeline_config = config.get("pipeline", {})
    stats = measure_latency_with_stats(
        model=model,
        inputs=inputs,
        num_repeats=pipeline_config.get("repeats", 1000),
        warmup=pipeline_config.get("warmup_iter", 50),
        device="cuda",
    )

    result = {
        "strategy": "quant_then_permute",
        "latency_stats": stats,
        "solver_stats": solver_stats,
        "permutation": pi,
    }

    logger.info(f"Strategy 1 latency: {stats['mean']:.3f} ± {stats['std']:.3f} ms")
    return model, result


def strategy_2_permute_then_quant(
    model: Any,
    config: Dict,
    model_name: str,
) -> tuple[Any, Dict]:
    """Strategy 2: Find permutation first, THEN apply quantization.

    This is the "linear pipeline" - optimize once, then transform.
    Paper hypothesis: This should be suboptimal vs. iterative.
    """
    logger.info("=" * 80)
    logger.info("STRATEGY 2: PERMUTE-THEN-QUANT (Linear Pipeline)")
    logger.info("=" * 80)

    # 1. Build graph from FP32 model
    logger.info("Step 1: Building graph from FP32 model...")
    W = build_graph_from_model(model, config)

    # 2. Find permutation for FP32 model
    logger.info("Step 2: Finding permutation for FP32 model...")
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

    # 3. Apply permutation to FP32 model
    logger.info("Step 3: Applying permutation to FP32 model...")
    if model_name == "mamba":
        apply_pi_to_mamba_hf(model, pi)
    elif model_name == "bert":
        apply_pi_to_bert(model, pi)

    # 4. Apply quantization (DO NOT re-permute!)
    logger.info("Step 4: Applying PTQ (NO re-permutation)...")
    quant_config = config.get("transform", {}).get("quant", {})
    model = apply_post_training_quantization(
        model,
        dtype=quant_config.get("dtype", "int8"),
        calibration_samples=quant_config.get("calibration_samples", 100),
    )

    # 5. Measure final latency
    logger.info("Step 5: Measuring final latency...")
    inputs = prepare_inputs(model_name)
    pipeline_config = config.get("pipeline", {})
    stats = measure_latency_with_stats(
        model=model,
        inputs=inputs,
        num_repeats=pipeline_config.get("repeats", 1000),
        warmup=pipeline_config.get("warmup_iter", 50),
        device="cuda",
    )

    result = {
        "strategy": "permute_then_quant",
        "latency_stats": stats,
        "solver_stats": solver_stats,
        "permutation": pi,
    }

    logger.info(f"Strategy 2 latency: {stats['mean']:.3f} ± {stats['std']:.3f} ms")
    return model, result


def strategy_3_iterative(
    model: Any,
    config: Dict,
    model_name: str,
) -> tuple[Any, Dict]:
    """Strategy 3: Permute → Quant → RE-PERMUTE (Iterative Co-Design).

    This is OUR METHOD. The key difference: we re-optimize layout AFTER
    quantization changes the activation patterns.

    Paper hypothesis: The re-permutation step should provide ~12% additional gain.
    """
    logger.info("=" * 80)
    logger.info("STRATEGY 3: ITERATIVE CO-DESIGN (Permute → Quant → Re-Permute)")
    logger.info("=" * 80)

    # 1. Build initial graph from FP32 model
    logger.info("Step 1: Building initial graph from FP32 model...")
    W0 = build_graph_from_model(model, config)

    # 2. Find initial permutation for FP32 model
    logger.info("Step 2: Finding initial permutation...")
    solver_config = config.get("solver", {})
    cost_config = CostConfig(
        alpha=solver_config.get("alpha", 1.0),
        beta=solver_config.get("beta", 0.2),
    )

    pi0, solver_stats0 = fit_permutation(
        W=W0,
        cfg=cost_config,
        time_budget_s=solver_config.get("time_budget_s", 60.0),
        refine_steps=solver_config.get("refine_steps", 500),
        seed=solver_config.get("rng_seed", 0),
        method=solver_config.get("method", "louvain"),
    )

    # 3. Apply initial permutation
    logger.info("Step 3: Applying initial permutation...")
    if model_name == "mamba":
        apply_pi_to_mamba_hf(model, pi0)
    elif model_name == "bert":
        apply_pi_to_bert(model, pi0)

    # 4. Apply quantization
    logger.info("Step 4: Applying PTQ...")
    quant_config = config.get("transform", {}).get("quant", {})
    model = apply_post_training_quantization(
        model,
        dtype=quant_config.get("dtype", "int8"),
        calibration_samples=quant_config.get("calibration_samples", 100),
    )

    # 5. RE-BUILD graph from quantized model (CRITICAL!)
    logger.info("Step 5: RE-BUILDING graph from quantized model...")
    W1 = build_graph_from_model(model, config)

    # 6. RE-PERMUTE based on new graph
    logger.info("Step 6: Finding RE-PERMUTATION...")
    pi1, solver_stats1 = fit_permutation(
        W=W1,
        cfg=cost_config,
        time_budget_s=solver_config.get("time_budget_s", 60.0),
        refine_steps=solver_config.get("refine_steps", 500),
        seed=solver_config.get("rng_seed", 0),
        method=solver_config.get("method", "louvain"),
    )

    # 7. Apply re-permutation
    logger.info("Step 7: Applying re-permutation...")
    if model_name == "mamba":
        apply_pi_to_mamba_hf(model, pi1)
    elif model_name == "bert":
        apply_pi_to_bert(model, pi1)

    # 8. Measure final latency
    logger.info("Step 8: Measuring final latency...")
    inputs = prepare_inputs(model_name)
    pipeline_config = config.get("pipeline", {})
    stats = measure_latency_with_stats(
        model=model,
        inputs=inputs,
        num_repeats=pipeline_config.get("repeats", 1000),
        warmup=pipeline_config.get("warmup_iter", 50),
        device="cuda",
    )

    result = {
        "strategy": "iterative",
        "latency_stats": stats,
        "initial_solver_stats": solver_stats0,
        "final_solver_stats": solver_stats1,
        "initial_permutation": pi0,
        "final_permutation": pi1,
        "modularity_improvement": solver_stats1.get("Q_louvain", 0) - solver_stats0.get("Q_louvain", 0),
    }

    logger.info(f"Strategy 3 latency: {stats['mean']:.3f} ± {stats['std']:.3f} ms")
    logger.info(f"Modularity: Initial={solver_stats0.get('Q_louvain', 'N/A')}, Final={solver_stats1.get('Q_louvain', 'N/A')}")
    return model, result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Quantization Co-Design Experiment (Paper Figure)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Example:
  python scripts/run_quantization_experiment.py \\
      --model mamba \\
      --config configs/mamba_3b.json \\
      --output results/quantization/mamba_quant_experiment.json
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

    logger.info("=" * 80)
    logger.info("QUANTIZATION CO-DESIGN EXPERIMENT")
    logger.info("Tests: Does iteration provide value with quantization?")
    logger.info("=" * 80)

    # Load model (need 3 separate copies for 3 strategies)
    logger.info("Loading model for Strategy 1...")
    model1, config = load_model_and_config(args.model, args.config)

    logger.info("Loading model for Strategy 2...")
    model2, _ = load_model_and_config(args.model, args.config)

    logger.info("Loading model for Strategy 3...")
    model3, _ = load_model_and_config(args.model, args.config)

    # Run all three strategies
    _, result1 = strategy_1_quant_then_permute(model1, config, args.model)
    _, result2 = strategy_2_permute_then_quant(model2, config, args.model)
    _, result3 = strategy_3_iterative(model3, config, args.model)

    # Compare results
    logger.info("\n" + "=" * 80)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 80)

    latency1 = result1["latency_stats"]["mean"]
    latency2 = result2["latency_stats"]["mean"]
    latency3 = result3["latency_stats"]["mean"]

    logger.info(f"Strategy 1 (Quant→Permute):       {latency1:.3f} ms")
    logger.info(f"Strategy 2 (Permute→Quant):       {latency2:.3f} ms")
    logger.info(f"Strategy 3 (Iterative/Ours):      {latency3:.3f} ms")
    logger.info("")

    improvement_vs_1 = (latency1 - latency3) / latency1 * 100
    improvement_vs_2 = (latency2 - latency3) / latency2 * 100

    logger.info(f"Improvement over Strategy 1: {improvement_vs_1:.1f}%")
    logger.info(f"Improvement over Strategy 2: {improvement_vs_2:.1f}%")
    logger.info("")

    # Statistical comparison
    comparison = compare_latencies(
        result2["latency_stats"]["samples"],
        result3["latency_stats"]["samples"],
    )

    logger.info(f"Statistical significance (Strategy 2 vs 3):")
    logger.info(f"  p-value: {comparison['p_value']:.6f}")
    logger.info(f"  Cohen's d: {comparison['cohen_d']:.3f}")
    logger.info(f"  Significant: {comparison['significant']}")

    # Save full results
    output = {
        "experiment": "quantization_codesign",
        "model": args.model,
        "strategies": {
            "quant_then_permute": result1,
            "permute_then_quant": result2,
            "iterative": result3,
        },
        "summary": {
            "latency_ms": {
                "quant_then_permute": latency1,
                "permute_then_quant": latency2,
                "iterative": latency3,
            },
            "improvement_pct": {
                "vs_strategy1": improvement_vs_1,
                "vs_strategy2": improvement_vs_2,
            },
            "statistical_comparison": comparison,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
