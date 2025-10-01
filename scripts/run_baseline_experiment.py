#!/usr/bin/env python3
"""Run a single baseline experiment (dense/algo/linear/iterative).

This script implements the 4 baselines described in the paper (Table 1):
  1. Dense: No optimization (original model)
  2. Algorithm-Only: Apply HDS sparsity without permutation
  3. Linear Pipeline: Permute → Sparsify (no re-permutation)
  4. Iterative Co-Design: Permute → Sparsify → RE-PERMUTE

Usage:
    python scripts/run_baseline_experiment.py \\
        --baseline iterative \\
        --model mamba \\
        --config configs/mamba_3b.json \\
        --output results/table1/mamba/iterative_metrics.json
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

from icd.adapters.sparsity import apply_structured_sparsity
from icd.core.cost import CostConfig
from icd.core.graph import build_csr_from_dict
from icd.core.solver import fit_permutation
from icd.measure.cuda_latency import measure_latency_with_stats
from icd.runtime.apply_pi import apply_pi_to_mamba_hf, apply_pi_to_bert
from icd.experiments.hf import load_hf_sequence_classifier

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
        from icd.experiments.hf import load_mamba_model
        model = load_mamba_model(config.get("model_name", "state-spaces/mamba-130m"))
    elif model_name == "bert":
        model = load_hf_sequence_classifier(
            config.get("model_name", "bert-large-uncased"),
            num_labels=2
        )
    elif model_name == "resnet":
        from icd.experiments.graph_loaders import load_resnet50
        model = load_resnet50(pretrained=True)
    elif model_name == "gcn":
        from icd.experiments.graph_loaders import load_gcn_ogbn_arxiv
        model = load_gcn_ogbn_arxiv()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = model.to("cuda")
    model.eval()

    return model, config


def build_graph_from_model(model: Any, config: Dict) -> Any:
    """Build co-access graph from model.

    Uses instrumented profiling for REAL co-access patterns as per paper.
    """
    graph_config = config.get("graph", {})
    source = graph_config.get("source", "instrumented")

    if source == "instrumented":
        from icd.core.graph_instrumented import build_instrumented_graph

        logger.info("Building REAL co-access graph via instrumented profiling...")
        W = build_instrumented_graph(
            model=model,
            temporal_window_ns=graph_config.get("instrumented", {}).get("temporal_window_ns", 100),
            num_samples=graph_config.get("instrumented", {}).get("num_samples", 10),
            cache_line_bytes=graph_config.get("instrumented", {}).get("cache_line_bytes", 64),
        )
    elif source == "pytorch":
        from icd.core.graph_pytorch import build_csr_from_fx_trace

        logger.warning("Using HEURISTIC graph construction (not real co-access patterns)")
        W = build_csr_from_fx_trace(model)
    elif source == "mock":
        from icd.core.graph import build_mock_graph

        mock_config = graph_config.get("mock", {})
        W = build_mock_graph(
            d=mock_config.get("d", 256),
            blocks=mock_config.get("blocks", 4),
            noise=mock_config.get("noise", 0.02),
            seed=mock_config.get("seed", 0),
        )
    else:
        raise ValueError(f"Unknown graph source: {source}")

    return W


def prepare_test_inputs(model: Any, model_name: str) -> Dict[str, torch.Tensor]:
    """Prepare test inputs for latency measurement."""
    if model_name in ["mamba", "bert"]:
        # NLP models: token IDs
        batch_size = 1
        seq_len = 512
        inputs = {
            "input_ids": torch.randint(0, 1000, (batch_size, seq_len), device="cuda"),
            "attention_mask": torch.ones(batch_size, seq_len, device="cuda"),
        }
    elif model_name == "resnet":
        # Vision model: image tensor
        batch_size = 1
        inputs = {
            "pixel_values": torch.randn(batch_size, 3, 224, 224, device="cuda")
        }
    elif model_name == "gcn":
        # Graph model: node features + edge index
        from icd.experiments.graph_loaders import get_ogbn_arxiv_sample
        inputs = get_ogbn_arxiv_sample(device="cuda")
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return inputs


def run_dense_baseline(
    model: Any,
    config: Dict,
    model_name: str,
    output_path: Path,
) -> Dict[str, Any]:
    """Baseline 1: No optimization (original model)."""
    logger.info("=" * 80)
    logger.info("BASELINE 1: DENSE (No Optimization)")
    logger.info("=" * 80)

    # Prepare inputs
    inputs = prepare_test_inputs(model, model_name)

    # Measure latency
    pipeline_config = config.get("pipeline", {})
    stats = measure_latency_with_stats(
        model=model,
        inputs=inputs,
        num_repeats=pipeline_config.get("repeats", 1000),
        warmup=pipeline_config.get("warmup_iter", 50),
        device="cuda",
    )

    result = {
        "baseline": "dense",
        "model": model_name,
        "latency_stats": stats,
        "description": "Original model, no optimizations",
    }

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Dense baseline: {stats['mean']:.3f} ± {stats['std']:.3f} ms")
    return result


def run_algo_only_baseline(
    model: Any,
    config: Dict,
    model_name: str,
    output_path: Path,
) -> Dict[str, Any]:
    """Baseline 2: Apply HDS sparsity, no permutation."""
    logger.info("=" * 80)
    logger.info("BASELINE 2: ALGORITHM-ONLY (HDS Sparsity)")
    logger.info("=" * 80)

    # Apply sparsity
    sparsity_config = config.get("transform", {}).get("sparsity", {})
    rate = sparsity_config.get("rate", 0.5)
    pattern = sparsity_config.get("pattern", "2:4")

    logger.info(f"Applying {pattern} structured sparsity at rate {rate}")
    model = apply_structured_sparsity(model, rate=rate, pattern=pattern)

    # Measure
    inputs = prepare_test_inputs(model, model_name)
    pipeline_config = config.get("pipeline", {})
    stats = measure_latency_with_stats(
        model=model,
        inputs=inputs,
        num_repeats=pipeline_config.get("repeats", 1000),
        warmup=pipeline_config.get("warmup_iter", 50),
        device="cuda",
    )

    result = {
        "baseline": "algo_only",
        "model": model_name,
        "latency_stats": stats,
        "sparsity_rate": rate,
        "sparsity_pattern": pattern,
        "description": "HDS sparsity without permutation",
    }

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Algo-only baseline: {stats['mean']:.3f} ± {stats['std']:.3f} ms")
    return result


def run_linear_pipeline(
    model: Any,
    config: Dict,
    model_name: str,
    output_path: Path,
) -> Dict[str, Any]:
    """Baseline 3: Permute → Sparsify (no re-permutation)."""
    logger.info("=" * 80)
    logger.info("BASELINE 3: LINEAR PIPELINE")
    logger.info("Permute → Sparsify (NO re-permutation)")
    logger.info("=" * 80)

    # 1. Build initial graph
    logger.info("Step 1: Building co-access graph...")
    W = build_graph_from_model(model, config)

    # 2. Find initial permutation
    logger.info("Step 2: Computing initial permutation...")
    solver_config = config.get("solver", {})
    cost_config = CostConfig(
        alpha=solver_config.get("alpha", 1.0),
        beta=solver_config.get("beta", 0.2),
        gamma_stability=solver_config.get("gamma_stability", 0.1),
        mu=solver_config.get("mu", 0.5),
    )

    pi0, solver_stats = fit_permutation(
        W=W,
        cfg=cost_config,
        time_budget_s=solver_config.get("time_budget_s", 60.0),
        refine_steps=solver_config.get("refine_steps", 500),
        seed=solver_config.get("rng_seed", 0),
        method=solver_config.get("method", "louvain"),
    )

    # 3. Apply permutation
    logger.info("Step 3: Applying permutation to model...")
    if model_name == "mamba":
        apply_pi_to_mamba_hf(model, pi0)
    elif model_name == "bert":
        apply_pi_to_bert(model, pi0)
    elif model_name == "resnet":
        from icd.runtime.apply_pi import apply_pi_to_resnet
        apply_pi_to_resnet(model, pi0)
    elif model_name == "gcn":
        from icd.runtime.apply_pi import apply_pi_to_gcn
        apply_pi_to_gcn(model, pi0)
    else:
        logger.warning(f"Permutation application not implemented for {model_name}")

    # 4. Apply sparsity (DO NOT re-permute!)
    logger.info("Step 4: Applying sparsity (NO re-permutation)...")
    sparsity_config = config.get("transform", {}).get("sparsity", {})
    model = apply_structured_sparsity(
        model,
        rate=sparsity_config.get("rate", 0.5),
        pattern=sparsity_config.get("pattern", "2:4"),
    )

    # 5. Measure
    logger.info("Step 5: Measuring latency...")
    inputs = prepare_test_inputs(model, model_name)
    pipeline_config = config.get("pipeline", {})
    stats = measure_latency_with_stats(
        model=model,
        inputs=inputs,
        num_repeats=pipeline_config.get("repeats", 1000),
        warmup=pipeline_config.get("warmup_iter", 50),
        device="cuda",
    )

    result = {
        "baseline": "linear",
        "model": model_name,
        "latency_stats": stats,
        "solver_stats": solver_stats,
        "initial_permutation": pi0,
        "description": "Linear pipeline: Permute then Sparsify",
    }

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Linear pipeline: {stats['mean']:.3f} ± {stats['std']:.3f} ms")
    logger.info(f"Initial modularity Q: {solver_stats.get('Q_louvain', 'N/A')}")
    return result


def run_iterative_codesign(
    model: Any,
    config: Dict,
    model_name: str,
    output_path: Path,
) -> Dict[str, Any]:
    """Baseline 4: Permute → Sparsify → RE-PERMUTE (our method)."""
    logger.info("=" * 80)
    logger.info("BASELINE 4: ITERATIVE CO-DESIGN")
    logger.info("Permute → Sparsify → RE-PERMUTE")
    logger.info("=" * 80)

    # 1. Build initial graph
    logger.info("Step 1: Building initial co-access graph...")
    W0 = build_graph_from_model(model, config)

    # 2. Find initial permutation
    logger.info("Step 2: Computing initial permutation...")
    solver_config = config.get("solver", {})
    cost_config = CostConfig(
        alpha=solver_config.get("alpha", 1.0),
        beta=solver_config.get("beta", 0.2),
        gamma_stability=solver_config.get("gamma_stability", 0.1),
        mu=solver_config.get("mu", 0.5),
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
    elif model_name == "resnet":
        from icd.runtime.apply_pi import apply_pi_to_resnet
        apply_pi_to_resnet(model, pi0)
    elif model_name == "gcn":
        from icd.runtime.apply_pi import apply_pi_to_gcn
        apply_pi_to_gcn(model, pi0)
    else:
        logger.warning(f"Permutation application not implemented for {model_name}")

    # 4. Apply sparsity
    logger.info("Step 4: Applying sparsity...")
    sparsity_config = config.get("transform", {}).get("sparsity", {})
    model = apply_structured_sparsity(
        model,
        rate=sparsity_config.get("rate", 0.5),
        pattern=sparsity_config.get("pattern", "2:4"),
    )

    # 5. RE-BUILD graph after sparsity (CRITICAL DIFFERENCE)
    logger.info("Step 5: RE-BUILDING graph after sparsity transformation...")
    W1 = build_graph_from_model(model, config)

    # 6. RE-PERMUTE
    logger.info("Step 6: Computing RE-PERMUTATION...")
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
    elif model_name == "resnet":
        from icd.runtime.apply_pi import apply_pi_to_resnet
        apply_pi_to_resnet(model, pi1)
    elif model_name == "gcn":
        from icd.runtime.apply_pi import apply_pi_to_gcn
        apply_pi_to_gcn(model, pi1)

    # 8. Measure
    logger.info("Step 8: Measuring final latency...")
    inputs = prepare_test_inputs(model, model_name)
    pipeline_config = config.get("pipeline", {})
    stats = measure_latency_with_stats(
        model=model,
        inputs=inputs,
        num_repeats=pipeline_config.get("repeats", 1000),
        warmup=pipeline_config.get("warmup_iter", 50),
        device="cuda",
    )

    result = {
        "baseline": "iterative",
        "model": model_name,
        "latency_stats": stats,
        "initial_solver_stats": solver_stats0,
        "final_solver_stats": solver_stats1,
        "initial_permutation": pi0,
        "final_permutation": pi1,
        "modularity_improvement": solver_stats1.get("Q_louvain", 0) - solver_stats0.get("Q_louvain", 0),
        "description": "Iterative co-design: Permute then Sparsify then RE-PERMUTE",
    }

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Iterative co-design: {stats['mean']:.3f} ± {stats['std']:.3f} ms")
    logger.info(f"Initial Q: {solver_stats0.get('Q_louvain', 'N/A')}")
    logger.info(f"Final Q: {solver_stats1.get('Q_louvain', 'N/A')}")
    logger.info(f"Modularity improvement: {result['modularity_improvement']:.4f}")

    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run single baseline experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Dense baseline
  python scripts/run_baseline_experiment.py --baseline dense --model mamba \\
      --config configs/mamba_3b.json --output results/table1/mamba/dense.json

  # Iterative co-design
  python scripts/run_baseline_experiment.py --baseline iterative --model mamba \\
      --config configs/mamba_3b.json --output results/table1/mamba/iterative.json
        """,
    )

    parser.add_argument(
        "--baseline",
        required=True,
        choices=["dense", "algo", "linear", "iterative"],
        help="Which baseline to run",
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["mamba", "bert", "resnet", "gcn"],
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

    # Load model
    logger.info(f"Loading {args.model} model...")
    model, config = load_model_and_config(args.model, args.config)

    # Run baseline
    if args.baseline == "dense":
        run_dense_baseline(model, config, args.model, args.output)
    elif args.baseline == "algo":
        run_algo_only_baseline(model, config, args.model, args.output)
    elif args.baseline == "linear":
        run_linear_pipeline(model, config, args.model, args.output)
    elif args.baseline == "iterative":
        run_iterative_codesign(model, config, args.model, args.output)

    logger.info(f"Results saved to: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
