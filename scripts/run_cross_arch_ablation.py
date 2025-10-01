#!/usr/bin/env python3
"""Cross-Architecture Ablation Study (Paper lines 370-373)

Compares modularity optimization vs TSP vs random permutations across all architectures.

This validates that modularity-based optimization is universally beneficial.

Usage:
    python scripts/run_cross_arch_ablation.py \\
        --models mamba bert resnet gcn \\
        --config-dir configs \\
        --output results/ablations/cross_arch_ablation.json
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
import torch

from icd.core.cost import CostConfig
from icd.core.solver import fit_permutation
from icd.experiments.hf import load_mamba_model, load_hf_sequence_classifier
from icd.measure.cuda_latency import measure_latency_with_stats
from icd.runtime.apply_pi import apply_pi_to_mamba_hf, apply_pi_to_bert, apply_pi_to_resnet, apply_pi_to_gcn

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
    """Build co-access graph."""
    graph_config = config.get("graph", {})
    source = graph_config.get("source", "instrumented")

    if source == "instrumented":
        from icd.core.graph_instrumented import build_instrumented_graph
        W = build_instrumented_graph(
            model=model,
            temporal_window_ns=graph_config.get("instrumented", {}).get("temporal_window_ns", 100),
            num_samples=graph_config.get("instrumented", {}).get("num_samples", 10),
            cache_line_bytes=graph_config.get("instrumented", {}).get("cache_line_bytes", 64),
        )
    else:
        from icd.core.graph_pytorch import build_csr_from_fx_trace
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
    elif model_name == "resnet":
        batch_size = 1
        return {
            "pixel_values": torch.randn(batch_size, 3, 224, 224, device="cuda")
        }
    elif model_name == "gcn":
        from icd.experiments.graph_loaders import get_ogbn_arxiv_sample
        return get_ogbn_arxiv_sample(device="cuda")
    else:
        raise ValueError(f"Unknown model: {model_name}")


def apply_permutation(model: Any, pi: np.ndarray, model_name: str) -> None:
    """Apply permutation to model."""
    if model_name == "mamba":
        apply_pi_to_mamba_hf(model, pi)
    elif model_name == "bert":
        apply_pi_to_bert(model, pi)
    elif model_name == "resnet":
        apply_pi_to_resnet(model, pi)
    elif model_name == "gcn":
        apply_pi_to_gcn(model, pi)


def run_cross_arch_ablation(
    models: List[str],
    config_dir: Path,
    output_path: Path,
) -> None:
    """Run cross-architecture ablation.

    Args:
        models: List of model names to test
        config_dir: Directory containing config files
        output_path: Output JSON path
    """
    logger.info("=" * 80)
    logger.info("CROSS-ARCHITECTURE ABLATION STUDY")
    logger.info("Comparing Modularity vs TSP vs Random")
    logger.info("=" * 80)

    all_results = []

    for model_name in models:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Testing {model_name.upper()}")
        logger.info(f"{'=' * 80}")

        # Find config file
        config_path = None
        for possible_name in [f"{model_name}.json", f"{model_name}_large.json", f"{model_name}_3b.json"]:
            candidate = config_dir / possible_name
            if candidate.exists():
                config_path = candidate
                break

        if config_path is None:
            logger.warning(f"Config not found for {model_name}, skipping")
            continue

        # Load model and config
        model_baseline, config = load_model_and_config(model_name, config_path)
        pipeline_config = config.get("pipeline", {})
        solver_config = config.get("solver", {})

        # Build graph
        logger.info("Building co-access graph...")
        W = build_graph_from_model(model_baseline, config)
        n = W.shape[0]

        # Prepare inputs
        inputs = prepare_inputs(model_name)

        # Test 1: Baseline (no permutation)
        logger.info("[1/4] Measuring baseline (identity permutation)...")
        stats_baseline = measure_latency_with_stats(
            model=model_baseline,
            inputs=inputs,
            num_repeats=pipeline_config.get("repeats", 1000),
            warmup=pipeline_config.get("warmup_iter", 50),
            device="cuda",
        )
        baseline_latency = stats_baseline['mean']
        logger.info(f"Baseline: {baseline_latency:.3f} ms")

        # Test 2: Random permutation
        logger.info("[2/4] Testing random permutation...")
        model_random, _ = load_model_and_config(model_name, config_path)
        pi_random = np.random.permutation(n)
        apply_permutation(model_random, pi_random, model_name)

        stats_random = measure_latency_with_stats(
            model=model_random,
            inputs=inputs,
            num_repeats=pipeline_config.get("repeats", 1000),
            warmup=pipeline_config.get("warmup_iter", 50),
            device="cuda",
        )
        random_latency = stats_random['mean']
        logger.info(f"Random: {random_latency:.3f} ms")

        # Test 3: TSP-based permutation
        logger.info("[3/4] Testing TSP-based permutation...")
        model_tsp, _ = load_model_and_config(model_name, config_path)

        # Use TSP solver (greedy nearest neighbor)
        from icd.core.solver import fit_permutation
        cost_config = CostConfig(
            alpha=solver_config.get("alpha", 1.0),
            beta=solver_config.get("beta", 0.2),
        )

        # For TSP, we can use a simpler method
        # Import the TSP solver from run_tsp_baseline.py
        from scripts.run_tsp_baseline import solve_tsp_2opt
        if hasattr(W, 'toarray'):
            W_dense = W.toarray()
        else:
            W_dense = np.array(W)

        pi_tsp = solve_tsp_2opt(W_dense, max_iterations=1000)
        apply_permutation(model_tsp, pi_tsp, model_name)

        stats_tsp = measure_latency_with_stats(
            model=model_tsp,
            inputs=inputs,
            num_repeats=pipeline_config.get("repeats", 1000),
            warmup=pipeline_config.get("warmup_iter", 50),
            device="cuda",
        )
        tsp_latency = stats_tsp['mean']
        logger.info(f"TSP: {tsp_latency:.3f} ms")

        # Test 4: Modularity-based permutation (our method)
        logger.info("[4/4] Testing modularity-based permutation...")
        model_mod, _ = load_model_and_config(model_name, config_path)

        pi_mod, solver_stats = fit_permutation(
            W=W,
            cfg=cost_config,
            time_budget_s=solver_config.get("time_budget_s", 60.0),
            refine_steps=solver_config.get("refine_steps", 500),
            seed=solver_config.get("rng_seed", 0),
            method=solver_config.get("method", "louvain"),
        )
        apply_permutation(model_mod, pi_mod, model_name)

        stats_mod = measure_latency_with_stats(
            model=model_mod,
            inputs=inputs,
            num_repeats=pipeline_config.get("repeats", 1000),
            warmup=pipeline_config.get("warmup_iter", 50),
            device="cuda",
        )
        mod_latency = stats_mod['mean']
        logger.info(f"Modularity: {mod_latency:.3f} ms")

        # Compute improvements
        random_improvement = (baseline_latency - random_latency) / baseline_latency * 100
        tsp_improvement = (baseline_latency - tsp_latency) / baseline_latency * 100
        mod_improvement = (baseline_latency - mod_latency) / baseline_latency * 100

        mod_vs_tsp = (tsp_latency - mod_latency) / tsp_latency * 100

        logger.info(f"\nResults for {model_name}:")
        logger.info(f"  Random: {random_improvement:+.1f}% vs baseline")
        logger.info(f"  TSP:    {tsp_improvement:+.1f}% vs baseline")
        logger.info(f"  Modularity: {mod_improvement:+.1f}% vs baseline")
        logger.info(f"  Modularity vs TSP: {mod_vs_tsp:+.1f}%")

        all_results.append({
            "model": model_name,
            "baseline_latency_ms": baseline_latency,
            "random_latency_ms": random_latency,
            "tsp_latency_ms": tsp_latency,
            "modularity_latency_ms": mod_latency,
            "random_improvement_pct": random_improvement,
            "tsp_improvement_pct": tsp_improvement,
            "modularity_improvement_pct": mod_improvement,
            "modularity_vs_tsp_pct": mod_vs_tsp,
            "modularity_score": solver_stats.get("Q_louvain", None),
        })

        # Clean up
        del model_baseline, model_random, model_tsp, model_mod
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)

    avg_mod_vs_tsp = np.mean([r['modularity_vs_tsp_pct'] for r in all_results])
    logger.info(f"Average modularity advantage over TSP: {avg_mod_vs_tsp:.1f}%")

    # Save results
    output = {
        "experiment": "cross_arch_ablation",
        "models": models,
        "results": all_results,
        "summary": {
            "avg_modularity_vs_tsp_pct": avg_mod_vs_tsp,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Cross-Architecture Ablation Study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Example:
  python scripts/run_cross_arch_ablation.py \\
      --models mamba bert resnet gcn \\
      --config-dir configs \\
      --output results/ablations/cross_arch_ablation.json
        """,
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=["mamba", "bert", "resnet", "gcn"],
        help="List of models to test",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs"),
        help="Directory containing config files",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output JSON path",
    )

    args = parser.parse_args(argv)

    run_cross_arch_ablation(
        models=args.models,
        config_dir=args.config_dir,
        output_path=args.output,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
