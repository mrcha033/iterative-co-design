#!/usr/bin/env python3
"""TSP Baseline for Ablation Study (Paper Table: Ablations)

Implements the TSP (Traveling Salesperson Problem) approach mentioned in the
paper's ablation study. This baseline treats data layout as a pairwise
optimization problem where we minimize the total "distance" between
consecutively accessed parameters.

The TSP formulation:
- Nodes: Model parameters/layers
- Edge weights: Co-access frequency (from graph W)
- Objective: Find permutation that minimizes sum of consecutive access costs

This is a simpler approach than our modularity-based method but tests whether
a greedy path-based optimization is sufficient.

Usage:
    python scripts/run_tsp_baseline.py \\
        --model mamba \\
        --config configs/mamba.json \\
        --output results/ablations/mamba_tsp_baseline.json
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

from icd.core.graph_instrumented import build_instrumented_graph
from icd.core.graph_pytorch import build_csr_from_fx_trace
from icd.experiments.hf import load_mamba_model, load_hf_sequence_classifier
from icd.measure.cuda_latency import measure_latency_with_stats
from icd.runtime.apply_pi import apply_pi_to_mamba_hf, apply_pi_to_bert

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def solve_tsp_greedy(W: np.ndarray) -> np.ndarray:
    """Solve TSP using greedy nearest-neighbor heuristic.

    This is a classic TSP approximation: start at node 0, always move to the
    nearest unvisited neighbor.

    Args:
        W: Weighted adjacency matrix (n × n), W[i,j] = co-access frequency

    Returns:
        Permutation array of shape (n,) representing the TSP tour
    """
    n = W.shape[0]
    visited = np.zeros(n, dtype=bool)
    tour = []

    # Start at node 0 (arbitrary choice)
    current = 0
    tour.append(current)
    visited[current] = True

    for _ in range(n - 1):
        # Find nearest unvisited neighbor
        neighbors = W[current].copy()
        neighbors[visited] = -np.inf  # Exclude visited nodes

        next_node = int(np.argmax(neighbors))
        tour.append(next_node)
        visited[next_node] = True
        current = next_node

    return np.array(tour, dtype=np.int64)


def solve_tsp_2opt(W: np.ndarray, max_iterations: int = 1000) -> np.ndarray:
    """Solve TSP using 2-opt local search.

    Starts with greedy solution, then iteratively improves by swapping edges.

    Args:
        W: Weighted adjacency matrix
        max_iterations: Maximum number of 2-opt swaps

    Returns:
        Improved permutation
    """
    # Start with greedy solution
    tour = solve_tsp_greedy(W)
    n = len(tour)

    def tour_cost(t: np.ndarray) -> float:
        """Compute total tour cost (sum of consecutive edge weights)."""
        cost = 0.0
        for i in range(n - 1):
            cost += W[t[i], t[i + 1]]
        return -cost  # Negative because higher co-access = better

    best_cost = tour_cost(tour)
    improved = True
    iterations = 0

    while improved and iterations < max_iterations:
        improved = False

        # Try all 2-opt swaps
        for i in range(1, n - 2):
            for j in range(i + 1, n):
                # Reverse segment [i:j]
                new_tour = tour.copy()
                new_tour[i:j] = new_tour[i:j][::-1]

                new_cost = tour_cost(new_tour)
                if new_cost < best_cost:
                    tour = new_tour
                    best_cost = new_cost
                    improved = True
                    break

            if improved:
                break

        iterations += 1

    logger.info(f"2-opt converged after {iterations} iterations")
    return tour


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


def build_graph_from_model(model: Any, config: Dict) -> np.ndarray:
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


def run_tsp_baseline(
    model_name: str,
    config_path: Path,
    output_path: Path,
    method: str = "2opt",
) -> None:
    """Run TSP baseline experiment.

    Args:
        model_name: Model architecture name
        config_path: Path to config JSON
        output_path: Output JSON path
        method: TSP solving method ("greedy" or "2opt")
    """
    logger.info("=" * 80)
    logger.info("TSP BASELINE EXPERIMENT")
    logger.info("=" * 80)

    # Load model and config
    logger.info(f"Loading {model_name}...")
    model, config = load_model_and_config(model_name, config_path)

    # Build co-access graph
    logger.info("Building co-access graph...")
    W = build_graph_from_model(model, config)

    # Convert to dense if sparse
    if hasattr(W, "toarray"):
        W_dense = W.toarray()
    else:
        W_dense = np.array(W)

    logger.info(f"Graph shape: {W_dense.shape}")

    # Solve TSP
    logger.info(f"Solving TSP using {method}...")
    if method == "greedy":
        pi = solve_tsp_greedy(W_dense)
    elif method == "2opt":
        pi = solve_tsp_2opt(W_dense, max_iterations=1000)
    else:
        raise ValueError(f"Unknown TSP method: {method}")

    # Compute tour quality metrics
    n = len(pi)
    tour_cost = 0.0
    for i in range(n - 1):
        tour_cost += W_dense[pi[i], pi[i + 1]]

    logger.info(f"Tour cost (total co-access): {tour_cost:.2f}")

    # Apply permutation to model
    logger.info("Applying permutation to model...")
    if model_name == "mamba":
        apply_pi_to_mamba_hf(model, pi)
    elif model_name == "bert":
        apply_pi_to_bert(model, pi)

    # Measure latency
    logger.info("Measuring latency...")
    inputs = prepare_inputs(model_name)
    pipeline_config = config.get("pipeline", {})
    latency_stats = measure_latency_with_stats(
        model=model,
        inputs=inputs,
        num_repeats=pipeline_config.get("repeats", 1000),
        warmup=pipeline_config.get("warmup_iter", 50),
        device="cuda",
    )

    # Save results
    result = {
        "experiment": "tsp_baseline",
        "model": model_name,
        "method": method,
        "latency_stats": latency_stats,
        "tour_cost": float(tour_cost),
        "permutation": pi.tolist(),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"\nResults:")
    logger.info(f"  Latency: {latency_stats['mean']:.3f} ± {latency_stats['std']:.3f} ms")
    logger.info(f"  Tour cost: {tour_cost:.2f}")
    logger.info(f"\nResults saved to: {output_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="TSP Baseline for Ablation Study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Example:
  python scripts/run_tsp_baseline.py \\
      --model mamba \\
      --config configs/mamba.json \\
      --output results/ablations/mamba_tsp_baseline.json
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
        "--method",
        default="2opt",
        choices=["greedy", "2opt"],
        help="TSP solving method (default: 2opt)",
    )

    args = parser.parse_args(argv)

    run_tsp_baseline(
        model_name=args.model,
        config_path=args.config,
        output_path=args.output,
        method=args.method,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
