#!/usr/bin/env python3
"""Synthetic Validation with Controlled Modularity (Paper Figure synthetic_validation)

Generates synthetic access patterns with known ground-truth community structures
and measures cache hit rates for layouts with varying modularity scores.

This validates that modularity is a robust, physically-grounded proxy for cache efficiency.

Usage:
    python scripts/run_synthetic_validation.py \\
        --num-nodes 1024 \\
        --num-communities 16 \\
        --output results/validation/synthetic_validation.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def generate_stochastic_block_model(
    n: int,
    k: int,
    p_in: float = 0.8,
    p_out: float = 0.1,
    seed: int = 0,
) -> Tuple[np.ndarray, List[List[int]]]:
    """Generate synthetic graph with known community structure.

    Args:
        n: Number of nodes
        k: Number of communities
        p_in: Probability of edge within community
        p_out: Probability of edge between communities
        seed: Random seed

    Returns:
        Tuple of (adjacency matrix, ground truth communities)
    """
    np.random.seed(seed)

    # Assign nodes to communities
    nodes_per_community = n // k
    communities = []
    for i in range(k):
        start = i * nodes_per_community
        end = start + nodes_per_community if i < k - 1 else n
        communities.append(list(range(start, end)))

    # Generate adjacency matrix
    A = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            # Check if i and j are in same community
            same_community = any(i in comm and j in comm for comm in communities)

            # Set edge probability
            p = p_in if same_community else p_out

            # Add edge with probability p
            if np.random.random() < p:
                A[i, j] = 1.0
                A[j, i] = 1.0

    return A, communities


def compute_modularity(A: np.ndarray, partition: List[List[int]]) -> float:
    """Compute modularity Q for a given partition.

    Q = (1/2m) * sum_ij [ A_ij - (k_i * k_j / 2m) ] * delta(c_i, c_j)

    Args:
        A: Adjacency matrix
        partition: List of communities (lists of node indices)

    Returns:
        Modularity score
    """
    n = A.shape[0]
    m = np.sum(A) / 2  # Number of edges

    if m == 0:
        return 0.0

    # Compute degrees
    k = np.sum(A, axis=1)

    Q = 0.0
    for community in partition:
        for i in community:
            for j in community:
                Q += A[i, j] - (k[i] * k[j] / (2 * m))

    Q /= (2 * m)
    return Q


def simulate_cache_hits(
    A: np.ndarray,
    permutation: np.ndarray,
    cache_line_size: int = 64,
    num_accesses: int = 10000,
    seed: int = 0,
) -> float:
    """Simulate cache behavior for a given memory layout.

    Args:
        A: Adjacency matrix (access patterns)
        permutation: Memory layout permutation
        cache_line_size: Number of elements per cache line
        num_accesses: Number of memory accesses to simulate
        seed: Random seed

    Returns:
        Cache hit rate (0-1)
    """
    np.random.seed(seed)
    n = A.shape[0]

    # Map original indices to permuted positions
    position = np.zeros(n, dtype=int)
    for new_pos, old_idx in enumerate(permutation):
        position[old_idx] = new_pos

    # Determine which cache line each element is in
    def cache_line(idx):
        return position[idx] // cache_line_size

    # Simulate random walk on graph (access pattern)
    current = np.random.randint(n)
    cache_state = {cache_line(current)}  # Set of loaded cache lines
    hits = 0
    misses = 0

    for _ in range(num_accesses):
        # Find neighbors (potential next accesses)
        neighbors = np.where(A[current] > 0)[0]

        if len(neighbors) == 0:
            # Random jump if no neighbors
            current = np.random.randint(n)
        else:
            # Access neighbor with probability proportional to edge weight
            weights = A[current, neighbors]
            weights = weights / np.sum(weights)
            current = np.random.choice(neighbors, p=weights)

        # Check cache
        current_line = cache_line(current)
        if current_line in cache_state:
            hits += 1
        else:
            misses += 1
            cache_state.add(current_line)

            # Simple LRU: limit cache size to 16 lines
            if len(cache_state) > 16:
                cache_state.pop()

    hit_rate = hits / (hits + misses) if (hits + misses) > 0 else 0.0
    return hit_rate


def partition_from_permutation(
    permutation: np.ndarray,
    block_size: int,
) -> List[List[int]]:
    """Convert permutation into block partition for modularity calculation.

    Args:
        permutation: Permutation array
        block_size: Size of each block

    Returns:
        List of communities (blocks)
    """
    n = len(permutation)
    num_blocks = (n + block_size - 1) // block_size

    partition = []
    for i in range(num_blocks):
        start = i * block_size
        end = min(start + block_size, n)
        block = [permutation[j] for j in range(start, end)]
        partition.append(block)

    return partition


def run_synthetic_validation(
    num_nodes: int,
    num_communities: int,
    output_path: Path,
) -> None:
    """Run synthetic validation experiment.

    Args:
        num_nodes: Number of nodes in synthetic graph
        num_communities: Number of ground-truth communities
        output_path: Output JSON path
    """
    logger.info("=" * 80)
    logger.info("SYNTHETIC VALIDATION WITH CONTROLLED MODULARITY")
    logger.info("=" * 80)

    logger.info(f"Generating synthetic graph: {num_nodes} nodes, {num_communities} communities")

    # Generate synthetic graph with known community structure
    A, ground_truth_communities = generate_stochastic_block_model(
        n=num_nodes,
        k=num_communities,
        p_in=0.8,
        p_out=0.1,
        seed=0,
    )

    logger.info(f"Graph density: {np.sum(A) / (num_nodes * num_nodes):.3f}")

    # Test multiple permutations with varying modularity
    results = []
    cache_line_size = 64

    # 1. Ground truth optimal permutation
    logger.info("\n[1/6] Testing ground truth optimal layout...")
    pi_optimal = np.concatenate([np.array(comm) for comm in ground_truth_communities])
    partition_optimal = partition_from_permutation(pi_optimal, cache_line_size)
    Q_optimal = compute_modularity(A, partition_optimal)
    hit_rate_optimal = simulate_cache_hits(A, pi_optimal, cache_line_size)

    logger.info(f"Ground truth: Q={Q_optimal:.3f}, hit_rate={hit_rate_optimal:.3f}")
    results.append({"method": "ground_truth", "modularity": Q_optimal, "hit_rate": hit_rate_optimal})

    # 2. Random permutation
    logger.info("[2/6] Testing random layout...")
    pi_random = np.random.permutation(num_nodes)
    partition_random = partition_from_permutation(pi_random, cache_line_size)
    Q_random = compute_modularity(A, partition_random)
    hit_rate_random = simulate_cache_hits(A, pi_random, cache_line_size)

    logger.info(f"Random: Q={Q_random:.3f}, hit_rate={hit_rate_random:.3f}")
    results.append({"method": "random", "modularity": Q_random, "hit_rate": hit_rate_random})

    # 3-6. Partially shuffled permutations (varying modularity)
    for i, shuffle_fraction in enumerate([0.25, 0.5, 0.75, 0.9]):
        logger.info(f"[{i+3}/6] Testing {shuffle_fraction*100:.0f}% shuffled layout...")

        # Start with optimal, shuffle a fraction
        pi_partial = pi_optimal.copy()
        num_shuffle = int(num_nodes * shuffle_fraction)
        shuffle_indices = np.random.choice(num_nodes, num_shuffle, replace=False)
        pi_partial[shuffle_indices] = np.random.permutation(pi_partial[shuffle_indices])

        partition_partial = partition_from_permutation(pi_partial, cache_line_size)
        Q_partial = compute_modularity(A, partition_partial)
        hit_rate_partial = simulate_cache_hits(A, pi_partial, cache_line_size)

        logger.info(f"{shuffle_fraction*100:.0f}% shuffled: Q={Q_partial:.3f}, hit_rate={hit_rate_partial:.3f}")
        results.append({
            "method": f"shuffled_{shuffle_fraction}",
            "modularity": Q_partial,
            "hit_rate": hit_rate_partial,
        })

    # -------------------------------------------------------------------------
    # Analysis
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS")
    logger.info("=" * 80)

    modularities = [r['modularity'] for r in results]
    hit_rates = [r['hit_rate'] for r in results]

    # Compute correlation
    correlation, p_value = stats.pearsonr(modularities, hit_rates)

    logger.info(f"Correlation (Modularity vs Cache Hit Rate): r={correlation:.3f}, p={p_value:.6f}")

    if correlation > 0.8 and p_value < 0.01:
        logger.info("✅ Strong positive correlation CONFIRMED")
        logger.info("   Modularity is a valid proxy for cache efficiency")
    else:
        logger.info("⚠️ Weak correlation observed")

    # Generate plot
    plot_path = output_path.with_suffix(".png")
    generate_plot(results, correlation, plot_path)
    logger.info(f"\nPlot saved to: {plot_path}")

    # Save results
    output = {
        "experiment": "synthetic_validation",
        "num_nodes": num_nodes,
        "num_communities": num_communities,
        "results": results,
        "correlation": correlation,
        "p_value": p_value,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Results saved to: {output_path}")


def generate_plot(results: List[Dict], correlation: float, output_path: Path) -> None:
    """Generate synthetic validation plot."""
    modularities = [r['modularity'] for r in results]
    hit_rates = [r['hit_rate'] for r in results]
    methods = [r['method'] for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter plot
    ax.scatter(modularities, hit_rates, s=150, alpha=0.7, edgecolors='black', linewidths=2)

    # Add labels for each point
    for i, method in enumerate(methods):
        ax.annotate(
            method.replace('_', ' ').title(),
            (modularities[i], hit_rates[i]),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontsize=9,
        )

    # Fit line
    z = np.polyfit(modularities, hit_rates, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(modularities), max(modularities), 100)
    ax.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.7, label=f'Linear fit (r={correlation:.3f})')

    ax.set_xlabel('Modularity (Q)', fontsize=12)
    ax.set_ylabel('Cache Hit Rate', fontsize=12)
    ax.set_title('Validation on Synthetic Data:\nModularity vs Cache Efficiency', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Synthetic Validation with Controlled Modularity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Example:
  python scripts/run_synthetic_validation.py \\
      --num-nodes 1024 \\
      --num-communities 16 \\
      --output results/validation/synthetic_validation.json
        """,
    )

    parser.add_argument(
        "--num-nodes",
        type=int,
        default=1024,
        help="Number of nodes in synthetic graph",
    )
    parser.add_argument(
        "--num-communities",
        type=int,
        default=16,
        help="Number of ground-truth communities",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output JSON path",
    )

    args = parser.parse_args(argv)

    run_synthetic_validation(
        num_nodes=args.num_nodes,
        num_communities=args.num_communities,
        output_path=args.output,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
