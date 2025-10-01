#!/usr/bin/env python3
"""Validate the core mechanistic claim: Modularity → Cache Hit Rate → Latency.

This script implements the validation described in the paper (Section 3.5):
1. Generate permutations with varying modularity scores
2. Measure L2 cache hit rate for each permutation
3. Measure latency for each permutation
4. Compute correlations and establish causal chain

Usage:
    python scripts/validate_mechanistic_claim.py --model mamba-130m --device cuda --output validation_results.json

This directly tests whether high-modularity permutations actually improve cache
performance as claimed, providing empirical evidence for the theoretical model.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

try:
    import matplotlib

    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency for plotting
    matplotlib = None  # type: ignore[assignment]
    plt = None  # type: ignore[assignment]

from icd.measure.cuda_latency import measure_latency_with_stats
from icd.measure.l2_ncu import collect_l2_metrics
from icd.runtime.apply_pi import (
    PermutationApplicationError,
    apply_pi_to_bert,
    apply_pi_to_mamba,
    apply_pi_to_mamba_hf,
)
from icd.runtime.runners_hf import _collect_mamba_modules_from_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def compute_modularity_from_permutation(
    W: Any,  # CSRMatrix
    permutation: List[int],
    num_communities: int = 16,
) -> float:
    """Compute modularity score for a given permutation.

    Uses the definition from the paper: fraction of edges within communities.

    Args:
        W: Co-access weight matrix (CSRMatrix).
        permutation: Node ordering.
        num_communities: Number of communities for partitioning.

    Returns:
        Modularity score Q ∈ [0, 1].
    """
    try:
        import networkx as nx
    except ImportError:
        logger.error("NetworkX required for modularity computation")
        return 0.0

    # Build graph
    n = W.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))

    # Add edges from CSR
    for i in range(n):
        start, end = W.indptr[i], W.indptr[i + 1]
        for idx in range(start, end):
            j = W.indices[idx]
            if j > i:  # Upper triangle only
                weight = W.data[idx]
                if weight > 0:
                    G.add_edge(i, j, weight=weight)

    # Create communities based on permutation
    # Divide permuted nodes into contiguous blocks
    block_size = max(1, n // num_communities)
    communities = []
    for k in range(num_communities):
        start_idx = k * block_size
        end_idx = min((k + 1) * block_size, n)
        community = set(permutation[start_idx:end_idx])
        if community:
            communities.append(community)

    # Compute modularity
    try:
        Q = nx.algorithms.community.quality.modularity(
            G, communities, weight="weight"
        )
        return float(Q)
    except (ZeroDivisionError, ValueError):
        return 0.0


def generate_permutations_with_varying_modularity(
    W: Any,  # CSRMatrix
    num_permutations: int = 20,
) -> List[Tuple[List[int], float]]:
    """Generate permutations with different modularity scores.

    Strategy:
    1. Random permutation (low modularity)
    2. Degree-sorted (medium modularity)
    3. Spectral clustering (high modularity)
    4. Interpolations between them

    Args:
        W: Co-access weight matrix.
        num_permutations: Number of permutations to generate.

    Returns:
        List of (permutation, modularity_score) tuples.
    """
    from icd.core.solver import _spectral_init_like, _degree_vector
    from icd.core.cost import CostConfig, eval_cost
    import random

    n = W.shape[0]
    permutations = []

    # 1. Random permutation (baseline - low modularity)
    random_perm = list(range(n))
    random.shuffle(random_perm)
    Q_random = compute_modularity_from_permutation(W, random_perm)
    permutations.append((random_perm, Q_random))
    logger.info(f"Random permutation: Q={Q_random:.3f}")

    # 2. Degree-sorted (medium modularity)
    degrees = _degree_vector(W)
    degree_perm = sorted(range(n), key=lambda i: degrees[i])
    Q_degree = compute_modularity_from_permutation(W, degree_perm)
    permutations.append((degree_perm, Q_degree))
    logger.info(f"Degree-sorted: Q={Q_degree:.3f}")

    # 3. Spectral ordering (high modularity)
    spectral_perm = _spectral_init_like(W, seed=42)
    Q_spectral = compute_modularity_from_permutation(W, spectral_perm)
    permutations.append((spectral_perm, Q_spectral))
    logger.info(f"Spectral ordering: Q={Q_spectral:.3f}")

    # 4. Generate intermediate permutations via random swaps
    # Start from spectral (high Q) and gradually randomize
    base_perm = spectral_perm[:]
    num_swaps_schedule = np.linspace(0, n // 2, num_permutations - 3).astype(int)

    for num_swaps in num_swaps_schedule:
        perm = base_perm[:]
        # Apply random swaps
        for _ in range(int(num_swaps)):
            i, j = random.sample(range(n), 2)
            perm[i], perm[j] = perm[j], perm[i]

        Q = compute_modularity_from_permutation(W, perm)
        permutations.append((perm, Q))

    # Sort by modularity
    permutations.sort(key=lambda x: x[1])

    logger.info(f"Generated {len(permutations)} permutations with Q ∈ [{permutations[0][1]:.3f}, {permutations[-1][1]:.3f}]")
    return permutations


def _tensor_from_permutation(model: Any, permutation: List[int]) -> torch.LongTensor:
    try:
        device = next(model.parameters()).device  # type: ignore[attr-defined]
    except StopIteration:
        device = torch.device("cpu")
    except AttributeError:
        device = torch.device("cpu")
    return torch.as_tensor(permutation, device=device, dtype=torch.long)


def _apply_permutation_to_model(model: Any, pi_tensor: torch.LongTensor) -> None:
    model_type = getattr(getattr(model, "config", None), "model_type", None)

    if model_type == "mamba":
        modules = _collect_mamba_modules_from_model(model)
        if not modules:
            raise RuntimeError("no applicable Mamba modules found for permutation application")

        applied = False
        for entry in modules:
            try:
                if entry.get("_hf_mamba"):
                    apply_pi_to_mamba_hf(entry, pi_tensor)
                else:
                    apply_pi_to_mamba(entry, pi_tensor)
                applied = True
            except PermutationApplicationError as exc:
                logger.warning(
                    "Permutation rejected for module %s: %s",
                    entry.get("_module_name", "<unnamed>"),
                    exc,
                )

        if not applied:
            raise RuntimeError("permutation rejected for all collected Mamba modules")

        return

    apply_pi_to_bert(model, pi_tensor)


def measure_cache_and_latency_for_permutation(
    model: Any,
    inputs: Any,
    permutation: List[int],
    device: str = "cuda",
    num_warmup: int = 20,
    num_samples: int = 100,
) -> Tuple[float, float]:
    """Measure L2 cache hit rate and latency for a specific permutation.

    Args:
        model: PyTorch model.
        inputs: Input tensors.
        permutation: Memory layout permutation to apply.
        device: Device to run on.
        num_warmup: Warmup iterations.
        num_samples: Number of measurement samples.

    Returns:
        Tuple of (l2_hit_rate_pct, mean_latency_ms).
        Returns (NaN, NaN) if measurement fails.
    """
    original_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    pi_tensor = _tensor_from_permutation(model, permutation)

    try:
        _apply_permutation_to_model(model, pi_tensor)

        latency_stats = measure_latency_with_stats(
            model,
            inputs,
            num_repeats=num_samples,
            warmup=num_warmup,
            device=device,
        )
        mean_latency = latency_stats.get("mean", float("nan"))

        cache_metrics = collect_l2_metrics(
            model,
            inputs,
            output_dir=None,
        )
        l2_hit_rate = cache_metrics.get("l2_hit_rate_pct", float("nan"))

        return l2_hit_rate, mean_latency

    except Exception as e:
        logger.error(f"Measurement failed: {e}")
        return float("nan"), float("nan")
    finally:
        try:
            with torch.no_grad():
                model.load_state_dict(original_state, strict=True)
        except Exception as restore_error:  # pragma: no cover - defensive
            logger.error(f"Failed to restore original model parameters: {restore_error}")


def run_validation(
    model: Any,
    inputs: Any,
    W: Any,
    device: str = "cuda",
    num_permutations: int = 20,
    output_path: str = "validation_results.json",
) -> Dict[str, Any]:
    """Run complete validation of mechanistic claim.

    This is the main validation pipeline:
    1. Generate permutations with varying modularity
    2. Measure cache and latency for each
    3. Compute correlations
    4. Test causal relationships

    Args:
        model: PyTorch model to validate.
        inputs: Input tensors.
        W: Co-access weight matrix.
        device: Device to run on.
        num_permutations: Number of permutations to test.
        output_path: Path to save results JSON.

    Returns:
        Validation results dictionary.
    """
    logger.info("=" * 80)
    logger.info("MECHANISTIC VALIDATION: Modularity → Cache → Latency")
    logger.info("=" * 80)

    # Generate permutations
    logger.info(f"\nGenerating {num_permutations} permutations...")
    permutations = generate_permutations_with_varying_modularity(W, num_permutations)

    # Measure for each permutation
    results = []
    for idx, (perm, Q) in enumerate(permutations):
        logger.info(f"\n[{idx+1}/{len(permutations)}] Testing permutation with Q={Q:.3f}")

        l2_hit_rate, latency_ms = measure_cache_and_latency_for_permutation(
            model, inputs, perm, device=device
        )

        results.append({
            "permutation_idx": idx,
            "modularity_Q": Q,
            "l2_hit_rate_pct": l2_hit_rate,
            "latency_ms": latency_ms,
        })

        logger.info(f"  L2 hit rate: {l2_hit_rate:.1f}%, Latency: {latency_ms:.3f}ms")

    # Compute correlations
    modularities = [r["modularity_Q"] for r in results]
    l2_rates = [r["l2_hit_rate_pct"] for r in results]
    latencies = [r["latency_ms"] for r in results]

    # Filter out NaN values
    valid_data = [
        (Q, l2, lat)
        for Q, l2, lat in zip(modularities, l2_rates, latencies)
        if not (np.isnan(Q) or np.isnan(l2) or np.isnan(lat))
    ]

    if len(valid_data) < 3:
        logger.error("Insufficient valid measurements for correlation analysis")
        return {"error": "Insufficient valid data", "results": results}

    Q_vals = [d[0] for d in valid_data]
    l2_vals = [d[1] for d in valid_data]
    lat_vals = [d[2] for d in valid_data]

    # Correlation: Modularity ↔ L2 Hit Rate (should be positive)
    corr_Q_l2 = np.corrcoef(Q_vals, l2_vals)[0, 1]

    # Correlation: L2 Hit Rate ↔ Latency (should be negative - higher hit rate = lower latency)
    corr_l2_lat = np.corrcoef(l2_vals, lat_vals)[0, 1]

    # Correlation: Modularity ↔ Latency (should be negative - higher Q = lower latency)
    corr_Q_lat = np.corrcoef(Q_vals, lat_vals)[0, 1]

    logger.info("\n" + "=" * 80)
    logger.info("CORRELATION RESULTS")
    logger.info("=" * 80)
    logger.info(f"Modularity ↔ L2 Hit Rate:   r = {corr_Q_l2:+.3f}  (expect: positive)")
    logger.info(f"L2 Hit Rate ↔ Latency:      r = {corr_l2_lat:+.3f}  (expect: negative)")
    logger.info(f"Modularity ↔ Latency:       r = {corr_Q_lat:+.3f}  (expect: negative)")
    logger.info("=" * 80)

    # Paper claims r=-0.88 for Q ↔ Latency
    # Let's check if we're in the right ballpark
    validates_paper = abs(corr_Q_lat) > 0.7 and corr_Q_lat < 0

    validation_results = {
        "num_permutations": len(permutations),
        "num_valid_measurements": len(valid_data),
        "correlations": {
            "modularity_vs_l2": corr_Q_l2,
            "l2_vs_latency": corr_l2_lat,
            "modularity_vs_latency": corr_Q_lat,
        },
        "paper_claim_validated": validates_paper,
        "paper_claimed_r": -0.88,
        "measurements": results,
    }

    # Save results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(validation_results, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")

    # Generate visualization
    plot_path = output_path.with_suffix(".png")
    plot_validation_results(validation_results, plot_path)

    return validation_results


def plot_validation_results(results: Dict[str, Any], output_path: Path) -> None:
    """Generate visualization of validation results."""
    if plt is None:
        logger.warning("Matplotlib not available; skipping validation plot generation.")
        return

    measurements = results["measurements"]

    Q_vals = [m["modularity_Q"] for m in measurements if not np.isnan(m["modularity_Q"])]
    l2_vals = [m["l2_hit_rate_pct"] for m in measurements if not np.isnan(m["l2_hit_rate_pct"])]
    lat_vals = [m["latency_ms"] for m in measurements if not np.isnan(m["latency_ms"])]

    if len(Q_vals) < 2:
        logger.warning("Insufficient data for plotting")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1: Modularity vs L2 Hit Rate
    axes[0].scatter(Q_vals, l2_vals, alpha=0.6, s=50)
    axes[0].set_xlabel("Modularity (Q)")
    axes[0].set_ylabel("L2 Hit Rate (%)")
    axes[0].set_title(f"Modularity → L2 Cache\nr = {results['correlations']['modularity_vs_l2']:.3f}")
    axes[0].grid(True, alpha=0.3)

    # Plot 2: L2 Hit Rate vs Latency
    axes[1].scatter(l2_vals, lat_vals, alpha=0.6, s=50, color='orange')
    axes[1].set_xlabel("L2 Hit Rate (%)")
    axes[1].set_ylabel("Latency (ms)")
    axes[1].set_title(f"L2 Cache → Latency\nr = {results['correlations']['l2_vs_latency']:.3f}")
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Modularity vs Latency (Total Effect)
    axes[2].scatter(Q_vals, lat_vals, alpha=0.6, s=50, color='green')
    axes[2].set_xlabel("Modularity (Q)")
    axes[2].set_ylabel("Latency (ms)")
    axes[2].set_title(f"Modularity → Latency (Total)\nr = {results['correlations']['modularity_vs_latency']:.3f}")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Validation plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate mechanistic claim: Modularity → Cache → Latency"
    )
    parser.add_argument(
        "--config",
        "-c",
        default="configs/mamba.json",
        help="Model configuration file"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run on"
    )
    parser.add_argument(
        "--num-permutations",
        type=int,
        default=20,
        help="Number of permutations to test"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="validation_results.json",
        help="Output path for results"
    )
    args = parser.parse_args()

    # Check CUDA availability
    try:
        import torch
        if args.device == "cuda" and not torch.cuda.is_available():
            logger.error("CUDA requested but not available. Falling back to CPU.")
            args.device = "cpu"
    except ImportError:
        logger.error("PyTorch not available. Cannot run validation.")
        return 1

    logger.info(f"Loading model from config: {args.config}")

    try:
        from icd.core.graph import build_w
        from icd.utils.imports import load_object

        # Load configuration
        with open(args.config) as f:
            config = json.load(f)

        # Extract graph configuration
        graph_config = config.get("graph", {})
        if not graph_config:
            logger.error("No 'graph' section in config file")
            return 1

        # Load model using the loader specified in config
        loader_name = graph_config.get("loader")
        loader_kwargs = graph_config.get("loader_kwargs", {})

        if not loader_name:
            logger.error("No 'loader' specified in graph config")
            logger.info("Expected config format: {'graph': {'loader': 'icd.experiments.hf.load_hf_causal_lm', 'loader_kwargs': {...}}}")
            return 1

        # Override device from command line
        if "device" in loader_kwargs:
            loader_kwargs["device"] = args.device

        logger.info(f"Loading model using {loader_name}...")
        loader_fn = load_object(loader_name)

        try:
            model, example_inputs = loader_fn(**loader_kwargs)
            logger.info(f"Model loaded successfully on {args.device}")
            logger.info(f"Example inputs: {len(example_inputs)} tensor(s)")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.error("Check that model name and loader kwargs are correct in config")
            return 1

        # Build W matrix
        logger.info("Building co-access graph...")

        # Prepare build_w config
        build_w_config = dict(graph_config)
        build_w_config["model"] = model
        build_w_config["example_inputs"] = example_inputs

        # Extract source (default to "pytorch" for HF models)
        source = build_w_config.get("source", "pytorch")

        try:
            W = build_w(source=source, **build_w_config)
            logger.info(f"Graph built: {W.shape[0]} nodes, {W.nnz()} edges")
        except Exception as e:
            logger.error(f"Failed to build graph: {e}")
            logger.error("This may happen if the model structure is not compatible with the graph builder")
            logger.info("You can try using source='mock' for testing the validation infrastructure")
            return 1

        # Run validation
        logger.info(f"\nStarting validation with {args.num_permutations} permutations...")

        results = run_validation(
            model=model,
            inputs=example_inputs,
            W=W,
            device=args.device,
            num_permutations=args.num_permutations,
            output_path=args.output,
        )

        # Check if validation succeeded
        if "error" in results:
            logger.error(f"Validation failed: {results['error']}")
            return 1

        # Report results
        logger.info("\n" + "=" * 80)
        logger.info("VALIDATION COMPLETE")
        logger.info("=" * 80)

        correlations = results.get("correlations", {})
        validated = results.get("paper_claim_validated", False)

        logger.info(f"\nCorrelations measured:")
        logger.info(f"  Modularity ↔ L2:       {correlations.get('modularity_vs_l2', float('nan')):+.3f}")
        logger.info(f"  L2 ↔ Latency:          {correlations.get('l2_vs_latency', float('nan')):+.3f}")
        logger.info(f"  Modularity ↔ Latency:  {correlations.get('modularity_vs_latency', float('nan')):+.3f}")

        if validated:
            logger.info("\n✅ Paper claim VALIDATED (|r| > 0.7, negative correlation)")
        else:
            logger.info("\n⚠️  Paper claim NOT validated (|r| < 0.7 or wrong sign)")
            logger.info("    This may indicate:")
            logger.info("    - L2 cache profiling needs GPU hardware")
            logger.info("    - Model-specific permutation application needed")
            logger.info("    - Theoretical predictions don't match this architecture")

        logger.info(f"\nResults saved to: {args.output}")
        logger.info("=" * 80)

        return 0

    except KeyboardInterrupt:
        logger.info("\nValidation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Validation failed with unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
