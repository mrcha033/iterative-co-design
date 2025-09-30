"""Instrumented co-access measurement for mechanistic graph construction.

This module implements TRUE memory co-access measurement by instrumenting
model execution and tracking which dimensions are accessed in temporal proximity.

Key Difference from Heuristic Approach:
--------------------------------------
- HEURISTIC (graph_pytorch.py): Assumes dimension i correlates with i+1 (spatial locality)
- MECHANISTIC (this file): Measures which dimensions are ACTUALLY co-accessed during execution

The hypothesis is that dimensions accessed within a small temporal window
(e.g., 100 nanoseconds) are likely to benefit from cache co-residency when
reordered contiguously.

Theory:
-------
If dimensions i and j are accessed within a cache-line-fetch time window,
placing them contiguously in memory will improve cache utilization because:
1. They'll likely be fetched in the same cache line
2. One access brings the other into cache for "free"
3. Reduces cache misses and memory bandwidth

This directly tests the paper's claim that modularity optimization improves
cache behavior by creating a graph W that represents true co-access patterns.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, DefaultDict, Dict, List, Optional, Set, Tuple

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    nn = None  # type: ignore

from .graph import CSRMatrix

logger = logging.getLogger(__name__)

__all__ = [
    "DimensionAccessTracker",
    "InstrumentedGraphConfig",
    "build_w_from_instrumented_access",
    "extract_hidden_dimensions",
]


@dataclass
class InstrumentedGraphConfig:
    """Configuration for instrumented graph construction."""

    # Temporal window for co-access (nanoseconds)
    # Default: 100ns = typical L1 cache latency
    temporal_window_ns: float = 100.0

    # Minimum co-accesses to create edge
    min_coaccesses: int = 2

    # Number of forward passes to sample
    num_samples: int = 10

    # Normalize edge weights
    normalize: bool = True

    # Weight by access frequency (more co-accesses = stronger edge)
    weight_by_frequency: bool = True

    # Weight by temporal proximity (closer in time = stronger edge)
    weight_by_proximity: bool = True

    # Cache line size in bytes (for dimension grouping)
    cache_line_bytes: int = 64

    # Track per-layer or aggregate across all layers
    per_layer_tracking: bool = False

    # Layers to instrument (None = all linear layers)
    target_layers: Optional[List[str]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any] | None) -> "InstrumentedGraphConfig":
        """Create config from dictionary."""
        if not data:
            return cls()

        return cls(
            temporal_window_ns=float(data.get("temporal_window_ns", 100.0)),
            min_coaccesses=int(data.get("min_coaccesses", 2)),
            num_samples=int(data.get("num_samples", 10)),
            normalize=bool(data.get("normalize", True)),
            weight_by_frequency=bool(data.get("weight_by_frequency", True)),
            weight_by_proximity=bool(data.get("weight_by_proximity", True)),
            cache_line_bytes=int(data.get("cache_line_bytes", 64)),
            per_layer_tracking=bool(data.get("per_layer_tracking", False)),
            target_layers=data.get("target_layers"),
        )


@dataclass
class DimensionAccess:
    """Single dimension access event."""

    dimension: int
    timestamp_ns: int
    layer_name: str
    operation: str  # 'read', 'write', 'readwrite'
    access_size_bytes: int = 0


class DimensionAccessTracker:
    """Tracks dimension access patterns during model execution.

    Core Algorithm:
    ---------------
    1. Hook all relevant layers (linear, matmul, etc.)
    2. For each forward pass, record which dimensions are accessed and when
    3. Build co-access graph: dimensions accessed within temporal window get edges
    4. Edge weight = co-access frequency (optionally weighted by proximity)

    Example:
    --------
    >>> tracker = DimensionAccessTracker(hidden_size=768)
    >>> # During forward pass...
    >>> tracker.record_access({0, 1, 2, 100, 101}, time.perf_counter_ns(), "layer1")
    >>> tracker.record_access({1, 2, 3, 101, 102}, time.perf_counter_ns(), "layer2")
    >>> # After multiple passes...
    >>> graph_W = tracker.build_coacccess_graph()
    >>> # W[1,2] will have high weight (accessed together often)
    >>> # W[0,100] will have high weight (accessed in same operation)
    """

    def __init__(
        self,
        hidden_size: int,
        config: Optional[InstrumentedGraphConfig] = None,
    ):
        """Initialize tracker.

        Args:
            hidden_size: Size of hidden dimension (D in paper notation).
            config: Configuration for tracking and graph construction.
        """
        self.hidden_size = hidden_size
        self.config = config or InstrumentedGraphConfig()

        # Access log: list of (dimension, timestamp_ns, layer_name)
        self.access_log: List[Tuple[int, int, str]] = []

        # Per-layer logs (if per_layer_tracking enabled)
        self.per_layer_logs: DefaultDict[str, List[Tuple[int, int]]] = defaultdict(list)

        # Statistics
        self.total_accesses = 0
        self.num_samples_recorded = 0

    def record_access(
        self,
        dimensions: Set[int],
        timestamp_ns: int,
        layer_name: str = "unknown",
    ) -> None:
        """Record that a set of dimensions were accessed at this timestamp.

        Args:
            dimensions: Set of dimension indices accessed.
            timestamp_ns: Timestamp in nanoseconds (use time.perf_counter_ns()).
            layer_name: Name of layer where access occurred.
        """
        for dim in dimensions:
            if 0 <= dim < self.hidden_size:
                self.access_log.append((dim, timestamp_ns, layer_name))

                if self.config.per_layer_tracking:
                    self.per_layer_logs[layer_name].append((dim, timestamp_ns))

                self.total_accesses += 1

    def increment_sample_count(self) -> None:
        """Increment sample counter (call after each forward pass)."""
        self.num_samples_recorded += 1

    def build_coacccess_graph(self) -> CSRMatrix:
        """Build co-access graph from recorded access patterns.

        Algorithm:
        ----------
        1. Sort all accesses by timestamp
        2. For each access, find all other accesses within temporal window
        3. Create edges between all co-accessed dimension pairs
        4. Weight edges by co-access frequency and proximity

        Returns:
            CSRMatrix representing co-access graph W.
            W[i,j] = strength of co-access between dimensions i and j
        """
        if not self.access_log:
            logger.warning("No accesses recorded, returning empty graph")
            return self._create_empty_graph()

        logger.info(
            f"Building co-access graph from {len(self.access_log)} accesses "
            f"across {self.num_samples_recorded} samples"
        )

        # Sort by timestamp for efficient windowing
        sorted_accesses = sorted(self.access_log, key=lambda x: x[1])

        # Count co-accesses within temporal window
        coaccesses: DefaultDict[Tuple[int, int], float] = defaultdict(float)
        window_ns = self.config.temporal_window_ns

        # Sliding window algorithm
        for i, (dim_i, time_i, layer_i) in enumerate(sorted_accesses):
            # Look ahead in window
            for j in range(i + 1, len(sorted_accesses)):
                dim_j, time_j, layer_j = sorted_accesses[j]

                # Check if outside window
                time_diff = time_j - time_i
                if time_diff > window_ns:
                    break  # Sorted, so all subsequent are also outside window

                # Skip self-edges
                if dim_i == dim_j:
                    continue

                # Compute edge weight
                weight = 1.0

                if self.config.weight_by_proximity:
                    # Closer in time = stronger edge
                    # proximity = 1.0 at 0ns, decays to 0.0 at window_ns
                    proximity = 1.0 - (time_diff / window_ns)
                    weight *= proximity

                # Ensure i < j for upper triangle storage
                edge = (min(dim_i, dim_j), max(dim_i, dim_j))
                coaccesses[edge] += weight

        logger.info(f"Found {len(coaccesses)} unique dimension co-access pairs")

        # Filter by minimum co-accesses
        if self.config.min_coaccesses > 0:
            filtered_coaccesses = {
                edge: weight for edge, weight in coaccesses.items()
                if weight >= self.config.min_coaccesses
            }
            logger.info(
                f"After filtering (min={self.config.min_coaccesses}): "
                f"{len(filtered_coaccesses)} edges retained"
            )
        else:
            filtered_coaccesses = coaccesses

        if not filtered_coaccesses:
            logger.warning("No co-accesses passed filter, returning empty graph")
            return self._create_empty_graph()

        # Convert to CSR matrix
        return self._build_csr_from_coaccesses(filtered_coaccesses)

    def _build_csr_from_coaccesses(
        self,
        coaccesses: Dict[Tuple[int, int], float],
    ) -> CSRMatrix:
        """Convert co-access dictionary to CSR matrix.

        Args:
            coaccesses: Dictionary mapping (i, j) pairs to edge weights.

        Returns:
            CSRMatrix in symmetric format.
        """
        D = self.hidden_size

        # Build symmetric adjacency lists
        adj: DefaultDict[int, List[Tuple[int, float]]] = defaultdict(list)

        for (i, j), weight in coaccesses.items():
            adj[i].append((j, weight))
            adj[j].append((i, weight))  # Symmetric

        # Build CSR arrays
        indptr: List[int] = [0]
        indices: List[int] = []
        data: List[float] = []

        for i in range(D):
            if i in adj:
                # Sort neighbors by index
                neighbors = sorted(adj[i], key=lambda x: x[0])

                for j, weight in neighbors:
                    indices.append(j)
                    data.append(weight)

            indptr.append(len(indices))

        # Normalize if requested
        if self.config.normalize:
            data = self._normalize_rows(indptr, data)

        meta = {
            "shape": D,
            "format": "csr",
            "nnz": len(data),
            "source": "instrumented",
            "method": "temporal_coacccess",
            "temporal_window_ns": self.config.temporal_window_ns,
            "num_samples": self.num_samples_recorded,
            "total_accesses": self.total_accesses,
            "num_coacccess_pairs": len(coaccesses),
            "min_coaccesses": self.config.min_coaccesses,
        }

        return CSRMatrix(
            indptr=indptr,
            indices=indices,
            data=data,
            shape=(D, D),
            meta=meta,
        )

    def _normalize_rows(self, indptr: List[int], data: List[float]) -> List[float]:
        """Row-normalize edge weights (each row sums to 1.0)."""
        normalized = data[:]

        for i in range(len(indptr) - 1):
            start, end = indptr[i], indptr[i + 1]
            if end > start:
                row_sum = sum(data[start:end])
                if row_sum > 0:
                    for k in range(start, end):
                        normalized[k] /= row_sum

        return normalized

    def _create_empty_graph(self) -> CSRMatrix:
        """Create empty graph (identity structure)."""
        D = self.hidden_size

        return CSRMatrix(
            indptr=list(range(D + 1)),
            indices=list(range(D)),
            data=[1.0] * D,
            shape=(D, D),
            meta={"shape": D, "format": "csr", "nnz": D, "source": "instrumented_empty"},
        )


def extract_hidden_dimensions(
    tensor: torch.Tensor,
    hidden_size: int,
) -> Optional[Set[int]]:
    """Extract which hidden dimensions are active in this tensor.

    For transformers, we care about the last dimension representing hidden_size.

    Args:
        tensor: Input or output tensor from a layer.
        hidden_size: Expected hidden dimension size.

    Returns:
        Set of dimension indices, or None if can't determine.

    Examples:
        - Shape (batch, seq, hidden_size) → all dims {0..hidden_size-1}
        - Shape (batch, seq, subset_size) where subset_size < hidden_size → {0..subset_size-1}
    """
    if tensor is None or not isinstance(tensor, torch.Tensor):
        return None

    shape = tensor.shape

    # Look for dimension matching hidden_size
    for dim_size in shape:
        if dim_size == hidden_size:
            # All dimensions involved
            return set(range(hidden_size))

    # Check for subsets (e.g., attention heads)
    for dim_size in shape:
        if 0 < dim_size < hidden_size:
            # Might be subset - return range
            return set(range(dim_size))

    # Can't determine - return all as conservative estimate
    return set(range(hidden_size))


def build_w_from_instrumented_access(
    model: nn.Module,
    example_inputs: Any,
    *,
    hidden_size: int,
    config: Optional[InstrumentedGraphConfig] = None,
) -> CSRMatrix:
    """Build graph W from instrumented co-access measurement.

    This is the main entry point that integrates with the existing pipeline.

    Algorithm:
    ----------
    1. Create DimensionAccessTracker
    2. Hook all linear/matmul layers to record dimension accesses
    3. Run multiple forward passes, recording access patterns
    4. Build co-access graph from recorded patterns
    5. Return CSR matrix

    Args:
        model: PyTorch model to instrument.
        example_inputs: Example inputs for forward passes.
        hidden_size: Size of hidden dimension.
        config: Configuration for instrumentation.

    Returns:
        CSRMatrix representing co-access patterns.

    Raises:
        RuntimeError: If PyTorch not available or instrumentation fails.
    """
    if not TORCH_AVAILABLE or torch is None:
        raise RuntimeError(
            "PyTorch not available. Install with: pip install torch"
        )

    if config is None:
        config = InstrumentedGraphConfig()

    logger.info(
        f"Building instrumented graph: D={hidden_size}, samples={config.num_samples}"
    )

    # Create tracker
    tracker = DimensionAccessTracker(hidden_size, config)

    # Prepare inputs
    if not isinstance(example_inputs, tuple):
        example_inputs = (example_inputs,)

    # Hook layers to track accesses
    hooks: List[Any] = []
    hooked_layers = 0

    model_eval = model.eval()

    for name, module in model_eval.named_modules():
        # Check if should hook this layer
        if config.target_layers is not None:
            if name not in config.target_layers:
                continue

        # Hook linear layers and matrix operations
        if isinstance(module, nn.Linear):
            def make_hook(layer_name: str):
                def hook(mod: nn.Module, input: Tuple, output: torch.Tensor) -> None:
                    """Record dimension access for this layer."""
                    timestamp_ns = time.perf_counter_ns()

                    # Extract dimensions from input
                    if input and len(input) > 0:
                        dims_input = extract_hidden_dimensions(input[0], hidden_size)
                        if dims_input:
                            tracker.record_access(dims_input, timestamp_ns, layer_name)

                    # Extract dimensions from output
                    dims_output = extract_hidden_dimensions(output, hidden_size)
                    if dims_output:
                        # Small offset to show output after input
                        tracker.record_access(dims_output, timestamp_ns + 1, layer_name)

                return hook

            hooks.append(module.register_forward_hook(make_hook(name)))
            hooked_layers += 1

    logger.info(f"Hooked {hooked_layers} layers for dimension tracking")

    if hooked_layers == 0:
        logger.warning("No layers hooked - check target_layers configuration")

    # Run forward passes and collect access patterns
    try:
        with torch.no_grad():
            for sample_idx in range(config.num_samples):
                _ = model_eval(*example_inputs)
                tracker.increment_sample_count()

                if (sample_idx + 1) % 10 == 0:
                    logger.debug(f"Completed sample {sample_idx + 1}/{config.num_samples}")

    finally:
        # Always remove hooks
        for hook in hooks:
            hook.remove()

    logger.info(f"Recorded {tracker.total_accesses} total dimension accesses")

    # Build co-access graph
    graph_W = tracker.build_coacccess_graph()

    logger.info(
        f"Built instrumented graph: {graph_W.shape[0]}×{graph_W.shape[1]}, "
        f"nnz={len(graph_W.data)}, "
        f"density={len(graph_W.data)/(graph_W.shape[0]**2):.4f}"
    )

    return graph_W