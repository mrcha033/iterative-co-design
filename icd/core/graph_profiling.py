"""Profiling-based graph construction for memory co-access patterns.

This module implements ACTUAL memory access profiling to build graph W.
Key Idea:
----------
Instead of assuming spatial locality (dimension i correlates with i+1),
we profile which dimensions are ACTUALLY accessed together in memory/time.

This makes the co-access graph mechanistically sound:
- Edges represent true temporal co-access patterns
- Weights proportional to access frequency
- Captures real cache/memory behavior

Methods:
--------
1. PyTorch Profiler: Track operations and their memory accesses
2. Temporal windowing: Dimensions accessed within time window get edges
3. Frequency weighting: More co-accesses → stronger edges
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional, Set

try:
    import torch
    from torch.profiler import profile, ProfilerActivity, record_function
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore

from .graph import CSRMatrix

logger = logging.getLogger(__name__)

__all__ = [
    "ProfilingConfig",
    "build_w_from_profiling",
    "build_w_from_torch_profiler",
    "MemoryAccessPattern",
]


@dataclass
class ProfilingConfig:
    """Configuration for profiling-based graph construction."""

    samples: int = 10
    temporal_window_ms: float = 0.1  # Co-access within 100μs = edge
    min_coaccesses: int = 2  # Minimum co-accesses to create edge
    normalize: bool = True
    cache_line_size: int = 64  # bytes (typical L1/L2 cache line)
    weight_by_frequency: bool = True
    weight_by_size: bool = True  # Weight by tensor size (larger = more important)

    @classmethod
    def from_dict(cls, data: Dict[str, Any] | None) -> "ProfilingConfig":
        if not data:
            return cls()
        return cls(
            samples=int(data.get("samples", 10)),
            temporal_window_ms=float(data.get("temporal_window_ms", 0.1)),
            min_coaccesses=int(data.get("min_coaccesses", 2)),
            normalize=bool(data.get("normalize", True)),
            cache_line_size=int(data.get("cache_line_size", 64)),
            weight_by_frequency=bool(data.get("weight_by_frequency", True)),
            weight_by_size=bool(data.get("weight_by_size", True)),
        )


@dataclass
class MemoryAccessPattern:
    """Represents memory access patterns from profiling."""

    dimension: int
    timestamp_ns: int
    operation: str
    tensor_shape: Tuple[int, ...]
    memory_size_bytes: int

    def temporal_distance_ms(self, other: "MemoryAccessPattern") -> float:
        """Compute temporal distance in milliseconds."""
        return abs(self.timestamp_ns - other.timestamp_ns) / 1e6


def _extract_dimension_from_shape(shape: Tuple[int, ...], hidden_size: int) -> Optional[Set[int]]:
    """Extract which hidden dimensions are involved in this tensor.

    For a tensor with shape (batch, seq_len, hidden_size), we care about the
    hidden_size dimension. Returns set of dimension indices involved.

    Examples:
        Shape (2, 128, 768) with hidden_size=768 → dimensions {0..767}
        Shape (768, 3072) (weight matrix) → depends on interpretation
    """
    if not shape:
        return None

    # Find dimension matching hidden_size
    for dim_size in shape:
        if dim_size == hidden_size:
            # All dimensions involved
            return set(range(hidden_size))

    # Check if this is a slice/subset
    for dim_size in shape:
        if dim_size < hidden_size and dim_size > 0:
            # Heuristic: might be subset of dimensions
            # More sophisticated logic could use actual indices
            return set(range(dim_size))

    return None


def build_w_from_torch_profiler(
    model: Any,
    example_inputs: Any,
    *,
    hidden_size: int,
    config: ProfilingConfig | None = None,
) -> CSRMatrix:
    """Build graph W from PyTorch profiler memory access patterns.

    Core Algorithm:
    ---------------
    1. Run model with profiler, record all operations and timestamps
    2. Extract which dimensions are accessed in each operation
    3. For operations within temporal window, create edges between dimensions
    4. Weight edges by co-access frequency

    Args:
        model: PyTorch model to profile
        example_inputs: Example inputs for profiling
        hidden_size: Size of hidden dimension (D in paper)
        config: Profiling configuration

    Returns:
        CSRMatrix graph where W[i,j] = co-access weight between dimensions i,j
    """
    if not TORCH_AVAILABLE or torch is None:
        logger.warning("PyTorch not available, falling back to heuristic graph")
        from .graph_pytorch import build_w_from_pytorch
        return build_w_from_pytorch(model, example_inputs)

    if config is None:
        config = ProfilingConfig()

    logger.info(f"Building profiling-based graph: D={hidden_size}, samples={config.samples}")

    # Collect memory access patterns
    access_patterns: List[MemoryAccessPattern] = []

    model_eval = model.eval()
    inputs = example_inputs if isinstance(example_inputs, tuple) else (example_inputs,)

    # Profile multiple samples to get robust patterns
    for sample_idx in range(config.samples):
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if torch.cuda.is_available() else [ProfilerActivity.CPU],
            record_shapes=True,
            with_stack=False,  # Faster
        ) as prof:
            with record_function("model_forward"):
                with torch.no_grad():
                    _ = model_eval(*inputs)

        # Extract access patterns from profiler events
        for event in prof.events():
            if not hasattr(event, 'input_shapes') or not event.input_shapes:
                continue

            for shape_tuple in event.input_shapes:
                if not shape_tuple:
                    continue

                # Extract dimensions involved
                dims = _extract_dimension_from_shape(tuple(shape_tuple), hidden_size)
                if dims is None:
                    continue

                # Estimate memory size
                mem_size = 1
                for s in shape_tuple:
                    mem_size *= s
                mem_size *= 4  # Assume float32 = 4 bytes

                # Record each dimension access
                for dim in dims:
                    access_patterns.append(MemoryAccessPattern(
                        dimension=dim,
                        timestamp_ns=event.time_range.start if hasattr(event, 'time_range') else 0,
                        operation=str(event.name),
                        tensor_shape=tuple(shape_tuple),
                        memory_size_bytes=mem_size,
                    ))

    if not access_patterns:
        logger.warning("No memory access patterns captured, falling back to heuristic")
        from .graph_pytorch import build_w_from_pytorch
        return build_w_from_pytorch(model, example_inputs)

    logger.info(f"Captured {len(access_patterns)} memory access events")

    # Build co-access graph
    D = hidden_size
    coaccesses: Dict[Tuple[int, int], float] = defaultdict(float)

    # For each pair of accesses within temporal window, create edge
    temporal_window_ns = config.temporal_window_ms * 1e6

    for i, access_i in enumerate(access_patterns):
        for j, access_j in enumerate(access_patterns[i+1:], start=i+1):
            # Check temporal proximity
            if access_i.temporal_distance_ms(access_j) > config.temporal_window_ms:
                continue

            dim_i = access_i.dimension
            dim_j = access_j.dimension

            if dim_i == dim_j:
                continue

            # Ensure i < j for upper triangle
            if dim_i > dim_j:
                dim_i, dim_j = dim_j, dim_i

            # Compute edge weight
            weight = 1.0

            if config.weight_by_frequency:
                weight *= 1.0  # Each co-access adds to frequency

            if config.weight_by_size:
                # Weight by geometric mean of memory sizes
                size_i = access_i.memory_size_bytes
                size_j = access_j.memory_size_bytes
                weight *= (size_i * size_j) ** 0.5 / 1024  # Normalize by KB

            coaccesses[(dim_i, dim_j)] += weight

    # Filter by minimum co-accesses
    filtered_coaccesses = {
        (i, j): w for (i, j), w in coaccesses.items()
        if w >= config.min_coaccesses
    }

    logger.info(f"Graph: {len(filtered_coaccesses)} edges from {len(coaccesses)} raw co-accesses")

    if not filtered_coaccesses:
        logger.warning("No co-accesses passed filter, falling back to heuristic")
        from .graph_pytorch import build_w_from_pytorch
        return build_w_from_pytorch(model, example_inputs)

    # Build CSR matrix
    indptr: List[int] = [0]
    indices: List[int] = []
    data: List[float] = []

    for i in range(D):
        row_data: List[Tuple[int, float]] = []
        for (row, col), weight in filtered_coaccesses.items():
            if row == i:
                row_data.append((col, weight))
            elif col == i and row < i:
                # Symmetric: add transpose entry
                row_data.append((row, weight))

        # Sort by column index
        row_data.sort(key=lambda x: x[0])

        for col, weight in row_data:
            indices.append(col)
            data.append(float(weight))

        indptr.append(len(indices))

    # Normalize if requested
    if config.normalize:
        # Simple row normalization
        for i in range(D):
            start, end = indptr[i], indptr[i+1]
            if end > start:
                row_sum = sum(data[start:end])
                if row_sum > 0:
                    for k in range(start, end):
                        data[k] /= row_sum

    meta = {
        "shape": D,
        "format": "csr",
        "nnz": len(data),
        "source": "profiling",
        "method": "torch_profiler",
        "samples": config.samples,
        "temporal_window_ms": config.temporal_window_ms,
        "num_access_patterns": len(access_patterns),
        "num_raw_coaccesses": len(coaccesses),
        "num_filtered_coaccesses": len(filtered_coaccesses),
    }

    return CSRMatrix(
        indptr=indptr,
        indices=indices,
        data=data,
        shape=(D, D),
        meta=meta,
    )


def build_w_from_profiling(
    model: Any,
    example_inputs: Any,
    *,
    hidden_size: int,
    method: str = "torch_profiler",
    config: ProfilingConfig | None = None,
) -> CSRMatrix:
    """Build graph W from profiling (dispatcher for different methods).

    Args:
        model: Model to profile
        example_inputs: Inputs for profiling
        hidden_size: Hidden dimension size
        method: Profiling method ("torch_profiler", "ncu", etc.)
        config: Profiling configuration

    Returns:
        CSRMatrix graph from profiling
    """
    method_lower = method.lower()

    if method_lower == "torch_profiler":
        return build_w_from_torch_profiler(
            model, example_inputs,
            hidden_size=hidden_size,
            config=config,
        )
    elif method_lower == "ncu":
        # Future: NCU-based profiling
        logger.warning(f"NCU profiling not yet implemented, falling back to torch_profiler")
        return build_w_from_torch_profiler(
            model, example_inputs,
            hidden_size=hidden_size,
            config=config,
        )
    else:
        raise ValueError(f"Unknown profiling method: {method}")