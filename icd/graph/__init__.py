"""Correlation utilities namespace with lazy imports."""

from importlib import import_module
from typing import Any

__all__ = [
    "CorrelationConfig",
    "collect_correlations",
    "correlation_to_csr",
    "save_correlation_artifacts",
    "ClusteringConfig",
    "cluster_graph",
    "StreamingAccumulator",
    "compute_streaming_correlation",
    "stream_batches",
]


_CORRELATION_MEMBERS = {
    "CorrelationConfig",
    "collect_correlations",
    "correlation_to_csr",
    "save_correlation_artifacts",
}
_CLUSTERING_MEMBERS = {"ClusteringConfig", "cluster_graph"}
_STREAMING_MEMBERS = {
    "StreamingAccumulator",
    "compute_streaming_correlation",
    "stream_batches",
}


def __getattr__(name: str) -> Any:
    if name in _CORRELATION_MEMBERS:
        module = import_module(".correlation", __name__)
        return getattr(module, name)
    if name in _CLUSTERING_MEMBERS:
        module = import_module(".clustering", __name__)
        return getattr(module, name)
    if name in _STREAMING_MEMBERS:
        module = import_module(".streaming_correlation", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
