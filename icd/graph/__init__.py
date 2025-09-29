"""Correlation utilities namespace with lazy imports."""

from dataclasses import dataclass, field
from pathlib import Path
import sys
from importlib import import_module
from typing import Any, Dict

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
    _ensure_package_search_path()
    if name in _CORRELATION_MEMBERS:
        try:
            module = import_module(".correlation", __name__)
        except ModuleNotFoundError as exc:  # pragma: no cover - exercised when torch missing
            if exc.name and exc.name.startswith("torch"):
                raise ImportError(
                    "icd.graph.correlation requires the optional PyTorch dependency. "
                    "Install the 'experiments' extra (pip install 'repermute[experiments]') to enable correlation utilities."
                ) from exc
            raise
        return getattr(module, name)
    if name in _CLUSTERING_MEMBERS:
        try:
            module = import_module(".clustering", __name__)
        except ModuleNotFoundError as exc:  # pragma: no cover - exercised without networkx
            if exc.name == "networkx":
                if not _FALLBACK_CLUSTERING:
                    @dataclass
                    class _ClusteringConfigFallback:
                        method: str = "louvain"
                        rng_seed: int = 0
                        resolution: float = 1.0
                        fallback_method: str | None = "spectral"
                        modularity_floor: float = 0.35
                        runtime_budget: float | None = None
                        last_meta: Dict[str, object] = field(default_factory=dict, init=False, repr=False)

                    def _cluster_graph_fallback(*args, **kwargs):
                        raise RuntimeError(
                            "cluster_graph requires the optional 'networkx' dependency. "
                            "Install it with pip install networkx to enable clustering."
                        )

                    _FALLBACK_CLUSTERING["ClusteringConfig"] = _ClusteringConfigFallback
                    _FALLBACK_CLUSTERING["cluster_graph"] = _cluster_graph_fallback
                return _FALLBACK_CLUSTERING[name]
            raise
        else:
            return getattr(module, name)
    if name in _STREAMING_MEMBERS:
        module = import_module(".streaming_correlation", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
_FALLBACK_CLUSTERING: Dict[str, Any] = {}
_MODULE_DIR = Path(__file__).resolve().parent


def _ensure_package_search_path() -> None:
    module = sys.modules.get(__name__)
    if module is None:
        return
    path = getattr(module, "__path__", None)
    if not path:
        module.__path__ = [str(_MODULE_DIR)]  # type: ignore[attr-defined]
    spec = getattr(module, "__spec__", None)
    if spec is not None and getattr(spec, "submodule_search_locations", None) is not None:
        if not spec.submodule_search_locations:  # type: ignore[attr-defined]
            spec.submodule_search_locations = [str(_MODULE_DIR)]  # type: ignore[attr-defined]


_ensure_package_search_path()

