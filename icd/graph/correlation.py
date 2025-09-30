"""Correlation graph construction for IASP."""

from __future__ import annotations

import json
import random
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterable, Iterator, List, MutableMapping, Optional, Sequence, Tuple

try:  # pragma: no cover - numpy optional for seeding
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:  # pragma: no cover - torch is an optional dependency for correlation utilities
    import torch  # type: ignore[import-not-found]
except Exception as _torch_exc:  # pragma: no cover - exercised on CPU-only CI
    torch = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR: Exception | None = _torch_exc
else:  # pragma: no branch
    _TORCH_IMPORT_ERROR = None

if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    import torch as _torch

    TorchTensor = _torch.Tensor
    TorchDType = _torch.dtype
    TorchModule = _torch.nn.Module
else:  # pragma: no cover - runtime fallbacks when torch is unavailable
    TorchTensor = Any
    TorchDType = Any
    TorchModule = Any

_TORCH_HAS_CORE_APIS = bool(
    torch is not None
    and hasattr(torch, "zeros")
    and hasattr(torch, "Tensor")
    and hasattr(torch, "nn")
)

if torch is not None and hasattr(torch, "Tensor"):
    _TENSOR_TYPES: Tuple[type, ...] = (torch.Tensor,)  # type: ignore[misc]
else:  # pragma: no cover - exercised when torch unavailable
    _TENSOR_TYPES = tuple()


def _ensure_torch(op: str) -> None:
    """Raise a clear error when a correlation helper requires PyTorch."""

    if not _TORCH_HAS_CORE_APIS:
        message = (
            f"{op} requires PyTorch. Install the 'experiments' extra (pip install 'repermute[experiments]') "
            "or otherwise provide a compatible torch implementation."
        )
        if _TORCH_IMPORT_ERROR is not None:
            raise RuntimeError(message) from _TORCH_IMPORT_ERROR
        raise RuntimeError(message)

from icd.core import graph as graph_mod
from icd.core.graph import CSRMatrix

__all__ = [
    "CorrelationConfig",
    "collect_correlations",
    "correlation_to_csr",
    "save_correlation_artifacts",
]


_DEFAULT_DTYPE: Any
if torch is not None and hasattr(torch, "float32"):
    _DEFAULT_DTYPE = torch.float32  # type: ignore[assignment]
else:  # pragma: no cover - fallback when torch unavailable
    _DEFAULT_DTYPE = "float32"


@dataclass
class CorrelationConfig:
    mode: str = "activation"
    layers: Optional[Sequence[str]] = None
    samples: int = 8
    seed: Optional[int] = None
    dtype: TorchDType | Any = _DEFAULT_DTYPE
    device_guard: bool = True
    threshold: float = 0.0
    normalize: str = "sym"
    nnz_cap: Optional[int] = None
    whiten: bool = False
    transfer_batch_size: Optional[int] = None
    expected_dim: Optional[int] = None  # Force dimension to match model hidden_size


class _ActivationStats:
    """Running statistics for a single layer's activations."""

    def __init__(self, feature_dim: int, dtype: TorchDType | Any, device: Any) -> None:
        _ensure_torch("_ActivationStats")
        self.device = torch.device(device)
        self.dtype = dtype
        self.sum = torch.zeros(feature_dim, dtype=dtype, device=self.device)
        self.sum_outer = torch.zeros(feature_dim, feature_dim, dtype=dtype, device=self.device)
        self.count = 0

    def update(self, activations: TorchTensor) -> None:
        if activations.ndim != 2:
            raise ValueError("expected 2-D activations (batch, features)")
        if activations.numel() == 0:
            return
        activations = activations.to(device=self.device, dtype=self.dtype)
        self.count += activations.shape[0]
        self.sum += activations.sum(dim=0)
        self.sum_outer += activations.t().mm(activations)

    def covariance(self) -> TorchTensor:
        if self.count == 0:
            raise RuntimeError("No activations recorded")
        mean = self.sum / float(self.count)
        cov = self.sum_outer / float(self.count) - torch.outer(mean, mean)
        cov = (cov + cov.t()) * 0.5
        return cov


class ActivationCollector:
    """Hook-based activation statistics collector with optional CPU staging."""

    def __init__(
        self,
        model: TorchModule,
        targets: Sequence[str],
        dtype: TorchDType | Any,
        *,
        device_guard: bool = True,
        transfer_batch_size: Optional[int] = None,
    ) -> None:
        _ensure_torch("ActivationCollector")
        self.model = model
        self.targets = list(targets)
        self.dtype = dtype
        self.device_guard = device_guard
        self.transfer_batch_size = int(transfer_batch_size) if transfer_batch_size else None
        self._stats: "OrderedDict[str, _ActivationStats]" = OrderedDict()
        self._layer_meta: "OrderedDict[str, MutableMapping[str, object]]" = OrderedDict()
        self._handles: List[Any] = []

        for name, module in model.named_modules():
            if not self.targets or name in self.targets:
                handle = module.register_forward_hook(self._hook(name))
                self._handles.append(handle)

    def _get_stats(self, name: str, feature_dim: int, device: Any) -> _ActivationStats:
        stats = self._stats.get(name)
        if stats is None:
            stats_device = torch.device("cpu") if self.device_guard else device
            stats = _ActivationStats(feature_dim, self.dtype, stats_device)
            self._stats[name] = stats
            layer_meta = self._layer_meta.setdefault(
                name,
                {
                    "name": name,
                    "feature_dim": feature_dim,
                    "samples": 0,
                    "capture_device": str(device),
                },
            )
            layer_meta["storage_device"] = str(stats.device)
        else:
            layer_meta = self._layer_meta[name]
            layer_meta.setdefault("capture_device", str(device))
        return stats

    def _hook(self, name: str):
        def _capture(module, inputs, output):  # type: ignore[unused-argument]
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(tensor, torch.Tensor):
                return
            tensor = tensor.detach()
            if tensor.ndim == 0:
                return
            tensor = tensor.to(dtype=self.dtype)
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)

            # For 3D tensors (batch, seq, features), take last dimension as features
            # For 2D tensors (batch, features), use as-is
            # For higher dims, flatten
            if tensor.ndim == 3:
                feature_dim = tensor.shape[2]
                # Average over sequence to get (batch, features)
                tensor = tensor.mean(dim=1)
            elif tensor.ndim == 2:
                feature_dim = tensor.shape[1]
            else:
                tensor = tensor.reshape(tensor.shape[0], -1)
                feature_dim = tensor.shape[1]

            stats = self._get_stats(name, feature_dim, tensor.device)
            batch_size = self.transfer_batch_size or tensor.shape[0]
            if batch_size <= 0:
                batch_size = tensor.shape[0]
            for chunk in tensor.split(batch_size, dim=0):
                stats.update(chunk)
                self._layer_meta[name]["samples"] = int(self._layer_meta[name].get("samples", 0)) + chunk.shape[0]

        return _capture

    def run(self, iterator: Iterator[Tuple], samples: int) -> None:
        self.model.eval()
        with torch.inference_mode():
            for _, inputs in zip(range(samples), iterator):
                if not isinstance(inputs, tuple):
                    inputs = (inputs,)
                self.model(*inputs)

    def covariance(self) -> Tuple[TorchTensor, List[Dict[str, object]]]:
        if not self._stats:
            raise RuntimeError("No activations captured")
        covariances = [stats.covariance() for stats in self._stats.values()]

        # Check if all covariances have the same shape
        shapes = [cov.shape for cov in covariances]
        if len(set(shapes)) > 1:
            # Different dimensions - only aggregate those matching the most common dimension
            from collections import Counter
            shape_counts = Counter(shapes)
            most_common_shape, _ = shape_counts.most_common(1)[0]
            filtered_covs = [cov for cov in covariances if cov.shape == most_common_shape]
            if not filtered_covs:
                raise RuntimeError("No valid covariance matrices found")
            covariances = filtered_covs

        base = covariances[0].clone()
        for cov in covariances[1:]:
            base += cov
        cov = base / len(covariances)
        layers_meta: List[Dict[str, object]] = []
        for name, stats in self._stats.items():
            layer_meta = dict(self._layer_meta.get(name, {}))
            layer_meta.setdefault("name", name)
            layer_meta["count"] = stats.count
            layers_meta.append(layer_meta)
        return cov, layers_meta

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def _input_iterator(example_inputs: object, samples: int) -> Iterator[Tuple]:
    if callable(example_inputs):  # Callable producing tuple
        def generator() -> Iterator[Tuple]:
            for idx in range(samples):
                result = example_inputs(idx)
                if not isinstance(result, tuple):
                    result = (result,)
                yield result
        return generator()

    tensor_guard = _TENSOR_TYPES + (str, bytes)
    if isinstance(example_inputs, Iterable) and not isinstance(example_inputs, tensor_guard):
        iterator = iter(example_inputs)
        peek = next(iterator, None)
        if peek is None:
            return iter(())
        def chained() -> Iterator[Tuple]:
            first = peek
            if not isinstance(first, tuple):
                first = (first,)
            yield first
            for item in iterator:
                if not isinstance(item, tuple):
                    item = (item,)
                yield item
        return chained()

    def default_iter() -> Iterator[Tuple]:
        inp = example_inputs if isinstance(example_inputs, tuple) else (example_inputs,)
        for _ in range(samples):
            yield inp
    return default_iter()


def _set_deterministic_seed(seed: int) -> None:
    _ensure_torch("set_deterministic_seed")
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - CUDA optional in CI
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)


def _whiten_covariance(matrix: torch.Tensor) -> torch.Tensor:
    _ensure_torch("_whiten_covariance")
    diag = matrix.diagonal()
    eps = torch.finfo(matrix.dtype).eps if matrix.is_floating_point() else 1e-12
    denom = torch.sqrt(torch.clamp(diag, min=eps))
    inv = torch.zeros_like(denom)
    mask = denom > 0
    inv[mask] = 1.0 / denom[mask]
    whitened = matrix * inv.unsqueeze(0) * inv.unsqueeze(1)
    whitened.fill_diagonal_(1.0)
    return whitened


def collect_correlations(
    model: TorchModule,
    example_inputs: object,
    *,
    cfg: CorrelationConfig,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    _ensure_torch("collect_correlations")
    if cfg.seed is not None:
        _set_deterministic_seed(int(cfg.seed))

    if cfg.samples <= 0:
        raise ValueError("cfg.samples must be positive")

    inputs_iter = _input_iterator(example_inputs, cfg.samples)

    if cfg.mode != "activation":
        raise NotImplementedError("Only activation-based correlation is supported currently")

    targets = list(cfg.layers or [])
    collector = ActivationCollector(
        model,
        targets,
        dtype=cfg.dtype,
        device_guard=cfg.device_guard,
        transfer_batch_size=cfg.transfer_batch_size,
    )
    try:
        collector.run(inputs_iter, cfg.samples)
        matrix, layers_meta = collector.covariance()
    finally:
        collector.close()

    if cfg.whiten:
        matrix = _whiten_covariance(matrix)

    meta = {
        "mode": cfg.mode,
        "targets": targets or "auto",
        "samples": cfg.samples,
        "dtype": str(cfg.dtype),
        "device_guard": cfg.device_guard,
        "transfer_batch_size": cfg.transfer_batch_size,
        "seed": cfg.seed,
        "whiten": cfg.whiten,
        "layers": layers_meta,
    }
    return matrix.cpu(), meta


def correlation_to_csr(
    matrix: TorchTensor,
    *,
    cfg: CorrelationConfig,
) -> CSRMatrix:
    _ensure_torch("correlation_to_csr")
    matrix = matrix.clone()
    matrix = torch.relu(matrix)
    matrix.fill_diagonal_(0.0)
    indices = matrix.nonzero(as_tuple=False)
    rows: Dict[int, List[Tuple[int, float]]] = {}
    for idx in range(indices.shape[0]):
        i, j = indices[idx].tolist()
        value = float(matrix[i, j])
        if cfg.threshold > 0.0 and value < cfg.threshold:
            continue
        rows.setdefault(i, []).append((j, value))

    indptr = [0]
    col_indices: List[int] = []
    data: List[float] = []
    for i in range(matrix.shape[0]):
        row = sorted(rows.get(i, []))
        for j, value in row:
            col_indices.append(j)
            data.append(value)
        indptr.append(len(col_indices))

    csr = CSRMatrix(indptr=indptr, indices=col_indices, data=data, shape=(matrix.shape[0], matrix.shape[0]), meta={
        "source": "correlation",
        "threshold": cfg.threshold,
    })

    if cfg.normalize == "sym":
        csr = graph_mod._normalize_sym(csr)
    elif cfg.normalize == "row":
        csr = graph_mod._normalize_row(csr)

    cap = cfg.nnz_cap or int(min(0.05 * matrix.shape[0] ** 2, 50_000_000))
    csr = graph_mod._cap_and_prune(csr, cap)
    return csr


def save_correlation_artifacts(out_dir: str, matrix: torch.Tensor, meta: Dict[str, object]) -> None:
    _ensure_torch("save_correlation_artifacts")
    payload = {
        "meta": meta,
        "shape": list(matrix.shape),
    }
    path = f"{out_dir}/correlation_meta.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    torch.save(matrix, f"{out_dir}/correlation.pt")
