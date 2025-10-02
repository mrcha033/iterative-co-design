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
        expected_dim: Optional[int] = None,
    ) -> None:
        _ensure_torch("ActivationCollector")
        self.model = model
        self.targets = list(targets)
        self.dtype = dtype
        self.device_guard = device_guard
        self.transfer_batch_size = int(transfer_batch_size) if transfer_batch_size else None
        self.expected_dim = int(expected_dim) if expected_dim and expected_dim > 0 else None
        self._stats: "OrderedDict[str, _ActivationStats]" = OrderedDict()
        self._layer_meta: "OrderedDict[str, MutableMapping[str, object]]" = OrderedDict()
        self._handles: List[Any] = []
        self._selection_info: Dict[str, object] = {}

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
                batch_size = tensor.shape[0]
                seq_len = tensor.shape[1]
                feature_dim = tensor.shape[2]

                # Validate: feature dimension should match expected_dim, not sequence length
                if self.expected_dim is not None:
                    if seq_len == self.expected_dim and feature_dim != self.expected_dim:
                        # Detected likely axis confusion: sequence length matches expected_dim
                        import warnings
                        warnings.warn(
                            f"Layer {name}: sequence_length={seq_len} matches expected_dim={self.expected_dim}, "
                            f"but feature_dim={feature_dim}. This suggests the tensor may have shape "
                            f"(batch, hidden, seq) instead of (batch, seq, hidden). Skipping this layer.",
                            UserWarning,
                            stacklevel=3,
                        )
                        return
                    if feature_dim != self.expected_dim:
                        # Feature dimension doesn't match expected - will be filtered later
                        pass

                # Average over sequence to get (batch, features)
                tensor = tensor.mean(dim=1)
            elif tensor.ndim == 2:
                feature_dim = tensor.shape[1]

                # Validate: warn if feature_dim looks suspiciously like a sequence length
                if self.expected_dim is not None and feature_dim != self.expected_dim:
                    # Check if this might be a (batch, seq) tensor instead of (batch, features)
                    # Sequence lengths are typically powers of 2 or common values like 512, 1024, 2048, 4096
                    common_seq_lens = {128, 256, 512, 1024, 2048, 4096, 8192}
                    if feature_dim in common_seq_lens and feature_dim > self.expected_dim:
                        import warnings
                        warnings.warn(
                            f"Layer {name}: feature_dim={feature_dim} looks like a sequence length "
                            f"(expected_dim={self.expected_dim}). This may indicate incorrect axis selection. "
                            f"This layer will be filtered out.",
                            UserWarning,
                            stacklevel=3,
                        )
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

    def covariance(
        self,
        *,
        expected_dim: Optional[int] = None,
    ) -> Tuple[TorchTensor, List[Dict[str, object]]]:
        if not self._stats:
            raise RuntimeError("No activations captured")
        stats_items = list(self._stats.items())
        covariances = [stats.covariance() for _, stats in stats_items]
        shapes = [cov.shape for cov in covariances]
        available_shapes = [list(map(int, shape)) for shape in shapes]

        selected_indices = list(range(len(covariances)))
        applied_expected_filter = False
        expected_shape = None
        if expected_dim is not None and expected_dim > 0:
            expected_dim_int = int(expected_dim)
            expected_shape = (expected_dim_int, expected_dim_int)
            filtered = [idx for idx, shape in enumerate(shapes) if shape == expected_shape]
            if filtered:
                selected_indices = filtered
                applied_expected_filter = True

        if not selected_indices:
            selected_indices = list(range(len(covariances)))

        # Align on the dominant shape among selected indices.
        from collections import Counter

        selected_shapes = [shapes[idx] for idx in selected_indices]
        if len(set(selected_shapes)) > 1:
            shape_counts = Counter(selected_shapes)
            most_common_shape, _ = shape_counts.most_common(1)[0]
            selected_indices = [idx for idx in selected_indices if shapes[idx] == most_common_shape]
            selected_shapes = [shapes[idx] for idx in selected_indices]

        if not selected_indices:
            raise RuntimeError("No valid covariance matrices found")

        selected_covariances = [covariances[idx] for idx in selected_indices]
        base = selected_covariances[0].clone()
        for cov in selected_covariances[1:]:
            base += cov
        cov = base / len(selected_covariances)
        selected_shape = selected_covariances[0].shape
        matched_expected_dim = bool(
            expected_shape is not None and selected_shape == expected_shape
        )

        layers_meta: List[Dict[str, object]] = []
        for idx, (name, stats) in enumerate(stats_items):
            layer_meta = dict(self._layer_meta.get(name, {}))
            layer_meta.setdefault("name", name)
            layer_meta["count"] = stats.count
            layer_meta["selected"] = idx in selected_indices
            if not layer_meta["selected"]:
                feature_dim = int(layer_meta.get("feature_dim", 0) or 0)
                if expected_dim is not None and feature_dim != int(expected_dim):
                    layer_meta.setdefault("ignored_reason", "feature_dim_mismatch")
            layers_meta.append(layer_meta)

        self._selection_info = {
            "available_shapes": available_shapes,
            "selected_shape": list(map(int, selected_shape)),
            "expected_dim": int(expected_dim) if expected_dim is not None else None,
            "matched_expected_dim": matched_expected_dim,
            "applied_expected_filter": applied_expected_filter,
        }

        return cov, layers_meta

    @property
    def selection_info(self) -> Dict[str, object]:
        return dict(self._selection_info)

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
        expected_dim=cfg.expected_dim,
    )
    try:
        collector.run(inputs_iter, cfg.samples)
        matrix, layers_meta = collector.covariance(expected_dim=cfg.expected_dim)
    finally:
        collector.close()

    if cfg.whiten:
        matrix = _whiten_covariance(matrix)

    selection_info = collector.selection_info
    selected_dim = int(matrix.shape[0])
    available_dims = sorted({
        int(val)
        for val in (
            layer.get("feature_dim")
            for layer in layers_meta
        )
        if isinstance(val, (int, float)) and int(val) > 0
    })

    for layer_meta in layers_meta:
        raw_dim = layer_meta.get("feature_dim")
        feature_dim = int(raw_dim) if isinstance(raw_dim, (int, float)) else 0
        if feature_dim != selected_dim and not layer_meta.get("ignored_reason"):
            layer_meta.setdefault("ignored_reason", "feature_dim_mismatch")

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
        "feature_dim": selected_dim,
        "expected_dim": int(cfg.expected_dim) if cfg.expected_dim is not None else None,
        "available_feature_dims": available_dims,
        "selection": selection_info,
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
