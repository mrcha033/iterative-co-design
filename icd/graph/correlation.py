"""Correlation graph construction for IASP."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch

from icd.core.graph import CSRMatrix
from icd.core import graph as graph_mod

__all__ = ["CorrelationConfig", "collect_correlations", "correlation_to_csr"]


@dataclass
class CorrelationConfig:
    mode: str = "activation"
    layers: Optional[Sequence[str]] = None
    samples: int = 8
    seed: Optional[int] = None
    dtype: torch.dtype = torch.float32
    device_guard: bool = True
    threshold: float = 0.0
    normalize: str = "sym"
    nnz_cap: Optional[int] = None


class _ActivationStats:
    def __init__(self, feature_dim: int, dtype: torch.dtype, device: torch.device) -> None:
        self.sum = torch.zeros(feature_dim, dtype=dtype, device=device)
        self.sum_outer = torch.zeros(feature_dim, feature_dim, dtype=dtype, device=device)
        self.count = 0

    def update(self, activations: torch.Tensor) -> None:
        # activations: (batch, features)
        if activations.numel() == 0:
            return
        self.count += activations.shape[0]
        self.sum += activations.sum(dim=0)
        self.sum_outer += activations.t().mm(activations)

    def covariance(self) -> torch.Tensor:
        if self.count == 0:
            raise RuntimeError("No activations recorded")
        mean = self.sum / float(self.count)
        cov = self.sum_outer / float(self.count) - torch.outer(mean, mean)
        cov = (cov + cov.t()) * 0.5
        return cov


class ActivationCollector:
    def __init__(self, model: torch.nn.Module, targets: Sequence[str], dtype: torch.dtype) -> None:
        self.model = model
        self.targets = list(targets)
        self.dtype = dtype
        self._stats: Dict[str, _ActivationStats] = {}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

        for name, module in model.named_modules():
            if not self.targets or name in self.targets:
                handle = module.register_forward_hook(self._hook(name))
                self._handles.append(handle)

    def _hook(self, name: str):
        def _capture(module, inputs, output):
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(tensor, torch.Tensor):
                return
            tensor = tensor.detach().to(dtype=self.dtype)
            tensor = tensor.reshape(tensor.shape[0], -1)
            stats = self._stats.get(name)
            if stats is None:
                stats = _ActivationStats(tensor.shape[1], self.dtype, tensor.device)
                self._stats[name] = stats
            stats.update(tensor)
        return _capture

    def run(self, iterator: Iterator[Tuple], samples: int) -> None:
        self.model.eval()
        with torch.no_grad():
            for idx, inputs in zip(range(samples), iterator):
                if not isinstance(inputs, tuple):
                    inputs = (inputs,)
                self.model(*inputs)

    def covariance(self) -> torch.Tensor:
        if not self._stats:
            raise RuntimeError("No activations captured")
        covariances = [stats.covariance() for stats in self._stats.values()]
        size = covariances[0].shape[0]
        cov = torch.zeros_like(covariances[0])
        for c in covariances:
            cov += c
        return cov / len(covariances)

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

    if isinstance(example_inputs, Iterable) and not isinstance(example_inputs, (torch.Tensor, str, bytes)):
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


def collect_correlations(
    model: torch.nn.Module,
    example_inputs: object,
    *,
    cfg: CorrelationConfig,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
    inputs_iter = _input_iterator(example_inputs, cfg.samples)

    if cfg.mode != "activation":
        raise NotImplementedError("Only activation-based correlation is supported currently")

    targets = list(cfg.layers or [])
    collector = ActivationCollector(model, targets, dtype=cfg.dtype)
    try:
        collector.run(inputs_iter, cfg.samples)
        matrix = collector.covariance()
    finally:
        collector.close()

    meta = {
        "mode": cfg.mode,
        "targets": targets or "auto",
        "samples": cfg.samples,
        "dtype": str(cfg.dtype),
    }
    return matrix.cpu(), meta


def correlation_to_csr(
    matrix: torch.Tensor,
    *,
    cfg: CorrelationConfig,
) -> CSRMatrix:
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
    payload = {
        "meta": meta,
        "shape": list(matrix.shape),
    }
    path = f"{out_dir}/correlation_meta.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    torch.save(matrix, f"{out_dir}/correlation.pt")
