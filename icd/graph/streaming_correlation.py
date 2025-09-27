"""Streaming correlation computation with bounded memory footprint."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np
import torch

from .correlation import CorrelationConfig, correlation_to_csr
from icd.core.graph import CSRMatrix


@dataclass
class StreamingAccumulator:
    feature_dim: int
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        self.sum = torch.zeros(self.feature_dim, dtype=self.dtype)
        self.sum_outer = torch.zeros((self.feature_dim, self.feature_dim), dtype=self.dtype)
        self.count = 0

    def update(self, batch: torch.Tensor) -> None:
        if batch.ndim != 2:
            raise ValueError("Expected (batch, features) tensor")
        if batch.shape[1] != self.feature_dim:
            raise ValueError("Feature dimension mismatch")
        if batch.numel() == 0:
            return
        batch = batch.to(dtype=self.dtype)
        self.sum += batch.sum(dim=0)
        self.sum_outer += batch.t().mm(batch)
        self.count += batch.shape[0]

    def covariance(self) -> torch.Tensor:
        if self.count == 0:
            raise RuntimeError("No samples processed")
        mean = self.sum / float(self.count)
        cov = self.sum_outer / float(self.count) - torch.outer(mean, mean)
        cov = (cov + cov.t()) * 0.5
        return cov

    def correlation(self) -> torch.Tensor:
        cov = self.covariance()
        std = torch.sqrt(torch.clamp(torch.diag(cov), min=1e-12))
        denom = torch.outer(std, std)
        corr = torch.zeros_like(cov)
        mask = denom > 0
        corr[mask] = cov[mask] / denom[mask]
        corr.fill_diagonal_(1.0)
        return corr


def stream_batches(iterator: Iterable[torch.Tensor] | Iterator[torch.Tensor]) -> Iterator[torch.Tensor]:
    for batch in iterator:
        if isinstance(batch, np.ndarray):
            batch = torch.from_numpy(batch)
        if not isinstance(batch, torch.Tensor):
            raise TypeError("Batches must be torch tensors or numpy arrays")
        if batch.ndim == 1:
            batch = batch.unsqueeze(0)
        yield batch


def compute_streaming_correlation(
    iterator: Iterable[torch.Tensor] | Iterator[torch.Tensor],
    feature_dim: int,
    *,
    cfg: CorrelationConfig | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[CSRMatrix, dict]:
    """Compute a correlation CSR matrix from a sample iterator.

    The iterator can be substantially larger than GPU memory because the
    accumulator only stores the first and second order statistics.
    """

    cfg = cfg or CorrelationConfig()
    acc = StreamingAccumulator(feature_dim=feature_dim, dtype=dtype)
    total_samples = 0
    for batch in stream_batches(iterator):
        acc.update(batch)
        total_samples += batch.shape[0]

    corr = acc.correlation()
    corr = torch.relu(corr)
    csr = correlation_to_csr(corr, cfg=cfg)
    meta = {
        "samples": total_samples,
        "feature_dim": feature_dim,
        "dtype": str(dtype),
        "mode": "streaming",
    }
    return csr, meta


__all__ = [
    "StreamingAccumulator",
    "compute_streaming_correlation",
    "stream_batches",
]
