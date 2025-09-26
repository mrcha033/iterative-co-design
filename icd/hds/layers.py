"""Structured sparsity-aware layer placeholders."""

from __future__ import annotations

from typing import Optional

import torch

from .topk import TopKMasker, TopKMaskerConfig


class NMLinear(torch.nn.Module):
    """Placeholder N:M sparse linear layer.

    This wrapper will eventually own both dense weights and a differentiable
    mask (via :class:`TopKMasker`). For now it simply subclasses ``nn.Linear``
    and provides hooks for future integration so we can gradually wire it into
    the orchestrator without breaking determinism.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        n_active: int = 2,
        m_group: int = 4,
        masker_config: Optional[TopKMaskerConfig] = None,
    ) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)
        mask_size = out_features * in_features
        config = masker_config or TopKMaskerConfig(active=n_active, group_size=m_group)
        self.masker = TopKMasker(mask_size, config=config)

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # pragma: no cover - pending implementation
        raise NotImplementedError("NMLinear forward pending HDS integration")

    def load_from_linear(self, linear: torch.nn.Linear) -> None:
        """Copy weights/bias from an existing dense layer.

        Useful when swapping in NMLinear for existing models before adding mask
        learning. Bias is copied if present; weights are cloned to avoid tying
        storages unexpectedly.
        """

        with torch.no_grad():
            self.linear.weight.copy_(linear.weight.detach())
            if self.linear.bias is not None and linear.bias is not None:
                self.linear.bias.copy_(linear.bias.detach())

