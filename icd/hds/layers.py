"""Structured sparsity-aware layer implementations."""

from __future__ import annotations

from typing import Optional

import torch

from .topk import TopKMasker, TopKMaskerConfig

__all__ = ["NMLinear"]


class NMLinear(torch.nn.Module):
    """Linear layer with differentiable N:M mask."""

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
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.linear = torch.nn.Linear(self.in_features, self.out_features, bias=bias)
        config = masker_config or TopKMaskerConfig(active=n_active, group_size=m_group)
        mask_size = self.out_features * self.in_features
        self.masker = TopKMasker(mask_size, config=config)

    def forward(
        self,
        input: torch.Tensor,
        *,
        step: Optional[int] = None,
        sample: Optional[bool] = None,
        temperature: Optional[float] = None,
    ) -> torch.Tensor:
        mask = self.masker(step=step, sample=sample, temperature=temperature)
        mask_matrix = mask.view(self.out_features, self.in_features)
        weight = self.linear.weight * mask_matrix
        return torch.nn.functional.linear(input, weight, self.linear.bias)

    def masked_weight(self) -> torch.Tensor:
        mask_matrix = self.masker.last_mask().view(self.out_features, self.in_features)
        return self.linear.weight * mask_matrix

    def load_from_linear(self, linear: torch.nn.Linear) -> None:
        with torch.no_grad():
            self.linear.weight.copy_(linear.weight.detach())
            if self.linear.bias is not None and linear.bias is not None:
                self.linear.bias.copy_(linear.bias.detach())

