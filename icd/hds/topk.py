"""Differentiable Top-K mask implementation for structured sparsity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

__all__ = ["TopKMasker", "TopKMaskerConfig"]


@dataclass
class TopKMaskerConfig:
    """Configuration for differentiable Top-K masking."""

    group_size: int = 4
    active: int = 2
    temperature_init: float = 1.0
    temperature_final: float = 0.1
    anneal_steps: int = 10_000
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.group_size <= 0:
            raise ValueError("group_size must be positive")
        if self.active <= 0:
            raise ValueError("active must be positive")
        if self.active > self.group_size:
            raise ValueError("active cannot exceed group_size")
        if self.temperature_init <= 0 or self.temperature_final <= 0:
            raise ValueError("temperatures must be strictly positive")


class TopKMasker(torch.nn.Module):
    """Differentiable N:M mask generator with straight-through gradients."""

    def __init__(self, size: int, config: Optional[TopKMaskerConfig] = None) -> None:
        super().__init__()
        self.size = int(size)
        self.config = config or TopKMaskerConfig()
        if self.size % self.config.group_size != 0:
            raise ValueError("size must be divisible by group_size")

        self.group_count = self.size // self.config.group_size
        logits = torch.zeros(self.group_count, self.config.group_size)
        self.logits = torch.nn.Parameter(logits)
        self.register_buffer("_step", torch.zeros(1, dtype=torch.long), persistent=False)
        self.register_buffer("_last_mask", torch.zeros_like(logits), persistent=False)

    def forward(
        self,
        *,
        step: Optional[int] = None,
        sample: Optional[bool] = None,
        temperature: Optional[float] = None,
    ) -> torch.Tensor:
        """Return a differentiable mask flattened to shape ``(size,)``."""

        step_val = int(step if step is not None else int(self._step.item()))
        sample = bool(self.training if sample is None else sample)
        temperature = float(
            temperature if temperature is not None else self.current_temperature(step_val)
        )
        temperature = max(temperature, 1e-6)

        logits = self.logits
        if sample:
            scores = (logits + self._gumbel_noise(logits.shape, step_val)) / temperature
        else:
            scores = logits / temperature

        topk_idx = torch.topk(scores, self.config.active, dim=-1).indices
        mask_hard = torch.zeros_like(logits)
        mask_hard.scatter_(dim=-1, index=topk_idx, value=1.0)

        probs = torch.softmax(logits / temperature, dim=-1)
        mask = mask_hard + probs - probs.detach()
        flat_mask = mask.reshape(-1)

        self._last_mask = mask_hard.detach()
        if step is None and self.training:
            self._step += 1

        return flat_mask

    def current_temperature(self, step: int) -> float:
        """Compute temperature via linear annealing."""

        step = max(0, int(step))
        if self.config.anneal_steps <= 0:
            return float(self.config.temperature_final)
        frac = min(1.0, step / float(self.config.anneal_steps))
        start = float(self.config.temperature_init)
        end = float(self.config.temperature_final)
        return start + frac * (end - start)

    def reset_step(self) -> None:
        self._step.zero_()

    def last_mask(self) -> torch.Tensor:
        return self._last_mask.reshape(-1)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _gumbel_noise(self, shape: Tuple[int, ...], step: int) -> torch.Tensor:
        device = self.logits.device
        if self.config.seed is not None:
            gen = torch.Generator(device=device)
            gen.manual_seed(int(self.config.seed + step))
            uniform = torch.rand(shape, generator=gen, device=device)
        else:
            uniform = torch.rand(shape, device=device)
        uniform = uniform.clamp_(1e-6, 1.0 - 1e-6)
        return -torch.log(-torch.log(uniform))
