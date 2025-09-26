"""Differentiable Top-K mask scaffolding for structured sparsity.

The eventual implementation will provide:
- Gumbel-TopK sampling with temperature scheduling.
- Straight-through gradient estimator for discrete mask application.
- Deterministic seeding hooks to satisfy SOP/QA requirements.

For now this module exposes a minimal `TopKMasker` class with configuration data
and placeholder methods that raise `NotImplementedError`. Subsequent patches
will flesh out the math and integrate with `NMLinear`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class TopKMaskerConfig:
    """Configuration for differentiable Top-K masking.

    Attributes
    ----------
    group_size: int
        Number of elements per group (e.g., 4 for 2:4 sparsity).
    active: int
        Number of elements to keep active within each group (e.g., 2 for 2:4).
    temperature_init: float
        Initial Gumbel temperature; annealed during training.
    temperature_final: float
        Target temperature when annealing completes.
    anneal_steps: int
        Number of steps over which to anneal the temperature.
    seed: Optional[int]
        Optional RNG seed for reproducibility.
    """

    group_size: int = 4
    active: int = 2
    temperature_init: float = 1.0
    temperature_final: float = 0.1
    anneal_steps: int = 10_000
    seed: Optional[int] = None


class TopKMasker(torch.nn.Module):
    """Placeholder differentiable Top-K masker.

    The forward pass currently raises `NotImplementedError`. It will eventually
    emit binary masks with straight-through gradients based on learned logits.
    """

    def __init__(self, size: int, config: Optional[TopKMaskerConfig] = None) -> None:
        super().__init__()
        self.size = int(size)
        self.config = config or TopKMaskerConfig()
        if self.config.active <= 0 or self.config.group_size <= 0:
            raise ValueError("group_size and active must be positive")
        if self.size % self.config.group_size != 0:
            raise ValueError("size must be divisible by group_size")
        # Logits parameter placeholder; actual initialization TBD.
        self.logits = torch.nn.Parameter(torch.zeros(self.size))

    def forward(self) -> torch.Tensor:  # pragma: no cover - pending implementation
        raise NotImplementedError("TopKMasker forward pass not yet implemented")

    def current_temperature(self, step: int) -> float:
        """Compute temperature along annealing schedule.

        The schedule is linear for now; will be revisited when implementing the
        full masker to match research requirements.
        """

        step = max(0, int(step))
        if self.config.anneal_steps <= 0:
            return float(self.config.temperature_final)
        frac = min(1.0, step / float(self.config.anneal_steps))
        start = float(self.config.temperature_init)
        end = float(self.config.temperature_final)
        return start + frac * (end - start)

