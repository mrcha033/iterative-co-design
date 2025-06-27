"""
Model wrapper utilities for permutation and optimization.

This module provides a wrapper class for PyTorch models that enables weight
permutation and other model transformations required for iterative co-design.
The wrapper maintains model functionality while allowing for structural changes
to optimize memory layout and hardware efficiency.

Key components:
- ModelWrapper: Main wrapper class for model permutation operations
- Weight permutation utilities that preserve model semantics
- Device management and tensor handling for wrapped models
"""

import torch
import torch.nn as nn
from typing import List
import logging

logger = logging.getLogger(__name__)


class ModelWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.device = (
            next(model.parameters()).device
            if list(model.parameters())
            else torch.device("cpu")
        )

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def cuda(self, *args, **kwargs):
        self.device = torch.device("cuda")
        self.model.cuda(*args, **kwargs)
        return self

    def cpu(self, *args, **kwargs):
        self.device = torch.device("cpu")
        self.model.cpu(*args, **kwargs)
        return self

    # ------------------------------------------------------------------
    # Attribute delegation helpers
    # ------------------------------------------------------------------

    @property
    def config(self):
        """Expose the underlying model's config (e.g., for hidden_size access)."""
        return getattr(self.model, "config", None)
