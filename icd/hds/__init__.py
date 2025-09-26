"""Hardware-native Differentiable Sparsity (HDS) scaffolding.

This package will host differentiable N:M sparsity primitives (e.g., 2:4 with
Gumbel-TopK sampling) and integration utilities for iterative co-design loops.

Current contents are structural placeholders; concrete implementations will be
landed incrementally in alignment with TASKS_TEMP.md.
"""

from .topk import TopKMasker  # noqa: F401
from .layers import NMLinear  # noqa: F401

__all__ = [
    "TopKMasker",
    "NMLinear",
]
