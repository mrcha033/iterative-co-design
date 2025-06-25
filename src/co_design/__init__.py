"""
Co-design algorithms for neural network optimization.

This package provides implementations of co-design algorithms including
HDS (Hardware-aware Dynamic Sparsity), IASP (Iterative Activation Sparsity and Permutation),
and modularity analysis tools.
"""

from .hds import apply_hds
from .iasp import find_optimal_permutation
from .modularity import calculate_modularity
from .layout_aware import apply_layout_aware_hds_finetuning

__all__ = [
    "apply_hds",
    "find_optimal_permutation",
    "calculate_modularity",
    "apply_layout_aware_hds_finetuning",
]
