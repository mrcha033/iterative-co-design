"""
Co-design algorithms for neural network optimization.

This package provides implementations of co-design algorithms including
HDS (Hardware-aware Dynamic Sparsity), IASP (Iterative Activation Sparsity and Permutation),
and modularity analysis tools.
"""

from .hds import apply_hds
from .iasp import run_iasp_on_mamba, run_iasp_on_bert
from .modularity import calculate_modularity
from .layout_aware import apply_layout_aware_hds_finetuning

__all__ = [
    "apply_hds",
    "run_iasp_on_mamba",
    "run_iasp_on_bert",
    "calculate_modularity",
    "apply_layout_aware_hds_finetuning",
]
