"""
Utilities for applying mathematically-sound, gradient-preserving permutations
to model parameters.
"""
import torch
import torch.nn as nn
from typing import Optional

def _check(param: nn.Parameter):
    if not isinstance(param, nn.Parameter):
        raise TypeError("param must be nn.Parameter")

def inplace_permute_rows(param: nn.Parameter, idx: torch.Tensor):
    """Alias-free, in-place permutation of a parameter's rows (dim 0)."""
    _check(param)
    with torch.no_grad():
        new_data = param.data.index_select(0, idx.to(param.device)).clone()
        param.data.copy_(new_data)

def inplace_permute_cols(param: nn.Parameter, idx: torch.Tensor):
    """Alias-free, in-place permutation of a parameter's columns (dim 1)."""
    _check(param)
    with torch.no_grad():
        if param.ndim not in [2, 3]:
            raise ValueError(f"Unsupported ndim ({param.ndim}) for column permute")
        new_data = param.data.index_select(1, idx.to(param.device)).clone()
        param.data.copy_(new_data)

def inplace_permute_vector(param: nn.Parameter, idx: torch.Tensor):
    """Alias-free, in-place permutation of a 1D parameter tensor (vector)."""
    _check(param)
    if param.ndim != 1:
        raise ValueError("Input must be a 1D vector.")
    with torch.no_grad():
        new_data = param.data.index_select(0, idx.to(param.device)).clone()
        param.data.copy_(new_data)

def inplace_permute_in_proj_split(
        weight: nn.Parameter,
        idx: torch.Tensor,
        bias: Optional[nn.Parameter] = None
    ):
    """
    Alias-free, in-place permutation for Mamba's split in_proj layer.
    """
    _check(weight)
    d = idx.numel()
    if weight.shape[0] != 2 * d:
        raise ValueError("weight.shape[0] must equal 2 * d_inner for split permutation.")

    device = weight.device
    idx = idx.to(device)

    with torch.no_grad():
        # Use a temporary tensor to avoid aliasing issues within the split
        w_data_clone = weight.data.clone()
        weight.data[:d].copy_(w_data_clone[:d].index_select(0, idx))
        weight.data[d:].copy_(w_data_clone[d:].index_select(0, idx))
        
        if bias is not None:
            _check(bias)
            if bias.shape[0] != 2 * d:
                 raise ValueError("bias.shape[0] must equal 2 * d_inner for split permutation.")
            # Use a temporary tensor for the bias as well
            b_data_clone = bias.data.clone()
            bias.data[:d].copy_(b_data_clone[:d].index_select(0, idx))
            bias.data[d:].copy_(b_data_clone[d:].index_select(0, idx)) 