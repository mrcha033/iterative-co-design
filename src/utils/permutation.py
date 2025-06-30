"""
Utilities for applying mathematically-sound, gradient-preserving permutations
to model parameters.
"""
import torch
import torch.nn as nn
from typing import Optional

def permute_rows(param: nn.Parameter, p: torch.Tensor) -> nn.Parameter:
    """
    Permutes the rows (dim 0) of a parameter tensor while preserving its
    gradient history. Re-wraps the result in nn.Parameter.
    """
    if not isinstance(param, nn.Parameter):
        raise TypeError("Input must be an nn.Parameter.")
    
    permuted_data = param.data[p].clone()
    return nn.Parameter(permuted_data, requires_grad=param.requires_grad)

def permute_cols(param: nn.Parameter, p: torch.Tensor) -> nn.Parameter:
    """
    Permutes the columns (dim 1) of a parameter tensor while preserving its
    gradient history. Re-wraps the result in nn.Parameter.
    """
    if not isinstance(param, nn.Parameter):
        raise TypeError("Input must be an nn.Parameter.")

    permuted_data = param.data[:, p].clone()
    return nn.Parameter(permuted_data, requires_grad=param.requires_grad)

def permute_vector(param: nn.Parameter, p: torch.Tensor) -> nn.Parameter:
    """
    Permutes a 1D parameter tensor (vector).
    """
    if not isinstance(param, nn.Parameter):
        raise TypeError("Input must be an nn.Parameter.")
    if param.ndim != 1:
        raise ValueError("Input must be a 1D vector.")
        
    permuted_data = param.data[p].clone()
    return nn.Parameter(permuted_data, requires_grad=param.requires_grad)

def permute_in_proj_split(param: nn.Parameter, p: torch.Tensor, bias: Optional[nn.Parameter] = None):
    """
    Permutes the rows of a split linear layer like Mamba's in_proj, where
    the first half and second half of the rows are permuted independently.

    Can optionally handle the bias term as well.
    """
    if not isinstance(param, nn.Parameter):
        raise TypeError("Input must be an nn.Parameter.")
        
    d_inner = p.numel()
    if param.shape[0] != 2 * d_inner:
        raise ValueError("Input parameter shape is not compatible with 2 * d_inner.")
        
    permuted_w_data = torch.cat(
        (param.data[:d_inner][p], param.data[d_inner:][p]), dim=0
    ).clone()
    
    new_w = nn.Parameter(permuted_w_data, requires_grad=param.requires_grad)

    if bias is not None:
        permuted_b_data = torch.cat(
            (bias.data[:d_inner][p], bias.data[d_inner:][p]), dim=0
        ).clone()
        new_b = nn.Parameter(permuted_b_data, requires_grad=bias.requires_grad)
        return new_w, new_b
        
    return new_w 