"""
Utilities for applying mathematically-sound, gradient-preserving permutations
to model parameters.
"""
import torch
import torch.nn as nn
from typing import Optional, Union
from torch.nn import Parameter

TensorOrParameter = torch.Tensor | Parameter

# ---------------------------------------------------------------------------
# Safe, Out-of-Place Permutation Utilities (IASP 2.0)
# These functions create a new tensor and re-assign .data to avoid
# aliasing issues with in-place operations on slices.
# ---------------------------------------------------------------------------

def safe_permute_rows(param: TensorOrParameter, idx: torch.Tensor):
    """Safely permutes rows of a 2D tensor out-of-place."""
    assert param.ndim == 2, "Input must be a 2D tensor."
    device = param.device
    p_safe = idx.to(device)
    param.data = param.data.index_select(0, p_safe).contiguous()

def safe_permute_cols(param: TensorOrParameter, idx: torch.Tensor):
    """Safely permutes columns of a 2D tensor out-of-place."""
    assert param.ndim == 2, "Input must be a 2D tensor."
    device = param.device
    p_safe = idx.to(device)
    param.data = param.data.index_select(1, p_safe).contiguous()

def safe_permute_vector(param: TensorOrParameter, idx: torch.Tensor):
    """Safely permutes elements of a 1D tensor out-of-place."""
    assert param.ndim == 1, "Input must be a 1D tensor."
    device = param.device
    p_safe = idx.to(device)
    param.data = param.data.index_select(0, p_safe).contiguous()


# ---------------------------------------------------------------------------
# In-place Permutation Utilities (Legacy)
# Kept for reference or for contexts where aliasing is not a concern.
# ---------------------------------------------------------------------------

def inplace_permute_rows(param: TensorOrParameter, idx: torch.Tensor):
    """Alias-free, in-place permutation of a parameter's rows (dim 0)."""
    with torch.no_grad():
        new_data = param.data.index_select(0, idx.to(param.device)).clone()
        param.data.copy_(new_data)

def inplace_permute_cols(param: TensorOrParameter, idx: torch.Tensor):
    """Alias-free, in-place permutation of a parameter's columns (dim 1)."""
    with torch.no_grad():
        if param.ndim not in [2, 3]:
            raise ValueError(f"Unsupported ndim ({param.ndim}) for column permute")
        new_data = param.data.index_select(1, idx.to(param.device)).clone()
        param.data.copy_(new_data)

def inplace_permute_vector(param: TensorOrParameter, idx: torch.Tensor):
    """Alias-free, in-place permutation of a 1D parameter tensor (vector)."""
    if param.ndim != 1:
        raise ValueError("Input must be a 1D vector.")
    with torch.no_grad():
        new_data = param.data.index_select(0, idx.to(param.device)).clone()
        param.data.copy_(new_data)

def inplace_permute_in_proj_split(
        weight: TensorOrParameter,
        idx: torch.Tensor,
        bias: Optional[TensorOrParameter] = None
    ):
    """
    Alias-free, in-place permutation for Mamba's split in_proj layer.
    """
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
            if bias.shape[0] != 2 * d:
                 raise ValueError("bias.shape[0] must equal 2 * d_inner for split permutation.")
            # Use a temporary tensor for the bias as well
            b_data_clone = bias.data.clone()
            bias.data[:d].copy_(b_data_clone[:d].index_select(0, idx))
            bias.data[d:].copy_(b_data_clone[d:].index_select(0, idx))

def alias_free_rows_slice(param: TensorOrParameter, idx: torch.Tensor, start: int, end: int):
    """
    Performs an alias-free row permutation on a slice of a parameter tensor.
    This is useful for layers with combined weights (e.g., Mamba's double-width conv).
    """
    with torch.no_grad():
        assert 0 <= start < end <= param.shape[0], \
               f"Slice [{start}:{end}] out of range for param with shape {param.shape}"
        # Clone the selected slice, permute it, and then copy it back.
        # This ensures the source and destination do not alias.
        device = param.device
        idx = idx.to(device)
        
        original_slice = param.data[start:end]
        permuted_slice = original_slice.index_select(0, idx).clone()
        param.data[start:end].copy_(permuted_slice)

def alias_free_vector_slice(param: TensorOrParameter, idx: torch.Tensor, start: int, end: int):
    """
    Performs an alias-free permutation on a slice of a 1D parameter tensor.
    """
    with torch.no_grad():
        if param.ndim != 1:
            raise ValueError("Input must be a 1D vector.")
        assert 0 <= start < end <= param.shape[0], \
               f"Slice [{start}:{end}] out of range for param with shape {param.shape}"
        
        device = param.device
        idx = idx.to(device)
        
        original_slice = param.data[start:end]
        permuted_slice = original_slice.index_select(0, idx).clone()
        param.data[start:end].copy_(permuted_slice) 