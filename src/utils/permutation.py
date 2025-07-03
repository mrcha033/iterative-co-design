"""
Utilities for applying mathematically-sound, gradient-preserving permutations
to model parameters.
"""
import torch
import torch.nn as nn
from typing import Optional, Union, Callable
from torch.nn import Parameter

TensorOrParameter = Union[torch.Tensor, Parameter]

# ---------------------------------------------------------------------------
# Safe, Out-of-Place Permutation Utilities (IASP 2.0)
# These functions create a new tensor and re-assign .data to avoid
# aliasing issues with in-place operations on slices.
# ---------------------------------------------------------------------------

def safe_permute_rows(param: TensorOrParameter, idx: torch.Tensor, *, optimizer: Optional[torch.optim.Optimizer] = None):
    """
    Safely permutes rows (dimension 0) of a tensor out-of-place.
    Works for tensors of any dimensionality (1D, 2D, 3D, etc.).
    For 1D: permutes elements
    For 2D: permutes rows  
    For 3D+: permutes along first dimension (e.g., output channels for Conv)
    
    Uses in-place copy to preserve TensorImpl for optimizer state consistency.
    
    Args:
        param: Tensor or Parameter to permute
        idx: Permutation indices
        optimizer: Optional optimizer - if param is a Parameter and optimizer is None, raises error
    """
    with torch.no_grad():
        assert param.ndim >= 1, "Input must have at least one dimension."
        device = param.device
        p_safe = idx.to(device)
        # Use in-place copy to preserve the same TensorImpl
        param.data.copy_(param.data.index_select(0, p_safe))
        
        # Mandatory optimizer sync for Parameters
        if isinstance(param, nn.Parameter):
            if optimizer is None:
                raise RuntimeError(
                    "Optimizer must be provided when permuting Parameters to ensure state synchronization. "
                    "Pass optimizer=None explicitly only for non-Parameter tensors."
                )
            permute_optimizer_state(optimizer, param, idx, axis=0)

def safe_permute_cols(param: TensorOrParameter, idx: torch.Tensor, *, optimizer: Optional[torch.optim.Optimizer] = None):
    """
    Safely permutes columns (dimension 1) of a tensor out-of-place.
    Requires at least 2 dimensions.
    
    Uses in-place copy to preserve TensorImpl for optimizer state consistency.
    
    Args:
        param: Tensor or Parameter to permute
        idx: Permutation indices
        optimizer: Optional optimizer - if param is a Parameter and optimizer is None, raises error
    """
    with torch.no_grad():
        assert param.ndim >= 2, "Input must have at least 2 dimensions for column permutation."
        device = param.device
        p_safe = idx.to(device)
        # Use in-place copy to preserve the same TensorImpl
        param.data.copy_(param.data.index_select(1, p_safe))
        
        # Mandatory optimizer sync for Parameters
        if isinstance(param, nn.Parameter):
            if optimizer is None:
                raise RuntimeError(
                    "Optimizer must be provided when permuting Parameters to ensure state synchronization. "
                    "Pass optimizer=None explicitly only for non-Parameter tensors."
                )
            permute_optimizer_state(optimizer, param, idx, axis=1)

def safe_permute_vector(param: TensorOrParameter, idx: torch.Tensor, *, optimizer: Optional[torch.optim.Optimizer] = None):
    """
    Safely permutes elements of a 1D tensor out-of-place.
    
    Uses in-place copy to preserve TensorImpl for optimizer state consistency.
    
    Args:
        param: Tensor or Parameter to permute
        idx: Permutation indices
        optimizer: Optional optimizer - if param is a Parameter and optimizer is None, raises error
    """
    with torch.no_grad():
        assert param.ndim == 1, "Input must be a 1D tensor."
        device = param.device
        p_safe = idx.to(device)
        # Use in-place copy to preserve the same TensorImpl
        param.data.copy_(param.data.index_select(0, p_safe))
        
        # Mandatory optimizer sync for Parameters
        if isinstance(param, nn.Parameter):
            if optimizer is None:
                raise RuntimeError(
                    "Optimizer must be provided when permuting Parameters to ensure state synchronization. "
                    "Pass optimizer=None explicitly only for non-Parameter tensors."
                )
            permute_optimizer_state(optimizer, param, idx, axis=0)

# ---------------------------------------------------------------------------
# Axis-aware permutation with semantic validation
# ---------------------------------------------------------------------------

def permute_tensor_axis(
    param: TensorOrParameter, 
    idx: torch.Tensor,
    axis: int,
    axis_validator: Optional[Callable[[TensorOrParameter, int], bool]] = None
) -> None:
    """
    Permute tensor along specified axis with optional validation.
    
    Args:
        param: Tensor or Parameter to permute
        idx: Permutation indices
        axis: Axis to permute along (can be negative)
        axis_validator: Optional callable to validate axis choice
    
    Raises:
        ValueError: If axis is invalid or validator rejects it
    """
    with torch.no_grad():
        # Normalize axis
        if axis < 0:
            axis = param.ndim + axis
        
        if not (0 <= axis < param.ndim):
            raise ValueError(f"Axis {axis} out of bounds for tensor with {param.ndim} dimensions")
        
        # Validate if validator provided
        if axis_validator and not axis_validator(param, axis):
            raise ValueError(f"Axis {axis} validation failed for tensor shape {param.shape}")
        
        # Move indices to correct device
        idx = idx.to(param.device)
        
        # Perform in-place permutation
        param.data.copy_(param.data.index_select(axis, idx))

# ---------------------------------------------------------------------------
# Optimizer state permutation
# ---------------------------------------------------------------------------

def permute_optimizer_state(
    optimizer: torch.optim.Optimizer,
    param: Parameter,
    idx: torch.Tensor,
    axis: int = 0
) -> None:
    """
    Permute optimizer state tensors to match parameter permutation.
    
    Handles common optimizer states like momentum buffers and adaptive learning rates.
    
    Args:
        optimizer: The optimizer containing state for param
        param: The parameter that was permuted
        idx: The permutation indices that were applied
        axis: The axis along which permutation was applied
    """
    if param not in optimizer.state:
        return  # No state to permute
    
    state = optimizer.state[param]
    
    with torch.no_grad():
        # Handle different optimizer types
        for key, value in state.items():
            if not isinstance(value, torch.Tensor):
                continue
                
            # Common state tensors that need permutation
            if key in ['momentum_buffer', 'exp_avg', 'exp_avg_sq', 'accumulator', 'delta']:
                if value.shape == param.shape:
                    # State has same shape as parameter, permute along same axis
                    permute_tensor_axis(value, idx, axis)
                elif value.ndim == 1 and axis == 0 and param.ndim > 1:
                    # Some optimizers store per-channel statistics as 1D
                    if value.numel() == param.shape[0]:
                        value.copy_(value.index_select(0, idx.to(value.device)))

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