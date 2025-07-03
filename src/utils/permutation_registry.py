"""
Layer-Type Registry for Safe Permutations
=========================================

Provides a registry of permutation rules for different layer types to prevent
accidental incorrect permutations.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Type, Callable, Tuple
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


@dataclass
class PermutationRule:
    """Defines permutation rules for a specific layer type."""
    
    # Axes that can be permuted (None means this axis should not be permuted)
    weight_out_axis: Optional[int] = None  # Output features axis
    weight_in_axis: Optional[int] = None   # Input features axis
    bias_axis: Optional[int] = None        # Bias permutation axis
    
    # Custom validation function
    validator: Optional[Callable[[nn.Module], bool]] = None
    
    # Whether this layer type requires special handling
    requires_special_handling: bool = False
    
    # Error message if permutation is attempted on unsupported layer
    error_msg: Optional[str] = None


# Global registry of layer types and their permutation rules
LAYER_PERMUTATION_REGISTRY: Dict[Type[nn.Module], PermutationRule] = {
    # Standard layers
    nn.Linear: PermutationRule(
        weight_out_axis=0,  # Rows (output features)
        weight_in_axis=1,   # Columns (input features)
        bias_axis=0         # Bias follows output features
    ),
    
    # Convolution layers
    nn.Conv1d: PermutationRule(
        weight_out_axis=0,  # Output channels
        weight_in_axis=1,   # Input channels
        bias_axis=0         # Bias follows output channels
    ),
    nn.Conv2d: PermutationRule(
        weight_out_axis=0,  # Output channels
        weight_in_axis=1,   # Input channels
        bias_axis=0
    ),
    nn.Conv3d: PermutationRule(
        weight_out_axis=0,  # Output channels
        weight_in_axis=1,   # Input channels
        bias_axis=0
    ),
    
    # Normalization layers - DO NOT PERMUTE
    nn.BatchNorm1d: PermutationRule(
        error_msg="BatchNorm parameters should not be permuted directly. Update num_features instead."
    ),
    nn.BatchNorm2d: PermutationRule(
        error_msg="BatchNorm parameters should not be permuted directly. Update num_features instead."
    ),
    nn.LayerNorm: PermutationRule(
        error_msg="LayerNorm parameters should not be permuted directly. Update normalized_shape instead."
    ),
    nn.GroupNorm: PermutationRule(
        error_msg="GroupNorm parameters should not be permuted. Permutation breaks group structure."
    ),
    
    # Embedding layers
    nn.Embedding: PermutationRule(
        weight_out_axis=0,  # Vocabulary dimension
        weight_in_axis=1,   # Embedding dimension
        validator=lambda m: m.padding_idx is None,  # No permutation if padding_idx is set
        error_msg="Cannot permute embedding with padding_idx set"
    ),
    
    # RNN layers - require special handling
    nn.LSTM: PermutationRule(
        requires_special_handling=True,
        error_msg="LSTM requires special handling for gate permutations. Use permute_lstm_weights()."
    ),
    nn.GRU: PermutationRule(
        requires_special_handling=True,
        error_msg="GRU requires special handling for gate permutations. Use permute_gru_weights()."
    ),
}


def register_layer_type(
    layer_type: Type[nn.Module],
    rule: PermutationRule,
    override: bool = False
) -> None:
    """
    Register a new layer type with its permutation rules.
    
    Args:
        layer_type: The layer class to register
        rule: Permutation rule for this layer type
        override: Whether to override existing registration
    """
    if layer_type in LAYER_PERMUTATION_REGISTRY and not override:
        raise ValueError(
            f"Layer type {layer_type.__name__} already registered. "
            "Set override=True to replace."
        )
    
    LAYER_PERMUTATION_REGISTRY[layer_type] = rule
    logger.info(f"Registered permutation rule for {layer_type.__name__}")


def get_permutation_rule(layer: nn.Module) -> PermutationRule:
    """
    Get permutation rule for a layer, checking inheritance chain.
    
    Args:
        layer: The layer instance
        
    Returns:
        PermutationRule for this layer type
        
    Raises:
        ValueError: If layer type is not registered
    """
    # Check exact type first
    layer_type = type(layer)
    if layer_type in LAYER_PERMUTATION_REGISTRY:
        return LAYER_PERMUTATION_REGISTRY[layer_type]
    
    # Check inheritance chain
    for registered_type, rule in LAYER_PERMUTATION_REGISTRY.items():
        if isinstance(layer, registered_type):
            logger.warning(
                f"Using rule for {registered_type.__name__} "
                f"for derived type {layer_type.__name__}"
            )
            return rule
    
    # Not found
    raise ValueError(
        f"No permutation rule registered for layer type {layer_type.__name__}. "
        f"Register with register_layer_type() or add to LAYER_PERMUTATION_REGISTRY."
    )


def validate_permutation(
    layer: nn.Module,
    param_name: str,
    axis: int
) -> Tuple[bool, Optional[str]]:
    """
    Validate if a permutation is allowed for a specific layer and parameter.
    
    Args:
        layer: The layer instance
        param_name: Name of parameter ('weight', 'bias', etc.)
        axis: Axis to permute along
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        rule = get_permutation_rule(layer)
    except ValueError as e:
        return False, str(e)
    
    # Check if layer requires special handling
    if rule.requires_special_handling:
        return False, rule.error_msg or f"{type(layer).__name__} requires special handling"
    
    # Check if general error message exists
    if rule.error_msg and not any([rule.weight_out_axis, rule.weight_in_axis, rule.bias_axis]):
        return False, rule.error_msg
    
    # Run custom validator if exists
    if rule.validator and not rule.validator(layer):
        return False, rule.error_msg or f"Custom validation failed for {type(layer).__name__}"
    
    # Check specific parameter and axis
    if param_name == 'weight':
        if axis == 0 and rule.weight_out_axis is not None:
            return True, None
        elif axis == 1 and rule.weight_in_axis is not None:
            return True, None
        else:
            return False, f"Cannot permute weight axis {axis} for {type(layer).__name__}"
    
    elif param_name == 'bias':
        if axis == 0 and rule.bias_axis is not None:
            return True, None
        else:
            return False, f"Cannot permute bias for {type(layer).__name__}"
    
    else:
        return False, f"Unknown parameter name: {param_name}"


def get_safe_permutation_axes(layer: nn.Module) -> Dict[str, List[int]]:
    """
    Get all safe permutation axes for a layer's parameters.
    
    Args:
        layer: The layer instance
        
    Returns:
        Dict mapping parameter names to list of permutable axes
    """
    try:
        rule = get_permutation_rule(layer)
    except ValueError:
        return {}
    
    if rule.requires_special_handling or rule.error_msg:
        return {}
    
    axes = {}
    
    if hasattr(layer, 'weight') and layer.weight is not None:
        weight_axes = []
        if rule.weight_out_axis is not None:
            weight_axes.append(rule.weight_out_axis)
        if rule.weight_in_axis is not None:
            weight_axes.append(rule.weight_in_axis)
        if weight_axes:
            axes['weight'] = weight_axes
    
    if hasattr(layer, 'bias') and layer.bias is not None:
        if rule.bias_axis is not None:
            axes['bias'] = [rule.bias_axis]
    
    return axes


# Additional registrations for common custom layers
def register_common_custom_layers():
    """Register common custom layer types."""
    
    # LoRA layers (if available)
    try:
        from peft.tuners.lora import LoRALinear
        register_layer_type(
            LoRALinear,
            PermutationRule(
                weight_out_axis=0,
                weight_in_axis=1,
                error_msg="LoRA weights should be permuted with base weights"
            )
        )
    except ImportError:
        pass
    
    # Grouped convolution
    register_layer_type(
        nn.Conv2d,
        PermutationRule(
            weight_out_axis=0,
            weight_in_axis=None,  # Don't permute input for grouped conv
            bias_axis=0,
            validator=lambda m: m.groups == 1,
            error_msg="Grouped convolutions require special handling"
        ),
        override=True  # Override the default Conv2d rule
    )


# Initialize common custom layers on import
register_common_custom_layers() 