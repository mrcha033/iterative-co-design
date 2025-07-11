"""
PermutableModel wrapper for safe layer access and permutation application.
"""
import re
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import numpy as np


class PermutableModel(nn.Module):
    """
    A wrapper around neural network models that provides safe layer access
    and permutation application for the iterative co-design framework.
    """
    
    def __init__(self, model: nn.Module, model_type: str, model_name: str):
        """
        Initialize the PermutableModel wrapper.
        
        Args:
            model: The underlying neural network model
            model_type: Type of model ('mamba', 'bert', 'resnet', 'gcn')
            model_name: Name of the model for identification
        """
        super().__init__()
        self.model = model
        self.model_type = model_type.lower()
        self.model_name = model_name
        self._applied_permutations: Dict[str, np.ndarray] = {}
        self._layer_info_cache: Optional[Dict] = None
    
    def forward(self, *args, **kwargs):
        """Forward pass through the underlying model."""
        return self.model(*args, **kwargs)
    
    def get_layer_names(self) -> List[str]:
        """
        Get all layer names in the model.
        
        Returns:
            List of layer names that can be accessed
        """
        layer_names = []
        for name, _ in self.model.named_modules():
            if name:  # Skip empty names
                layer_names.append(name)
        return sorted(layer_names)
    
    def get_layer(self, layer_name: str) -> nn.Module:
        """
        Get a specific layer by name.
        
        Args:
            layer_name: Name of the layer to retrieve
            
        Returns:
            The requested layer module
            
        Raises:
            ValueError: If layer doesn't exist
        """
        try:
            # Handle nested layer names like 'layers.0.mixer'
            layer = self.model
            for attr in layer_name.split('.'):
                if attr.isdigit():
                    layer = layer[int(attr)]
                else:
                    layer = getattr(layer, attr)
            return layer
        except (AttributeError, IndexError, TypeError):
            available_layers = self.get_layer_names()
            raise ValueError(
                f"Layer '{layer_name}' not found in {self.model_name}. "
                f"Available layers: {available_layers[:10]}..."
                f"{'(showing first 10)' if len(available_layers) > 10 else ''}"
            )
    
    def get_layer_info(self, layer_name: str) -> Dict:
        """
        Get detailed information about a layer.
        
        Args:
            layer_name: Name of the layer
            
        Returns:
            Dictionary with layer information
        """
        layer = self.get_layer(layer_name)
        
        info = {
            'name': layer_name,
            'type': type(layer).__name__,
            'parameters': sum(p.numel() for p in layer.parameters()),
            'trainable_parameters': sum(p.numel() for p in layer.parameters() if p.requires_grad),
            'has_weight': hasattr(layer, 'weight'),
            'has_bias': hasattr(layer, 'bias'),
        }
        
        if hasattr(layer, 'weight') and layer.weight is not None:
            info['weight_shape'] = list(layer.weight.shape)
            info['weight_dtype'] = str(layer.weight.dtype)
        
        if hasattr(layer, 'bias') and layer.bias is not None:
            info['bias_shape'] = list(layer.bias.shape)
            info['bias_dtype'] = str(layer.bias.dtype)
        
        return info
    
    def get_permutable_layers(self) -> List[str]:
        """
        Get layers that can be permuted (typically linear/attention layers).
        
        Returns:
            List of layer names that support permutation
        """
        permutable_layers = []
        
        for name, module in self.model.named_modules():
            if self._is_permutable_layer(module):
                permutable_layers.append(name)
        
        return permutable_layers
    
    def _is_permutable_layer(self, module: nn.Module) -> bool:
        """Check if a layer can be permuted."""
        # Linear layers are always permutable
        if isinstance(module, nn.Linear):
            return True
        
        # Model-specific permutable layers
        if self.model_type == 'mamba':
            # Mamba-specific layers that can be permuted
            return any(layer_type in type(module).__name__.lower() 
                      for layer_type in ['mixer', 'mlp', 'attention'])
        elif self.model_type == 'bert':
            # BERT layers that can be permuted
            return any(layer_type in type(module).__name__.lower() 
                      for layer_type in ['attention', 'intermediate', 'output'])
        elif self.model_type == 'resnet':
            # ResNet layers that can be permuted
            return isinstance(module, (nn.Conv2d, nn.Linear))
        elif self.model_type == 'gcn':
            # GCN layers that can be permuted
            return any(layer_type in type(module).__name__.lower() 
                      for layer_type in ['gcn', 'graph', 'conv'])
        
        return False
    
    def apply_permutation(
        self,
        layer_name: str,
        permutation: Union[np.ndarray, List[int], torch.Tensor],
        dimension: str = 'input',
        validate: bool = True
    ) -> None:
        """
        Apply a permutation to a layer's weights.
        
        Args:
            layer_name: Name of the layer to permute
            permutation: Permutation indices
            dimension: Which dimension to permute ('input', 'output', 'both')
            validate: Whether to validate permutation before applying
            
        Raises:
            ValueError: If layer doesn't exist or permutation is invalid
        """
        # Get the layer
        layer = self.get_layer(layer_name)
        
        # Convert permutation to numpy array
        if isinstance(permutation, torch.Tensor):
            perm = permutation.cpu().numpy()
        else:
            perm = np.array(permutation)
        
        # Validate permutation
        if validate:
            self._validate_permutation(layer, perm, dimension)
        
        # Apply permutation based on layer type and dimension
        if isinstance(layer, nn.Linear):
            self._apply_linear_permutation(layer, perm, dimension)
        elif isinstance(layer, nn.Conv2d):
            self._apply_conv_permutation(layer, perm, dimension)
        else:
            self._apply_generic_permutation(layer, perm, dimension)
        
        # Store applied permutation for tracking
        self._applied_permutations[layer_name] = perm
    
    def _validate_permutation(self, layer: nn.Module, perm: np.ndarray, dimension: str):
        """Validate that a permutation is valid for a layer."""
        if not hasattr(layer, 'weight') or layer.weight is None:
            raise ValueError(f"Layer {layer} has no weight to permute")
        
        weight_shape = layer.weight.shape
        
        if dimension == 'input':
            expected_size = weight_shape[1] if len(weight_shape) > 1 else weight_shape[0]
        elif dimension == 'output':
            expected_size = weight_shape[0]
        elif dimension == 'both':
            # For 'both', we need square weights
            if weight_shape[0] != weight_shape[1]:
                raise ValueError(f"Cannot apply 'both' permutation to non-square weight matrix")
            expected_size = weight_shape[0]
        else:
            raise ValueError(f"Invalid dimension: {dimension}")
        
        if len(perm) != expected_size:
            raise ValueError(
                f"Permutation size {len(perm)} doesn't match expected size {expected_size} "
                f"for dimension '{dimension}' of layer with weight shape {weight_shape}"
            )
        
        # Check if permutation is valid (contains all indices exactly once)
        if not np.array_equal(np.sort(perm), np.arange(len(perm))):
            raise ValueError("Invalid permutation: must contain each index exactly once")
    
    def _apply_linear_permutation(self, layer: nn.Linear, perm: np.ndarray, dimension: str):
        """Apply permutation to a linear layer."""
        with torch.no_grad():
            if dimension == 'input':
                # Permute input features: W_new = W[:, perm]
                layer.weight.data = layer.weight.data[:, perm]
            elif dimension == 'output':
                # Permute output features: W_new = W[perm, :]
                layer.weight.data = layer.weight.data[perm, :]
                if layer.bias is not None:
                    layer.bias.data = layer.bias.data[perm]
            elif dimension == 'both':
                # Permute both dimensions: W_new = W[perm, :][:, perm]
                layer.weight.data = layer.weight.data[perm, :][:, perm]
                if layer.bias is not None:
                    layer.bias.data = layer.bias.data[perm]
    
    def _apply_conv_permutation(self, layer: nn.Conv2d, perm: np.ndarray, dimension: str):
        """Apply permutation to a convolutional layer."""
        with torch.no_grad():
            if dimension == 'input':
                # Permute input channels: W_new = W[:, perm, :, :]
                layer.weight.data = layer.weight.data[:, perm, :, :]
            elif dimension == 'output':
                # Permute output channels: W_new = W[perm, :, :, :]
                layer.weight.data = layer.weight.data[perm, :, :, :]
                if layer.bias is not None:
                    layer.bias.data = layer.bias.data[perm]
            elif dimension == 'both':
                # For conv layers, 'both' usually means permuting output channels
                layer.weight.data = layer.weight.data[perm, :, :, :]
                if layer.bias is not None:
                    layer.bias.data = layer.bias.data[perm]
    
    def _apply_generic_permutation(self, layer: nn.Module, perm: np.ndarray, dimension: str):
        """Apply permutation to a generic layer (fallback)."""
        # This is a fallback for custom layer types
        # Try to apply permutation to weight if it exists
        if hasattr(layer, 'weight') and layer.weight is not None:
            with torch.no_grad():
                if dimension == 'input' and len(layer.weight.shape) > 1:
                    layer.weight.data = layer.weight.data[:, perm]
                elif dimension == 'output':
                    layer.weight.data = layer.weight.data[perm]
                elif dimension == 'both' and len(layer.weight.shape) > 1:
                    layer.weight.data = layer.weight.data[perm, :][:, perm]
                
                if hasattr(layer, 'bias') and layer.bias is not None and dimension != 'input':
                    layer.bias.data = layer.bias.data[perm]
    
    def get_applied_permutations(self) -> Dict[str, np.ndarray]:
        """Get all applied permutations."""
        return self._applied_permutations.copy()
    
    def has_permutation(self, layer_name: str) -> bool:
        """Check if a layer has had a permutation applied."""
        return layer_name in self._applied_permutations
    
    def inject_hds_masks(self, layer_name: str, mask: torch.Tensor) -> None:
        """
        Inject HDS (Hardware-Native Differentiable Sparsity) masks into a layer.
        
        Args:
            layer_name: Name of the layer
            mask: Sparsity mask to apply
        """
        layer = self.get_layer(layer_name)
        
        if not hasattr(layer, 'weight') or layer.weight is None:
            raise ValueError(f"Layer {layer_name} has no weight to mask")
        
        # Apply mask to weights
        with torch.no_grad():
            layer.weight.data *= mask
        
        # Store mask for tracking
        if not hasattr(self, '_applied_masks'):
            self._applied_masks = {}
        self._applied_masks[layer_name] = mask
    
    def get_dimension_size(self, layer_name: str, dimension: str = 'input') -> int:
        """
        Get the size of a specific dimension for a layer.
        
        Args:
            layer_name: Name of the layer
            dimension: Which dimension to get size for
            
        Returns:
            Size of the specified dimension
        """
        layer = self.get_layer(layer_name)
        
        if not hasattr(layer, 'weight') or layer.weight is None:
            raise ValueError(f"Layer {layer_name} has no weight")
        
        weight_shape = layer.weight.shape
        
        if dimension == 'input':
            return weight_shape[1] if len(weight_shape) > 1 else weight_shape[0]
        elif dimension == 'output':
            return weight_shape[0]
        else:
            raise ValueError(f"Invalid dimension: {dimension}")
    
    def clone(self) -> 'PermutableModel':
        """Create a deep copy of the model."""
        import copy
        cloned_model = copy.deepcopy(self.model)
        return PermutableModel(cloned_model, self.model_type, self.model_name)
    
    def get_model_summary(self) -> Dict:
        """Get a summary of the model structure."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'total_layers': len(self.get_layer_names()),
            'permutable_layers': len(self.get_permutable_layers()),
            'applied_permutations': len(self._applied_permutations),
            'device': next(self.model.parameters()).device.type if total_params > 0 else 'unknown',
            'dtype': str(next(self.model.parameters()).dtype) if total_params > 0 else 'unknown'
        }