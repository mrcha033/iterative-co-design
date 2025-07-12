"""
Permutation application module for safely applying permutations to model weights.
"""
import warnings
from typing import List, Dict, Any, Optional, Union

import torch
import torch.nn as nn
import numpy as np

from ..models.permutable_model import PermutableModel
from ..utils.exceptions import IterativeCoDesignError, PermutationApplicationError


class PermutationApplicator:
    """
    Safe permutation application to model weights.
    
    This class handles the safe application of permutations to all relevant
    model weights, including proper dimension checking and error handling.
    """
    
    def __init__(self, model: PermutableModel):
        """
        Initialize the permutation applicator.
        
        Args:
            model: The permutable model to apply permutations to
        """
        self.model = model
        self.applied_permutations: Dict[str, np.ndarray] = {}
    
    def apply_permutation(
        self,
        layer_name: str,
        permutation: Union[np.ndarray, List[int], torch.Tensor],
        dimension: str = 'input',
        validate: bool = True,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Apply a permutation to a model layer with comprehensive safety checks.
        
        Args:
            layer_name: Name of the layer to permute
            permutation: Permutation indices
            dimension: Which dimension to permute ('input', 'output', 'both')
            validate: Whether to validate permutation before applying
            dry_run: If True, only validate without applying
            
        Returns:
            Dictionary with application results and metadata
        """
        # Convert permutation to numpy array
        if isinstance(permutation, torch.Tensor):
            perm = permutation.cpu().numpy()
        else:
            perm = np.array(permutation)
        
        # Get the layer
        try:
            layer = self.model.get_layer(layer_name)
        except ValueError as e:
            raise PermutationApplicationError(layer_name, str(e))
        
        # Get layer information
        layer_info = self.model.get_layer_info(layer_name)
        
        # Validate permutation
        if validate:
            validation_result = self._validate_permutation(layer, perm, dimension, layer_info)
            if not validation_result['valid']:
                raise PermutationApplicationError(layer_name, validation_result['error'])
        
        # Identify all affected layers
        affected_layers = self._identify_affected_layers(layer_name, layer_info)
        
        if dry_run:
            return {
                'layer_name': layer_name,
                'permutation_size': len(perm),
                'dimension': dimension,
                'affected_layers': affected_layers,
                'validation_result': validation_result if validate else None,
                'dry_run': True
            }
        
        # Apply permutation to all affected layers
        application_results = []
        
        for affected_layer_name in affected_layers:
            try:
                result = self._apply_to_single_layer(
                    affected_layer_name, perm, dimension, layer_info
                )
                application_results.append(result)
            except Exception as e:
                raise PermutationApplicationError(
                    affected_layer_name, f"Failed to apply permutation: {e}"
                )
        
        # Store applied permutation for tracking
        self.applied_permutations[layer_name] = perm
        
        return {
            'layer_name': layer_name,
            'permutation_size': len(perm),
            'dimension': dimension,
            'affected_layers': affected_layers,
            'application_results': application_results,
            'success': True
        }
    
    def _validate_permutation(
        self,
        layer: nn.Module,
        permutation: np.ndarray,
        dimension: str,
        layer_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate permutation against layer requirements.
        
        Args:
            layer: The layer to validate against
            permutation: Permutation to validate
            dimension: Which dimension to permute
            layer_info: Layer information
            
        Returns:
            Validation result dictionary
        """
        # Check if layer has weight
        if not hasattr(layer, 'weight') or layer.weight is None:
            return {
                'valid': False,
                'error': f"Layer has no weight to permute"
            }
        
        weight_shape = layer.weight.shape
        
        # Determine expected size based on dimension
        if dimension == 'input':
            expected_size = weight_shape[1] if len(weight_shape) > 1 else weight_shape[0]
        elif dimension == 'output':
            expected_size = weight_shape[0]
        elif dimension == 'both':
            if len(weight_shape) < 2 or weight_shape[0] != weight_shape[1]:
                return {
                    'valid': False,
                    'error': f"Cannot apply 'both' permutation to non-square weight matrix"
                }
            expected_size = weight_shape[0]
        else:
            return {
                'valid': False,
                'error': f"Invalid dimension: {dimension}"
            }
        
        # Check permutation size
        if len(permutation) != expected_size:
            return {
                'valid': False,
                'error': f"Permutation size {len(permutation)} doesn't match expected size {expected_size} "
                        f"for dimension '{dimension}' of layer with weight shape {weight_shape}"
            }
        
        # Check if permutation is valid (contains all indices exactly once)
        expected_indices = set(range(len(permutation)))
        actual_indices = set(permutation)
        
        if expected_indices != actual_indices:
            return {
                'valid': False,
                'error': "Invalid permutation: must contain each index exactly once"
            }
        
        # Check for out-of-bounds indices
        if np.any(permutation < 0) or np.any(permutation >= len(permutation)):
            return {
                'valid': False,
                'error': f"Permutation contains out-of-bounds indices"
            }
        
        return {
            'valid': True,
            'expected_size': expected_size,
            'weight_shape': weight_shape,
            'dimension': dimension
        }
    
    def _identify_affected_layers(
        self,
        layer_name: str,
        layer_info: Dict[str, Any]
    ) -> List[str]:
        """
        Identify all layers that need to be permuted together.
        
        For certain model architectures, permuting one layer requires
        permuting related layers to maintain consistency.
        
        Args:
            layer_name: Name of the primary layer
            layer_info: Layer information
            
        Returns:
            List of layer names that need to be permuted
        """
        affected_layers = [layer_name]
        
        # Model-specific logic for identifying affected layers
        if self.model.model_type == 'mamba':
            affected_layers.extend(self._identify_mamba_affected_layers(layer_name))
        elif self.model.model_type == 'bert':
            affected_layers.extend(self._identify_bert_affected_layers(layer_name))
        elif self.model.model_type == 'resnet':
            affected_layers.extend(self._identify_resnet_affected_layers(layer_name))
        elif self.model.model_type == 'gcn':
            affected_layers.extend(self._identify_gcn_affected_layers(layer_name))
        
        # Remove duplicates while preserving order
        unique_layers = []
        seen = set()
        for layer in affected_layers:
            if layer not in seen:
                unique_layers.append(layer)
                seen.add(layer)
        
        return unique_layers
    
    def _identify_mamba_affected_layers(self, layer_name: str) -> List[str]:
        """Identify affected layers for Mamba architecture."""
        affected = []
        
        # For Mamba blocks, certain layers are interconnected
        if 'mixer' in layer_name:
            # Extract base layer name (e.g., 'layers.0.mixer' -> 'layers.0')
            base_name = '.'.join(layer_name.split('.')[:-1])
            
            # Add related layers
            potential_layers = [
                f"{base_name}.in_proj",
                f"{base_name}.out_proj",
                f"{base_name}.conv1d",
                f"{base_name}.x_proj"
            ]
            
            # Check which layers actually exist
            for layer in potential_layers:
                try:
                    self.model.get_layer(layer)
                    affected.append(layer)
                except ValueError:
                    # Layer doesn't exist, skip
                    pass
        
        return affected
    
    def _identify_bert_affected_layers(self, layer_name: str) -> List[str]:
        """Identify affected layers for BERT architecture."""
        affected = []
        
        # For BERT attention layers
        if 'attention' in layer_name:
            base_name = '.'.join(layer_name.split('.')[:-1])
            
            potential_layers = [
                f"{base_name}.query",
                f"{base_name}.key",
                f"{base_name}.value",
                f"{base_name}.dense"
            ]
            
            for layer in potential_layers:
                try:
                    self.model.get_layer(layer)
                    affected.append(layer)
                except ValueError:
                    pass
        
        return affected
    
    def _identify_resnet_affected_layers(self, layer_name: str) -> List[str]:
        """Identify affected layers for ResNet architecture."""
        affected = []
        
        # For ResNet, usually only the specific layer is affected
        # unless it's part of a residual block
        if 'conv' in layer_name or 'bn' in layer_name:
            # Could add block-specific logic here
            pass
        
        return affected
    
    def _identify_gcn_affected_layers(self, layer_name: str) -> List[str]:
        """Identify affected layers for GCN architecture."""
        affected = []
        
        # For GCN, layers are usually independent
        # unless they're part of a multi-layer block
        
        return affected
    
    def _apply_to_single_layer(
        self,
        layer_name: str,
        permutation: np.ndarray,
        dimension: str,
        layer_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply permutation to a single layer.
        
        Args:
            layer_name: Name of the layer
            permutation: Permutation to apply
            dimension: Which dimension to permute
            layer_info: Layer information
            
        Returns:
            Application result dictionary
        """
        layer = self.model.get_layer(layer_name)
        layer_type = type(layer).__name__
        
        # Store original weight shape for verification
        original_shape = layer.weight.shape if hasattr(layer, 'weight') else None
        
        # Apply permutation based on layer type
        if isinstance(layer, nn.Linear):
            result = self._apply_linear_permutation(layer, permutation, dimension)
        elif isinstance(layer, nn.Conv1d):
            result = self._apply_conv1d_permutation(layer, permutation, dimension)
        elif isinstance(layer, nn.Conv2d):
            result = self._apply_conv2d_permutation(layer, permutation, dimension)
        else:
            result = self._apply_generic_permutation(layer, permutation, dimension)
        
        # Verify weight shape didn't change unexpectedly
        if original_shape is not None:
            new_shape = layer.weight.shape
            if new_shape != original_shape:
                warnings.warn(
                    f"Weight shape changed for layer {layer_name}: "
                    f"{original_shape} -> {new_shape}"
                )
        
        return {
            'layer_name': layer_name,
            'layer_type': layer_type,
            'dimension': dimension,
            'permutation_size': len(permutation),
            'original_shape': original_shape,
            'new_shape': layer.weight.shape if hasattr(layer, 'weight') else None,
            'success': result
        }
    
    def _apply_linear_permutation(
        self,
        layer: nn.Linear,
        permutation: np.ndarray,
        dimension: str
    ) -> bool:
        """Apply permutation to a linear layer."""
        with torch.no_grad():
            if dimension == 'input':
                # Permute input features: W_new = W[:, perm]
                layer.weight.data = layer.weight.data[:, permutation]
            elif dimension == 'output':
                # Permute output features: W_new = W[perm, :]
                layer.weight.data = layer.weight.data[permutation, :]
                if layer.bias is not None:
                    layer.bias.data = layer.bias.data[permutation]
            elif dimension == 'both':
                # Permute both dimensions: W_new = W[perm, :][:, perm]
                layer.weight.data = layer.weight.data[permutation, :][:, permutation]
                if layer.bias is not None:
                    layer.bias.data = layer.bias.data[permutation]
        
        return True
    
    def _apply_conv1d_permutation(
        self,
        layer: nn.Conv1d,
        permutation: np.ndarray,
        dimension: str
    ) -> bool:
        """Apply permutation to a 1D convolutional layer."""
        with torch.no_grad():
            if dimension == 'input':
                # Permute input channels: W_new = W[:, perm, :]
                layer.weight.data = layer.weight.data[:, permutation, :]
            elif dimension == 'output':
                # Permute output channels: W_new = W[perm, :, :]
                layer.weight.data = layer.weight.data[permutation, :, :]
                if layer.bias is not None:
                    layer.bias.data = layer.bias.data[permutation]
            elif dimension == 'both':
                # For conv1d, 'both' usually means permuting output channels
                layer.weight.data = layer.weight.data[permutation, :, :]
                if layer.bias is not None:
                    layer.bias.data = layer.bias.data[permutation]
        
        return True
    
    def _apply_conv2d_permutation(
        self,
        layer: nn.Conv2d,
        permutation: np.ndarray,
        dimension: str
    ) -> bool:
        """Apply permutation to a 2D convolutional layer."""
        with torch.no_grad():
            if dimension == 'input':
                # Permute input channels: W_new = W[:, perm, :, :]
                layer.weight.data = layer.weight.data[:, permutation, :, :]
            elif dimension == 'output':
                # Permute output channels: W_new = W[perm, :, :, :]
                layer.weight.data = layer.weight.data[permutation, :, :, :]
                if layer.bias is not None:
                    layer.bias.data = layer.bias.data[permutation]
            elif dimension == 'both':
                # For conv2d, 'both' usually means permuting output channels
                layer.weight.data = layer.weight.data[permutation, :, :, :]
                if layer.bias is not None:
                    layer.bias.data = layer.bias.data[permutation]
        
        return True
    
    def _apply_generic_permutation(
        self,
        layer: nn.Module,
        permutation: np.ndarray,
        dimension: str
    ) -> bool:
        """Apply permutation to a generic layer (fallback)."""
        if not hasattr(layer, 'weight') or layer.weight is None:
            return False
        
        with torch.no_grad():
            if dimension == 'input' and len(layer.weight.shape) > 1:
                layer.weight.data = layer.weight.data[:, permutation]
            elif dimension == 'output':
                layer.weight.data = layer.weight.data[permutation]
            elif dimension == 'both' and len(layer.weight.shape) > 1:
                layer.weight.data = layer.weight.data[permutation, :][:, permutation]
            
            if hasattr(layer, 'bias') and layer.bias is not None and dimension != 'input':
                layer.bias.data = layer.bias.data[permutation]
        
        return True
    
    def get_applied_permutations(self) -> Dict[str, np.ndarray]:
        """Get all applied permutations."""
        return self.applied_permutations.copy()
    
    def has_permutation(self, layer_name: str) -> bool:
        """Check if a layer has had a permutation applied."""
        return layer_name in self.applied_permutations
    
    def reset_permutations(self):
        """Reset all applied permutations tracking."""
        self.applied_permutations.clear()
    
    def get_permutation_summary(self) -> Dict[str, Any]:
        """Get summary of all applied permutations."""
        return {
            'num_permuted_layers': len(self.applied_permutations),
            'permuted_layers': list(self.applied_permutations.keys()),
            'total_permutations': sum(len(perm) for perm in self.applied_permutations.values())
        }