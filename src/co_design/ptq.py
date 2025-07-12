"""
Post-Training Quantization (PTQ) implementation.

This module implements uniform symmetric INT8 quantization for neural networks,
as described in the paper for the iterative co-design framework.
"""
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from ..models.permutable_model import PermutableModel
from ..utils.exceptions import IterativeCoDesignError
from ..utils.config import BaseConfig


@dataclass
class PTQConfig(BaseConfig):
    """Configuration for Post-Training Quantization."""
    
    # Quantization settings
    weight_bits: int = 8  # Number of bits for weights
    activation_bits: int = 8  # Number of bits for activations
    
    # Calibration settings
    calibration_samples: int = 1000  # Number of samples for calibration
    calibration_batch_size: int = 32
    
    # Quantization mode
    symmetric: bool = True  # Use symmetric quantization
    per_channel: bool = True  # Per-channel quantization for weights
    
    # Range estimation
    percentile: float = 99.99  # Percentile for range estimation
    moving_average_momentum: float = 0.9  # For running statistics
    
    # Layer selection
    target_layers: List[str] = None  # If None, quantize all Linear/Conv layers
    exclude_layers: List[str] = None  # Layers to exclude from quantization
    
    # Advanced settings
    use_kl_divergence: bool = False  # Use KL divergence for optimal scaling
    folding_bn: bool = True  # Fold batch normalization into convolution
    
    # Logging
    verbose: bool = True


class QuantizationObserver:
    """
    Observer to collect statistics for quantization calibration.
    """
    
    def __init__(
        self,
        dtype: torch.dtype = torch.qint8,
        qscheme: torch.qscheme = torch.per_tensor_symmetric,
        reduce_range: bool = False,
        quant_min: Optional[int] = None,
        quant_max: Optional[int] = None
    ):
        """
        Initialize quantization observer.
        
        Args:
            dtype: Quantized data type
            qscheme: Quantization scheme
            reduce_range: Whether to reduce quantization range
            quant_min: Minimum quantization value
            quant_max: Maximum quantization value
        """
        self.dtype = dtype
        self.qscheme = qscheme
        self.reduce_range = reduce_range
        
        # Set quantization range
        if quant_min is None or quant_max is None:
            if dtype == torch.qint8:
                self.quant_min = -128 if not reduce_range else -127
                self.quant_max = 127
            elif dtype == torch.quint8:
                self.quant_min = 0
                self.quant_max = 255 if not reduce_range else 127
            else:
                raise ValueError(f"Unsupported dtype: {dtype}")
        else:
            self.quant_min = quant_min
            self.quant_max = quant_max
        
        # Statistics
        self.min_val = float('inf')
        self.max_val = float('-inf')
        self.num_samples = 0
        
        # For moving average
        self.running_min = None
        self.running_max = None
        self.momentum = 0.9
    
    def update(self, tensor: torch.Tensor):
        """Update statistics with new tensor."""
        if tensor.numel() == 0:
            return
        
        # Detach and move to CPU if needed
        tensor = tensor.detach()
        if tensor.device != torch.device('cpu'):
            tensor = tensor.cpu()
        
        # Update min/max
        batch_min = tensor.min().item()
        batch_max = tensor.max().item()
        
        if self.running_min is None:
            self.running_min = batch_min
            self.running_max = batch_max
        else:
            self.running_min = self.momentum * self.running_min + (1 - self.momentum) * batch_min
            self.running_max = self.momentum * self.running_max + (1 - self.momentum) * batch_max
        
        self.min_val = min(self.min_val, batch_min)
        self.max_val = max(self.max_val, batch_max)
        self.num_samples += tensor.numel()
    
    def calculate_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate quantization parameters (scale and zero_point)."""
        if self.num_samples == 0:
            warnings.warn("No samples observed for quantization calibration")
            return torch.tensor(1.0), torch.tensor(0)
        
        # Use running statistics for more stable estimates
        min_val = self.running_min if self.running_min is not None else self.min_val
        max_val = self.running_max if self.running_max is not None else self.max_val
        
        # Ensure min_val != max_val
        if abs(max_val - min_val) < 1e-8:
            max_val = min_val + 1e-8
        
        if self.qscheme == torch.per_tensor_symmetric:
            # Symmetric quantization
            max_range = max(abs(min_val), abs(max_val))
            scale = max_range / ((self.quant_max - self.quant_min) / 2)
            zero_point = 0
        else:
            # Asymmetric quantization
            scale = (max_val - min_val) / (self.quant_max - self.quant_min)
            zero_point = self.quant_min - min_val / scale
            zero_point = max(self.quant_min, min(self.quant_max, int(round(zero_point))))
        
        return torch.tensor(scale, dtype=torch.float32), torch.tensor(zero_point, dtype=torch.int32)


class QuantizedLinear(nn.Module):
    """
    Quantized Linear layer.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        weight_scale: torch.Tensor = None,
        weight_zero_point: torch.Tensor = None,
        activation_scale: torch.Tensor = None,
        activation_zero_point: torch.Tensor = None
    ):
        """
        Initialize quantized linear layer.
        
        Args:
            in_features: Input features
            out_features: Output features
            bias: Whether to use bias
            weight_scale: Weight quantization scale
            weight_zero_point: Weight quantization zero point
            activation_scale: Activation quantization scale
            activation_zero_point: Activation quantization zero point
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Quantized weight
        self.weight_scale = weight_scale if weight_scale is not None else torch.tensor(1.0)
        self.weight_zero_point = weight_zero_point if weight_zero_point is not None else torch.tensor(0)
        
        # Quantized activations
        self.activation_scale = activation_scale if activation_scale is not None else torch.tensor(1.0)
        self.activation_zero_point = activation_zero_point if activation_zero_point is not None else torch.tensor(0)
        
        # Register buffers
        self.register_buffer('_weight_scale', self.weight_scale)
        self.register_buffer('_weight_zero_point', self.weight_zero_point)
        self.register_buffer('_activation_scale', self.activation_scale)
        self.register_buffer('_activation_zero_point', self.activation_zero_point)
        
        # Quantized weight storage
        self.register_buffer('_packed_weight', torch.zeros(out_features, in_features, dtype=torch.qint8))
        
        if bias:
            self.register_buffer('bias', torch.zeros(out_features))
        else:
            self.register_buffer('bias', None)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantized operations."""
        # For now, simulate quantized operations
        # In practice, this would use optimized quantized kernels
        
        # Quantize input
        input_quantized = torch.quantize_per_tensor(
            input, 
            self._activation_scale.item(),
            self._activation_zero_point.item(),
            torch.qint8
        )
        
        # Dequantize for computation (simulation)
        input_dequantized = input_quantized.dequantize()
        weight_dequantized = self._packed_weight.dequantize()
        
        # Perform linear operation
        output = F.linear(input_dequantized, weight_dequantized, self.bias)
        
        return output
    
    def set_weight(self, weight: torch.Tensor):
        """Set quantized weight."""
        weight_quantized = torch.quantize_per_tensor(
            weight,
            self._weight_scale.item(),
            self._weight_zero_point.item(),
            torch.qint8
        )
        self._packed_weight.copy_(weight_quantized)
    
    @classmethod
    def from_float(cls, float_linear: nn.Linear, weight_observer: QuantizationObserver, activation_observer: QuantizationObserver):
        """Create quantized linear layer from float linear layer."""
        # Calculate quantization parameters
        weight_scale, weight_zero_point = weight_observer.calculate_qparams()
        activation_scale, activation_zero_point = activation_observer.calculate_qparams()
        
        # Create quantized layer
        quantized_linear = cls(
            in_features=float_linear.in_features,
            out_features=float_linear.out_features,
            bias=float_linear.bias is not None,
            weight_scale=weight_scale,
            weight_zero_point=weight_zero_point,
            activation_scale=activation_scale,
            activation_zero_point=activation_zero_point
        )
        
        # Set quantized weight
        quantized_linear.set_weight(float_linear.weight)
        
        # Set bias
        if float_linear.bias is not None:
            quantized_linear.bias.copy_(float_linear.bias)
        
        return quantized_linear


class PostTrainingQuantizer:
    """
    Post-Training Quantization implementation.
    """
    
    def __init__(self, config: PTQConfig):
        """
        Initialize PTQ.
        
        Args:
            config: PTQ configuration
        """
        self.config = config
        self.observers = {}
        self.quantized_layers = {}
        
        # Statistics collection
        self.calibration_stats = {}
    
    def prepare_model(self, model: PermutableModel) -> PermutableModel:
        """
        Prepare model for quantization by inserting observers.
        
        Args:
            model: The model to prepare
            
        Returns:
            Model with observers
        """
        # Get target layers
        target_layers = self._get_target_layers(model)
        
        # Insert observers for each target layer
        for layer_name in target_layers:
            try:
                layer = model.get_layer(layer_name)
                
                if isinstance(layer, nn.Linear):
                    # Create observers for weights and activations
                    weight_observer = QuantizationObserver(
                        dtype=torch.qint8,
                        qscheme=torch.per_tensor_symmetric
                    )
                    activation_observer = QuantizationObserver(
                        dtype=torch.qint8,
                        qscheme=torch.per_tensor_symmetric
                    )
                    
                    # Observe weights immediately
                    weight_observer.update(layer.weight)
                    
                    self.observers[layer_name] = {
                        'weight': weight_observer,
                        'activation': activation_observer,
                        'layer': layer
                    }
                    
                    # Hook to observe activations
                    def make_activation_hook(obs):
                        def hook(module, input, output):
                            obs.update(output)
                        return hook
                    
                    layer.register_forward_hook(make_activation_hook(activation_observer))
                    
            except Exception as e:
                warnings.warn(f"Failed to prepare layer {layer_name}: {e}")
        
        if self.config.verbose:
            print(f"Prepared {len(self.observers)} layers for quantization")
        
        return model
    
    def _get_target_layers(self, model: PermutableModel) -> List[str]:
        """Get list of layers to quantize."""
        if self.config.target_layers:
            return self.config.target_layers
        
        # Default: all Linear and Conv layers
        target_layers = []
        for name, layer in model.model.named_modules():
            if isinstance(layer, (nn.Linear, nn.Conv1d, nn.Conv2d)) and name:
                # Check if layer should be excluded
                if self.config.exclude_layers and name in self.config.exclude_layers:
                    continue
                target_layers.append(name)
        
        return target_layers
    
    def calibrate(
        self,
        model: PermutableModel,
        calibration_dataloader: torch.utils.data.DataLoader,
        device: str = 'cuda'
    ):
        """
        Calibrate quantization parameters using calibration data.
        
        Args:
            model: The model to calibrate
            calibration_dataloader: Calibration data loader
            device: Device to use
        """
        if self.config.verbose:
            print("Starting quantization calibration...")
        
        model.to(device)
        model.eval()
        
        num_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(calibration_dataloader, desc="Calibrating"):
                if num_samples >= self.config.calibration_samples:
                    break
                
                # Move batch to device
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(device)
                else:
                    inputs = batch.to(device)
                
                # Forward pass to collect statistics
                try:
                    _ = model(inputs)
                    num_samples += inputs.size(0)
                except Exception as e:
                    warnings.warn(f"Calibration forward pass failed: {e}")
                    continue
        
        if self.config.verbose:
            print(f"Calibration completed with {num_samples} samples")
        
        # Calculate and store quantization parameters
        self._calculate_quantization_parameters()
    
    def _calculate_quantization_parameters(self):
        """Calculate quantization parameters from collected statistics."""
        for layer_name, observer_dict in self.observers.items():
            weight_observer = observer_dict['weight']
            activation_observer = observer_dict['activation']
            
            # Calculate quantization parameters
            weight_scale, weight_zero_point = weight_observer.calculate_qparams()
            activation_scale, activation_zero_point = activation_observer.calculate_qparams()
            
            # Store calibration stats
            self.calibration_stats[layer_name] = {
                'weight_scale': weight_scale.item(),
                'weight_zero_point': weight_zero_point.item(),
                'activation_scale': activation_scale.item(),
                'activation_zero_point': activation_zero_point.item(),
                'weight_range': (weight_observer.min_val, weight_observer.max_val),
                'activation_range': (activation_observer.min_val, activation_observer.max_val)
            }
    
    def quantize_model(self, model: PermutableModel) -> PermutableModel:
        """
        Convert model to quantized version.
        
        Args:
            model: The model to quantize
            
        Returns:
            Quantized model
        """
        if self.config.verbose:
            print("Converting model to quantized version...")
        
        # Convert each observed layer
        for layer_name, observer_dict in self.observers.items():
            try:
                layer = observer_dict['layer']
                weight_observer = observer_dict['weight']
                activation_observer = observer_dict['activation']
                
                if isinstance(layer, nn.Linear):
                    # Create quantized layer
                    quantized_layer = QuantizedLinear.from_float(
                        layer, weight_observer, activation_observer
                    )
                    
                    # Replace layer in model
                    self._replace_layer(model, layer_name, quantized_layer)
                    self.quantized_layers[layer_name] = quantized_layer
                    
            except Exception as e:
                warnings.warn(f"Failed to quantize layer {layer_name}: {e}")
        
        if self.config.verbose:
            print(f"Quantized {len(self.quantized_layers)} layers")
        
        return model
    
    def _replace_layer(self, model: PermutableModel, layer_name: str, new_layer: nn.Module):
        """Replace a layer in the model."""
        # Navigate to the parent module
        parts = layer_name.split('.')
        current = model.model
        
        for part in parts[:-1]:
            current = getattr(current, part)
        
        # Replace the final layer
        setattr(current, parts[-1], new_layer)
    
    def get_quantization_stats(self) -> Dict[str, Any]:
        """Get quantization statistics."""
        stats = {
            'calibration_stats': self.calibration_stats,
            'num_quantized_layers': len(self.quantized_layers),
            'quantized_layers': list(self.quantized_layers.keys())
        }
        
        # Calculate model size reduction
        total_params = 0
        quantized_params = 0
        
        for layer_name, layer in self.quantized_layers.items():
            if hasattr(layer, 'weight'):
                layer_params = layer.weight.numel()
                total_params += layer_params
                quantized_params += layer_params
        
        if total_params > 0:
            # Estimate size reduction (float32 to int8)
            size_reduction = 1.0 - (quantized_params * 1) / (total_params * 4)  # 8-bit vs 32-bit
            stats['estimated_size_reduction'] = size_reduction
        
        return stats


# Utility functions
def quantize_model(
    model: PermutableModel,
    calibration_dataloader: torch.utils.data.DataLoader,
    config: Optional[PTQConfig] = None,
    device: str = 'cuda'
) -> Tuple[PermutableModel, Dict[str, Any]]:
    """
    Apply post-training quantization to a model.
    
    Args:
        model: The model to quantize
        calibration_dataloader: Calibration data loader
        config: PTQ configuration
        device: Device to use
        
    Returns:
        Tuple of (quantized_model, quantization_stats)
    """
    if config is None:
        config = PTQConfig()
    
    # Create quantizer
    quantizer = PostTrainingQuantizer(config)
    
    # Prepare model
    model = quantizer.prepare_model(model)
    
    # Calibrate
    quantizer.calibrate(model, calibration_dataloader, device)
    
    # Quantize
    quantized_model = quantizer.quantize_model(model)
    
    # Get statistics
    stats = quantizer.get_quantization_stats()
    
    return quantized_model, stats


def validate_quantization(
    original_model: PermutableModel,
    quantized_model: PermutableModel,
    test_dataloader: torch.utils.data.DataLoader,
    device: str = 'cuda'
) -> Dict[str, Any]:
    """
    Validate quantization by comparing original and quantized models.
    
    Args:
        original_model: Original float model
        quantized_model: Quantized model
        test_dataloader: Test data loader
        device: Device to use
        
    Returns:
        Validation results
    """
    results = {
        'mse_loss': 0.0,
        'max_error': 0.0,
        'correlation': 0.0,
        'num_samples': 0
    }
    
    original_model.to(device)
    quantized_model.to(device)
    
    original_model.eval()
    quantized_model.eval()
    
    all_original_outputs = []
    all_quantized_outputs = []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Validating"):
            if isinstance(batch, (list, tuple)):
                inputs = batch[0].to(device)
            else:
                inputs = batch.to(device)
            
            try:
                # Get outputs from both models
                original_output = original_model(inputs)
                quantized_output = quantized_model(inputs)
                
                # Flatten outputs for comparison
                original_flat = original_output.view(-1)
                quantized_flat = quantized_output.view(-1)
                
                all_original_outputs.append(original_flat.cpu())
                all_quantized_outputs.append(quantized_flat.cpu())
                
                # Update MSE
                mse = F.mse_loss(quantized_flat, original_flat).item()
                results['mse_loss'] += mse
                
                # Update max error
                max_error = torch.max(torch.abs(quantized_flat - original_flat)).item()
                results['max_error'] = max(results['max_error'], max_error)
                
                results['num_samples'] += inputs.size(0)
                
            except Exception as e:
                warnings.warn(f"Validation failed for batch: {e}")
                continue
    
    # Calculate final metrics
    if results['num_samples'] > 0:
        results['mse_loss'] /= len(test_dataloader)
        
        # Calculate correlation
        if all_original_outputs and all_quantized_outputs:
            original_concat = torch.cat(all_original_outputs)
            quantized_concat = torch.cat(all_quantized_outputs)
            
            correlation = torch.corrcoef(torch.stack([original_concat, quantized_concat]))[0, 1]
            results['correlation'] = correlation.item()
    
    return results