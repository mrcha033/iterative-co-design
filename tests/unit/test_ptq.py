"""
Unit tests for Post-Training Quantization (PTQ) implementation.
"""
import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch

from src.co_design.ptq import (
    PTQConfig, QuantizationObserver, QuantizedLinear, PostTrainingQuantizer,
    quantize_model, validate_quantization
)
from src.models.permutable_model import PermutableModel


class TestPTQConfig:
    """Test PTQConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = PTQConfig()
        
        assert config.weight_bits == 8
        assert config.activation_bits == 8
        assert config.calibration_samples == 1000
        assert config.symmetric is True
        assert config.per_channel is True
        assert config.percentile == 99.99
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = PTQConfig(
            weight_bits=4,
            activation_bits=8,
            calibration_samples=500,
            symmetric=False,
            per_channel=False
        )
        
        assert config.weight_bits == 4
        assert config.activation_bits == 8
        assert config.calibration_samples == 500
        assert config.symmetric is False
        assert config.per_channel is False


class TestQuantizationObserver:
    """Test QuantizationObserver class."""
    
    def test_initialization(self):
        """Test QuantizationObserver initialization."""
        observer = QuantizationObserver(
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric
        )
        
        assert observer.dtype == torch.qint8
        assert observer.qscheme == torch.per_tensor_symmetric
        assert observer.quant_min == -128
        assert observer.quant_max == 127
        assert observer.num_samples == 0
    
    def test_update_statistics(self):
        """Test statistics update."""
        observer = QuantizationObserver()
        
        # Test with sample tensors
        tensor1 = torch.randn(4, 8)
        tensor2 = torch.randn(2, 8)
        
        observer.update(tensor1)
        observer.update(tensor2)
        
        assert observer.num_samples == tensor1.numel() + tensor2.numel()
        assert observer.min_val < observer.max_val
        assert observer.running_min is not None
        assert observer.running_max is not None
    
    def test_calculate_qparams_symmetric(self):
        """Test quantization parameter calculation for symmetric quantization."""
        observer = QuantizationObserver(qscheme=torch.per_tensor_symmetric)
        
        # Update with known values
        tensor = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        observer.update(tensor)
        
        scale, zero_point = observer.calculate_qparams()
        
        assert scale.item() > 0
        assert zero_point.item() == 0  # Symmetric quantization
    
    def test_calculate_qparams_asymmetric(self):
        """Test quantization parameter calculation for asymmetric quantization."""
        observer = QuantizationObserver(qscheme=torch.per_tensor_affine)
        
        # Update with known values
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        observer.update(tensor)
        
        scale, zero_point = observer.calculate_qparams()
        
        assert scale.item() > 0
        assert isinstance(zero_point.item(), int)
    
    def test_empty_observer(self):
        """Test observer with no samples."""
        observer = QuantizationObserver()
        
        with pytest.warns(UserWarning):
            scale, zero_point = observer.calculate_qparams()
            
        assert scale.item() == 1.0
        assert zero_point.item() == 0


class TestQuantizedLinear:
    """Test QuantizedLinear class."""
    
    def test_initialization(self):
        """Test QuantizedLinear initialization."""
        in_features = 4
        out_features = 2
        
        quantized_linear = QuantizedLinear(
            in_features=in_features,
            out_features=out_features,
            bias=True,
            weight_scale=torch.tensor(0.1),
            weight_zero_point=torch.tensor(0),
            activation_scale=torch.tensor(0.1),
            activation_zero_point=torch.tensor(0)
        )
        
        assert quantized_linear.in_features == in_features
        assert quantized_linear.out_features == out_features
        assert quantized_linear.bias is not None
    
    def test_forward(self):
        """Test forward pass."""
        quantized_linear = QuantizedLinear(
            in_features=4,
            out_features=2,
            bias=True
        )
        
        input_tensor = torch.randn(2, 4)
        output = quantized_linear(input_tensor)
        
        assert output.shape == (2, 2)
    
    def test_from_float(self):
        """Test creating quantized layer from float layer."""
        # Create float layer
        float_linear = nn.Linear(4, 2)
        
        # Create observers
        weight_observer = QuantizationObserver()
        activation_observer = QuantizationObserver()
        
        # Update observers
        weight_observer.update(float_linear.weight)
        activation_observer.update(torch.randn(2, 4))
        
        # Create quantized layer
        quantized_linear = QuantizedLinear.from_float(
            float_linear, weight_observer, activation_observer
        )
        
        assert quantized_linear.in_features == float_linear.in_features
        assert quantized_linear.out_features == float_linear.out_features
        assert quantized_linear.bias is not None
    
    def test_set_weight(self):
        """Test setting quantized weight."""
        quantized_linear = QuantizedLinear(4, 2)
        
        new_weight = torch.randn(2, 4)
        quantized_linear.set_weight(new_weight)
        
        # Should complete without error
        assert True


class TestPostTrainingQuantizer:
    """Test PostTrainingQuantizer class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = PTQConfig(calibration_samples=10, verbose=False)
        self.quantizer = PostTrainingQuantizer(self.config)
        
        # Create simple test model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(4, 2)
                self.linear2 = nn.Linear(2, 1)
            
            def forward(self, x):
                return self.linear2(torch.relu(self.linear1(x)))
        
        self.model = SimpleModel()
        self.permutable_model = PermutableModel(self.model, 'test', 'test')
    
    def test_initialization(self):
        """Test PostTrainingQuantizer initialization."""
        assert self.quantizer.config == self.config
        assert len(self.quantizer.observers) == 0
        assert len(self.quantizer.quantized_layers) == 0
    
    def test_get_target_layers(self):
        """Test target layer identification."""
        target_layers = self.quantizer._get_target_layers(self.permutable_model)
        
        assert 'linear1' in target_layers
        assert 'linear2' in target_layers
        assert len(target_layers) == 2
    
    def test_prepare_model(self):
        """Test model preparation."""
        prepared_model = self.quantizer.prepare_model(self.permutable_model)
        
        assert len(self.quantizer.observers) == 2
        assert 'linear1' in self.quantizer.observers
        assert 'linear2' in self.quantizer.observers
        assert prepared_model == self.permutable_model
    
    def test_calibrate(self):
        """Test calibration process."""
        # Prepare model first
        self.quantizer.prepare_model(self.permutable_model)
        
        # Create mock dataloader
        dataloader = Mock()
        dataloader.__iter__ = Mock(return_value=iter([
            torch.randn(2, 4) for _ in range(3)
        ]))
        
        # Run calibration
        self.quantizer.calibrate(self.permutable_model, dataloader)
        
        # Check that calibration stats were calculated
        assert len(self.quantizer.calibration_stats) == 2
        assert 'linear1' in self.quantizer.calibration_stats
        assert 'linear2' in self.quantizer.calibration_stats
        
        # Check stats structure
        for stats in self.quantizer.calibration_stats.values():
            assert 'weight_scale' in stats
            assert 'weight_zero_point' in stats
            assert 'activation_scale' in stats
            assert 'activation_zero_point' in stats
    
    def test_quantize_model(self):
        """Test model quantization."""
        # Prepare and calibrate model
        self.quantizer.prepare_model(self.permutable_model)
        
        dataloader = Mock()
        dataloader.__iter__ = Mock(return_value=iter([
            torch.randn(2, 4) for _ in range(3)
        ]))
        
        self.quantizer.calibrate(self.permutable_model, dataloader)
        
        # Quantize model
        quantized_model = self.quantizer.quantize_model(self.permutable_model)
        
        assert len(self.quantizer.quantized_layers) == 2
        assert 'linear1' in self.quantizer.quantized_layers
        assert 'linear2' in self.quantizer.quantized_layers
        assert quantized_model == self.permutable_model
    
    def test_get_quantization_stats(self):
        """Test quantization statistics."""
        # Prepare model
        self.quantizer.prepare_model(self.permutable_model)
        
        # Create some fake quantized layers
        self.quantizer.quantized_layers['linear1'] = Mock()
        self.quantizer.quantized_layers['linear2'] = Mock()
        
        stats = self.quantizer.get_quantization_stats()
        
        assert 'calibration_stats' in stats
        assert 'num_quantized_layers' in stats
        assert 'quantized_layers' in stats
        assert stats['num_quantized_layers'] == 2


class TestUtilityFunctions:
    """Test utility functions."""
    
    def setup_method(self):
        """Setup test fixtures."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 2)
            
            def forward(self, x):
                return self.linear(x)
        
        self.model = SimpleModel()
        self.permutable_model = PermutableModel(self.model, 'test', 'test')
        
        # Create mock dataloader
        self.dataloader = Mock()
        self.dataloader.__iter__ = Mock(return_value=iter([
            torch.randn(2, 4) for _ in range(3)
        ]))
    
    def test_quantize_model_function(self):
        """Test quantize_model utility function."""
        config = PTQConfig(calibration_samples=10, verbose=False)
        
        quantized_model, stats = quantize_model(
            self.permutable_model,
            self.dataloader,
            config
        )
        
        assert quantized_model is not None
        assert 'calibration_stats' in stats
        assert 'num_quantized_layers' in stats
        assert 'quantized_layers' in stats
    
    def test_validate_quantization(self):
        """Test quantization validation."""
        # Create original and quantized models
        original_model = self.permutable_model
        
        # For simplicity, use the same model as "quantized"
        quantized_model = self.permutable_model
        
        # Create mock test dataloader
        test_dataloader = Mock()
        test_dataloader.__iter__ = Mock(return_value=iter([
            torch.randn(2, 4) for _ in range(2)
        ]))
        
        results = validate_quantization(
            original_model,
            quantized_model,
            test_dataloader
        )
        
        assert 'mse_loss' in results
        assert 'max_error' in results
        assert 'correlation' in results
        assert 'num_samples' in results
        assert results['num_samples'] >= 0


class TestIntegration:
    """Integration tests for PTQ module."""
    
    def test_end_to_end_ptq_workflow(self):
        """Test complete PTQ workflow."""
        # Create simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 2)
            
            def forward(self, x):
                return self.linear(x)
        
        model = SimpleModel()
        permutable_model = PermutableModel(model, 'test', 'test')
        
        # Create mock dataloader
        dataloader = Mock()
        dataloader.__iter__ = Mock(return_value=iter([
            torch.randn(2, 4) for _ in range(3)
        ]))
        
        # Create PTQ quantizer
        config = PTQConfig(calibration_samples=10, verbose=False)
        quantizer = PostTrainingQuantizer(config)
        
        # Prepare model
        prepared_model = quantizer.prepare_model(permutable_model)
        
        # Check that observers were created
        assert len(quantizer.observers) == 1
        assert 'linear' in quantizer.observers
        
        # Calibrate
        quantizer.calibrate(permutable_model, dataloader)
        
        # Check calibration stats
        assert len(quantizer.calibration_stats) == 1
        assert 'linear' in quantizer.calibration_stats
        
        # Quantize model
        quantized_model = quantizer.quantize_model(permutable_model)
        
        # Check quantized layers
        assert len(quantizer.quantized_layers) == 1
        assert 'linear' in quantizer.quantized_layers
        
        # Get stats
        stats = quantizer.get_quantization_stats()
        assert stats['num_quantized_layers'] == 1
        
        # The test should complete without errors
        assert True
    
    def test_observer_statistics_collection(self):
        """Test that observers correctly collect statistics."""
        observer = QuantizationObserver()
        
        # Test with multiple updates
        tensors = [
            torch.randn(2, 4),
            torch.randn(3, 4),
            torch.randn(1, 4)
        ]
        
        for tensor in tensors:
            observer.update(tensor)
        
        # Check statistics
        assert observer.num_samples == sum(t.numel() for t in tensors)
        assert observer.running_min is not None
        assert observer.running_max is not None
        
        # Calculate quantization parameters
        scale, zero_point = observer.calculate_qparams()
        
        assert scale.item() > 0
        assert isinstance(zero_point.item(), int)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])