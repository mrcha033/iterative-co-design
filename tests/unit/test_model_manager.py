"""
Unit tests for ModelManager class.
"""
import pytest
import tempfile
import torch
import torch.nn as nn
from pathlib import Path

from src.models.manager import ModelManager
from src.models.permutable_model import PermutableModel
from src.utils.exceptions import ModelNotSupportedError


class MockModel(nn.Module):
    """Mock model for testing."""
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=64):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class TestModelManager:
    """Test cases for ModelManager."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = ModelManager(cache_dir=self.temp_dir)
    
    def test_initialization(self):
        """Test ModelManager initialization."""
        assert self.manager.cache_dir == Path(self.temp_dir)
        assert len(self.manager._loaded_models) == 0
    
    def test_supported_models(self):
        """Test supported model queries."""
        # Test valid models
        assert self.manager.is_model_supported('mamba-3b')
        assert self.manager.is_model_supported('bert-large')
        assert self.manager.is_model_supported('resnet-50')
        assert self.manager.is_model_supported('gcn')
        
        # Test invalid model
        assert not self.manager.is_model_supported('invalid-model')
        
        # Test case insensitive
        assert self.manager.is_model_supported('MAMBA-3B')
    
    def test_list_supported_models(self):
        """Test listing supported models."""
        models = self.manager.list_supported_models()
        expected_models = ['mamba-3b', 'bert-large', 'resnet-50', 'gcn']
        assert set(models) == set(expected_models)
    
    def test_get_model_info(self):
        """Test getting model information."""
        info = self.manager.get_model_info('mamba-3b')
        assert info['type'] == 'mamba'
        assert info['requires_transformers'] is True
        
        with pytest.raises(ValueError):
            self.manager.get_model_info('invalid-model')
    
    def test_model_loading_unsupported(self):
        """Test loading unsupported model."""
        with pytest.raises(ValueError, match="Model invalid-model not supported"):
            self.manager.load_model('invalid-model')
    
    def test_cache_management(self):
        """Test cache management."""
        # Test cache info
        info = self.manager.get_cache_info()
        assert 'cache_dir' in info
        assert 'loaded_models' in info
        assert 'cache_size_mb' in info
        
        # Test cache clearing
        self.manager.clear_cache()
        assert len(self.manager._loaded_models) == 0
    
    def test_layer_validation(self):
        """Test layer name validation."""
        # This test would require a real model, so we'll test the interface
        assert not self.manager.validate_layer_name('mamba-3b', 'nonexistent.layer')
    
    def test_precision_setting(self):
        """Test precision setting."""
        model = MockModel()
        
        # Test float32
        model_fp32 = self.manager._set_precision(model, 'float32')
        assert next(model_fp32.parameters()).dtype == torch.float32
        
        # Test float16
        model_fp16 = self.manager._set_precision(model, 'float16')
        assert next(model_fp16.parameters()).dtype == torch.float16
        
        # Test invalid precision
        with pytest.raises(ValueError, match="Unsupported precision"):
            self.manager._set_precision(model, 'invalid')


class TestPermutableModel:
    """Test cases for PermutableModel."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.mock_model = MockModel()
        self.permutable_model = PermutableModel(
            model=self.mock_model,
            model_type='test',
            model_name='test-model'
        )
    
    def test_initialization(self):
        """Test PermutableModel initialization."""
        assert self.permutable_model.model == self.mock_model
        assert self.permutable_model.model_type == 'test'
        assert self.permutable_model.model_name == 'test-model'
        assert len(self.permutable_model._applied_permutations) == 0
    
    def test_forward_pass(self):
        """Test forward pass."""
        x = torch.randn(1, 128)
        output = self.permutable_model(x)
        assert output.shape == (1, 64)
    
    def test_layer_access(self):
        """Test layer access."""
        # Test valid layer
        layer = self.permutable_model.get_layer('linear1')
        assert isinstance(layer, nn.Linear)
        assert layer.in_features == 128
        assert layer.out_features == 256
        
        # Test invalid layer
        with pytest.raises(ValueError, match="Layer 'nonexistent' not found"):
            self.permutable_model.get_layer('nonexistent')
    
    def test_layer_info(self):
        """Test layer information."""
        info = self.permutable_model.get_layer_info('linear1')
        assert info['name'] == 'linear1'
        assert info['type'] == 'Linear'
        assert info['has_weight'] is True
        assert info['has_bias'] is True
        assert info['weight_shape'] == [256, 128]
    
    def test_permutation_validation(self):
        """Test permutation validation."""
        layer = self.permutable_model.get_layer('linear1')
        
        # Valid permutation
        valid_perm = torch.randperm(128).numpy()
        self.permutable_model._validate_permutation(layer, valid_perm, 'input')
        
        # Invalid size
        with pytest.raises(ValueError, match="Permutation size"):
            invalid_perm = torch.randperm(64).numpy()
            self.permutable_model._validate_permutation(layer, invalid_perm, 'input')
        
        # Invalid permutation (not a permutation)
        with pytest.raises(ValueError, match="Invalid permutation"):
            invalid_perm = torch.zeros(128).numpy()
            self.permutable_model._validate_permutation(layer, invalid_perm, 'input')
    
    def test_permutation_application(self):
        """Test permutation application."""
        # Create a simple permutation
        perm = torch.randperm(128).numpy()
        
        # Get original weight
        original_weight = self.permutable_model.get_layer('linear1').weight.data.clone()
        
        # Apply permutation
        self.permutable_model.apply_permutation('linear1', perm, 'input')
        
        # Check that weight changed
        new_weight = self.permutable_model.get_layer('linear1').weight.data
        assert not torch.equal(original_weight, new_weight)
        
        # Check that permutation is tracked
        assert self.permutable_model.has_permutation('linear1')
        applied_perms = self.permutable_model.get_applied_permutations()
        assert 'linear1' in applied_perms
    
    def test_dimension_sizes(self):
        """Test dimension size queries."""
        input_size = self.permutable_model.get_dimension_size('linear1', 'input')
        assert input_size == 128
        
        output_size = self.permutable_model.get_dimension_size('linear1', 'output')
        assert output_size == 256
    
    def test_model_summary(self):
        """Test model summary."""
        summary = self.permutable_model.get_model_summary()
        assert 'model_name' in summary
        assert 'model_type' in summary
        assert 'total_parameters' in summary
        assert 'trainable_parameters' in summary
        assert summary['total_parameters'] > 0
    
    def test_clone(self):
        """Test model cloning."""
        cloned = self.permutable_model.clone()
        assert cloned.model_name == self.permutable_model.model_name
        assert cloned.model_type == self.permutable_model.model_type
        
        # Ensure they are separate objects
        assert cloned is not self.permutable_model
        assert cloned.model is not self.permutable_model.model


if __name__ == '__main__':
    pytest.main([__file__])