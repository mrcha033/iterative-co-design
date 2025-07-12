"""
Unit tests for Hardware-Native Differentiable Sparsity (HDS) implementation.
"""
import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch

from src.co_design.hds import (
    HDSConfig, GumbelTopK, StructuredSparsityMask, HDSLayer, 
    HDSOptimizer, apply_hds_to_model, validate_sparsity_pattern
)
from src.models.permutable_model import PermutableModel


class TestHDSConfig:
    """Test HDSConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = HDSConfig()
        
        assert config.sparsity_ratio == "2:4"
        assert config.target_sparsity == 0.5
        assert config.learning_rate == 1e-4
        assert config.num_epochs == 10
        assert config.gumbel_temperature == 1.0
        assert config.block_size == 16
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = HDSConfig(
            sparsity_ratio="1:4",
            target_sparsity=0.75,
            learning_rate=1e-3,
            num_epochs=20
        )
        
        assert config.sparsity_ratio == "1:4"
        assert config.target_sparsity == 0.75
        assert config.learning_rate == 1e-3
        assert config.num_epochs == 20


class TestGumbelTopK:
    """Test GumbelTopK module."""
    
    def test_initialization(self):
        """Test GumbelTopK initialization."""
        shape = (4, 8)
        k = 3
        temp = 0.5
        
        gumbel = GumbelTopK(shape, k, temp)
        
        assert gumbel.shape == shape
        assert gumbel.k == k
        assert gumbel.temperature == temp
        assert gumbel.mask_logits.shape == shape
    
    def test_forward_training(self):
        """Test forward pass during training."""
        shape = (4, 8)
        k = 3
        
        gumbel = GumbelTopK(shape, k)
        gumbel.train()
        
        mask = gumbel(training=True)
        
        assert mask.shape == shape
        assert torch.all(mask >= 0)
        assert torch.all(mask <= 1)
    
    def test_forward_inference(self):
        """Test forward pass during inference."""
        shape = (4, 8)
        k = 3
        
        gumbel = GumbelTopK(shape, k)
        gumbel.eval()
        
        mask = gumbel(training=False)
        
        assert mask.shape == shape
        assert torch.all((mask == 0) | (mask == 1))  # Binary mask
        assert torch.sum(mask).item() == k  # Exactly k elements
    
    def test_temperature_update(self):
        """Test temperature update."""
        gumbel = GumbelTopK((4, 8), 3, 1.0)
        
        gumbel.update_temperature(0.5)
        assert gumbel.temperature == 0.5
        
        # Test minimum temperature
        gumbel.update_temperature(0.0)
        assert gumbel.temperature == 1e-8


class TestStructuredSparsityMask:
    """Test StructuredSparsityMask module."""
    
    def test_initialization(self):
        """Test StructuredSparsityMask initialization."""
        weight_shape = (16, 32)
        sparsity_ratio = "2:4"
        
        mask = StructuredSparsityMask(weight_shape, sparsity_ratio)
        
        assert mask.weight_shape == weight_shape
        assert mask.sparsity_ratio == sparsity_ratio
        assert mask.n == 2
        assert mask.m == 4
    
    def test_forward(self):
        """Test forward pass."""
        weight_shape = (8, 16)
        sparsity_ratio = "2:4"
        
        mask = StructuredSparsityMask(weight_shape, sparsity_ratio)
        
        # Training mode
        mask.train()
        output = mask(training=True)
        assert output.shape == weight_shape
        
        # Inference mode
        mask.eval()
        output = mask(training=False)
        assert output.shape == weight_shape
    
    def test_num_blocks_calculation(self):
        """Test number of blocks calculation."""
        weight_shape = (8, 16)  # 128 elements
        sparsity_ratio = "2:4"  # 4 elements per block
        
        mask = StructuredSparsityMask(weight_shape, sparsity_ratio)
        
        expected_blocks = 128 // 4  # 32 blocks
        assert mask.num_blocks == expected_blocks
        assert len(mask.mask_generators) == expected_blocks


class TestHDSLayer:
    """Test HDSLayer wrapper."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.linear_layer = nn.Linear(8, 4)
        self.config = HDSConfig(sparsity_ratio="2:4")
        self.hds_layer = HDSLayer(self.linear_layer, self.config, "test_layer")
    
    def test_initialization(self):
        """Test HDSLayer initialization."""
        assert self.hds_layer.layer == self.linear_layer
        assert self.hds_layer.config == self.config
        assert self.hds_layer.layer_name == "test_layer"
        assert 'weight' in self.hds_layer.weight_masks
        assert 'weight' in self.hds_layer.original_weights
    
    def test_forward(self):
        """Test forward pass."""
        input_tensor = torch.randn(2, 8)
        
        # Original layer output
        original_output = self.linear_layer(input_tensor)
        
        # HDS layer output
        hds_output = self.hds_layer(input_tensor)
        
        assert hds_output.shape == original_output.shape
    
    def test_get_sparsity_ratio(self):
        """Test sparsity ratio calculation."""
        sparsity_ratios = self.hds_layer.get_sparsity_ratio()
        
        assert 'weight' in sparsity_ratios
        assert 0.0 <= sparsity_ratios['weight'] <= 1.0
    
    def test_temperature_update(self):
        """Test temperature update."""
        new_temp = 0.5
        self.hds_layer.update_temperature(new_temp)
        
        # Verify temperature was updated
        for mask in self.hds_layer.weight_masks.values():
            assert mask.temperature == new_temp


class TestHDSOptimizer:
    """Test HDSOptimizer class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = HDSConfig(num_epochs=2, learning_rate=1e-3)
        self.optimizer = HDSOptimizer(self.config)
        
        # Create a simple test model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(8, 4)
                self.linear2 = nn.Linear(4, 2)
            
            def forward(self, x):
                return self.linear2(torch.relu(self.linear1(x)))
        
        self.model = SimpleModel()
        self.permutable_model = PermutableModel(self.model, 'test', 'test')
    
    def test_initialization(self):
        """Test HDSOptimizer initialization."""
        assert self.optimizer.config == self.config
        assert len(self.optimizer.hds_layers) == 0
        assert self.optimizer.current_epoch == 0
    
    def test_get_target_layers(self):
        """Test target layer identification."""
        target_layers = self.optimizer._get_target_layers(self.permutable_model)
        
        assert 'linear1' in target_layers
        assert 'linear2' in target_layers
        assert len(target_layers) == 2
    
    def test_prepare_model(self):
        """Test model preparation."""
        prepared_model = self.optimizer.prepare_model(self.permutable_model)
        
        assert len(self.optimizer.hds_layers) == 2
        assert 'linear1' in self.optimizer.hds_layers
        assert 'linear2' in self.optimizer.hds_layers
        assert prepared_model == self.permutable_model
    
    def test_sparsity_info(self):
        """Test sparsity information gathering."""
        # Prepare model first
        self.optimizer.prepare_model(self.permutable_model)
        
        sparsity_info = self.optimizer._get_sparsity_info()
        
        assert 'linear1' in sparsity_info
        assert 'linear2' in sparsity_info
        assert 'avg_sparsity' in sparsity_info
        assert 0.0 <= sparsity_info['avg_sparsity'] <= 1.0
    
    def test_temperature_update(self):
        """Test temperature update across layers."""
        # Prepare model first
        self.optimizer.prepare_model(self.permutable_model)
        
        new_temp = 0.3
        self.optimizer._update_temperature(new_temp)
        
        # Check all layers have updated temperature
        for hds_layer in self.optimizer.hds_layers.values():
            for mask in hds_layer.weight_masks.values():
                assert mask.temperature == new_temp


class TestUtilityFunctions:
    """Test utility functions."""
    
    def setup_method(self):
        """Setup test fixtures."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(8, 4)
            
            def forward(self, x):
                return self.linear(x)
        
        self.model = SimpleModel()
        self.permutable_model = PermutableModel(self.model, 'test', 'test')
        
        # Create mock dataloader
        self.dataloader = Mock()
        self.dataloader.__iter__ = Mock(return_value=iter([
            torch.randn(2, 8) for _ in range(3)
        ]))
        self.dataloader.__len__ = Mock(return_value=3)
    
    def test_apply_hds_to_model(self):
        """Test apply_hds_to_model function."""
        config = HDSConfig(num_epochs=1)
        
        # Mock the training process
        with patch('src.co_design.hds.HDSOptimizer.train') as mock_train:
            mock_train.return_value = {'training_history': []}
            
            sparse_model, results = apply_hds_to_model(
                self.permutable_model,
                self.dataloader,
                config
            )
            
            assert sparse_model is not None
            assert 'training_history' in results
            mock_train.assert_called_once()
    
    def test_validate_sparsity_pattern(self):
        """Test sparsity pattern validation."""
        # Create a model with some sparse weights
        model = self.permutable_model
        
        # Make some weights sparse manually
        with torch.no_grad():
            weight = model.model.linear.weight
            # Create a simple 2:4 pattern
            mask = torch.zeros_like(weight)
            mask[0, 0] = 1  # First 2 elements in first block
            mask[0, 1] = 1
            # mask[0, 2] = 0  # Last 2 elements zero
            # mask[0, 3] = 0
            
            weight.data = weight.data * mask
        
        results = validate_sparsity_pattern(model, "2:4")
        
        assert 'valid_layers' in results
        assert 'invalid_layers' in results
        assert 'overall_sparsity' in results
        assert 'pattern_compliance' in results
        assert 0.0 <= results['overall_sparsity'] <= 1.0
        assert 0.0 <= results['pattern_compliance'] <= 1.0


class TestIntegration:
    """Integration tests for HDS module."""
    
    def test_end_to_end_hds_workflow(self):
        """Test complete HDS workflow."""
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
            torch.randn(2, 4) for _ in range(2)
        ]))
        dataloader.__len__ = Mock(return_value=2)
        
        # Create HDS optimizer
        config = HDSConfig(num_epochs=1, learning_rate=1e-3)
        optimizer = HDSOptimizer(config)
        
        # Prepare model
        prepared_model = optimizer.prepare_model(permutable_model)
        
        # Check that layers were wrapped
        assert len(optimizer.hds_layers) == 1
        assert 'linear' in optimizer.hds_layers
        
        # Check sparsity info
        sparsity_info = optimizer._get_sparsity_info()
        assert 'linear' in sparsity_info
        assert 'avg_sparsity' in sparsity_info
        
        # Test temperature update
        optimizer._update_temperature(0.5)
        
        # Test finalize masks
        optimizer._finalize_masks()
        
        # The test should complete without errors
        assert True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])