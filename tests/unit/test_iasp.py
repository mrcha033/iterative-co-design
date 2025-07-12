"""
Unit tests for IASP (IO-Aware Scan Permutation) module.

These tests verify the core IASP permutation optimization functionality.
"""
import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch

from src.co_design.iasp import (
    IASPConfig, IASPResults, IASPPermutationOptimizer,
    optimize_memory_layout, compute_memory_layout_permutation
)


class TestIASPConfig:
    """Test IASPConfig validation."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = IASPConfig()
        
        assert config.num_clusters == 64
        assert config.method == 'spectral'
        assert config.correlation_threshold == 0.1
        assert config.max_iterations == 100
        assert config.random_seed == 42
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = IASPConfig(
            num_clusters=32,
            method='kmeans',
            correlation_threshold=0.2,
            max_iterations=50
        )
        
        assert config.num_clusters == 32
        assert config.method == 'kmeans'
        assert config.correlation_threshold == 0.2
        assert config.max_iterations == 50
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        valid_config = IASPConfig(num_clusters=16)
        valid_config.validate()  # Should not raise
        
        # Invalid configs
        with pytest.raises(ValueError, match="num_clusters must be positive"):
            IASPConfig(num_clusters=0).validate()
        
        with pytest.raises(ValueError, match="correlation_threshold must be between 0 and 1"):
            IASPConfig(correlation_threshold=-0.1).validate()
        
        with pytest.raises(ValueError, match="correlation_threshold must be between 0 and 1"):
            IASPConfig(correlation_threshold=1.1).validate()
        
        with pytest.raises(ValueError, match="max_iterations must be positive"):
            IASPConfig(max_iterations=-1).validate()
        
        with pytest.raises(ValueError, match="Unknown method"):
            IASPConfig(method='invalid').validate()


class TestIASPResults:
    """Test IASPResults functionality."""
    
    def test_results_creation(self):
        """Test creating IASPResults."""
        permutation = torch.tensor([2, 0, 1, 3])
        results = IASPResults(
            permutation=permutation,
            modularity=0.75,
            num_clusters=4,
            correlation_matrix=torch.eye(4),
            optimization_time=1.5
        )
        
        assert torch.equal(results.permutation, permutation)
        assert results.modularity == 0.75
        assert results.num_clusters == 4
        assert results.optimization_time == 1.5
    
    def test_to_dict(self):
        """Test converting results to dictionary."""
        permutation = torch.tensor([1, 0, 2])
        correlation_matrix = torch.randn(3, 3)
        
        results = IASPResults(
            permutation=permutation,
            modularity=0.65,
            num_clusters=2,
            correlation_matrix=correlation_matrix,
            optimization_time=0.8
        )
        
        result_dict = results.to_dict()
        
        assert 'permutation' in result_dict
        assert result_dict['modularity'] == 0.65
        assert result_dict['num_clusters'] == 2
        assert result_dict['optimization_time'] == 0.8
        assert 'correlation_matrix' in result_dict


class TestIASPPermutationOptimizer:
    """Test IASPPermutationOptimizer functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = IASPConfig(
            num_clusters=4,
            method='spectral',
            correlation_threshold=0.1,
            random_seed=42
        )
        self.layer_name = 'mixer.in_proj'
        
    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        optimizer = IASPPermutationOptimizer(
            layer_name=self.layer_name,
            config=self.config
        )
        
        assert optimizer.layer_name == self.layer_name
        assert optimizer.config.num_clusters == 4
        assert optimizer.correlation_matrix is None
        assert optimizer.last_permutation is None
    
    def test_create_toy_model(self):
        """Test creation of toy model for testing."""
        optimizer = IASPPermutationOptimizer(self.layer_name, self.config)
        
        # Create a simple model with the target layer
        class ToyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.mixer = nn.ModuleDict({
                    'in_proj': nn.Linear(16, 32)
                })
            
            def forward(self, x):
                return self.mixer.in_proj(x)
        
        model = ToyModel()
        
        # Test that we can extract the target layer
        layer = optimizer._get_target_layer(model)
        assert isinstance(layer, nn.Linear)
        assert layer.in_features == 16
        assert layer.out_features == 32
    
    def test_extract_target_layer_success(self):
        """Test successful target layer extraction."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.mixer = nn.ModuleDict({
                    'in_proj': nn.Linear(8, 16)
                })
        
        model = TestModel()
        optimizer = IASPPermutationOptimizer(self.layer_name, self.config)
        
        layer = optimizer._get_target_layer(model)
        assert isinstance(layer, nn.Linear)
        assert layer.in_features == 8
    
    def test_extract_target_layer_failure(self):
        """Test target layer extraction failure."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.other_layer = nn.Linear(8, 16)
        
        model = TestModel()
        optimizer = IASPPermutationOptimizer(self.layer_name, self.config)
        
        with pytest.raises(ValueError, match="Layer .* not found"):
            optimizer._get_target_layer(model)
    
    def test_validate_correlation_matrix(self):
        """Test correlation matrix validation."""
        optimizer = IASPPermutationOptimizer(self.layer_name, self.config)
        
        # Valid correlation matrix
        valid_matrix = torch.tensor([
            [1.0, 0.5, 0.2],
            [0.5, 1.0, 0.3],
            [0.2, 0.3, 1.0]
        ])
        
        optimizer._validate_correlation_matrix(valid_matrix)  # Should not raise
        
        # Invalid matrices
        with pytest.raises(ValueError, match="must be 2D"):
            optimizer._validate_correlation_matrix(torch.randn(5))
        
        with pytest.raises(ValueError, match="must be square"):
            optimizer._validate_correlation_matrix(torch.randn(3, 4))
        
        # Non-symmetric matrix
        asymmetric = torch.tensor([
            [1.0, 0.5],
            [0.3, 1.0]
        ])
        with pytest.raises(ValueError, match="must be symmetric"):
            optimizer._validate_correlation_matrix(asymmetric)
    
    @patch('src.co_design.spectral.spectral_clustering')
    def test_compute_permutation_spectral(self, mock_spectral):
        """Test permutation computation using spectral clustering."""
        # Mock spectral clustering result
        mock_spectral.return_value = torch.tensor([0, 0, 1, 1])  # 2 clusters
        
        optimizer = IASPPermutationOptimizer(self.layer_name, self.config)
        
        # Create correlation matrix
        correlation_matrix = torch.tensor([
            [1.0, 0.8, 0.1, 0.2],
            [0.8, 1.0, 0.2, 0.1],
            [0.1, 0.2, 1.0, 0.7],
            [0.2, 0.1, 0.7, 1.0]
        ])
        
        permutation, modularity = optimizer._compute_permutation_from_correlation(correlation_matrix)
        
        # Should group similar elements together: [0, 1, 2, 3] -> cluster-based order
        assert len(permutation) == 4
        assert set(permutation.tolist()) == {0, 1, 2, 3}  # All indices present
        assert modularity >= 0.0  # Modularity should be non-negative
        
        mock_spectral.assert_called_once()
    
    def test_compute_permutation_kmeans(self):
        """Test permutation computation using k-means clustering."""
        config = IASPConfig(method='kmeans', num_clusters=2, random_seed=42)
        optimizer = IASPPermutationOptimizer(self.layer_name, config)
        
        # Create correlation matrix with clear block structure
        correlation_matrix = torch.tensor([
            [1.0, 0.9, 0.1, 0.1],
            [0.9, 1.0, 0.1, 0.1], 
            [0.1, 0.1, 1.0, 0.8],
            [0.1, 0.1, 0.8, 1.0]
        ])
        
        permutation, modularity = optimizer._compute_permutation_from_correlation(correlation_matrix)
        
        assert len(permutation) == 4
        assert set(permutation.tolist()) == {0, 1, 2, 3}
        assert modularity >= 0.0
    
    def test_apply_permutation_to_layer(self):
        """Test applying permutation to model layer."""
        optimizer = IASPPermutationOptimizer(self.layer_name, self.config)
        
        # Create test layer
        layer = nn.Linear(4, 6)
        original_weight = layer.weight.data.clone()
        original_bias = layer.bias.data.clone()
        
        # Create permutation [0,1,2,3] -> [2,0,1,3]
        permutation = torch.tensor([2, 0, 1, 3])
        
        optimizer._apply_permutation_to_layer(layer, permutation)
        
        # Check that weights were permuted correctly
        expected_weight = original_weight[:, permutation]
        assert torch.allclose(layer.weight.data, expected_weight)
        
        # Bias should remain unchanged for input permutation
        assert torch.allclose(layer.bias.data, original_bias)
    
    def test_create_mock_dataloader(self):
        """Test creation of mock dataloader for testing."""
        optimizer = IASPPermutationOptimizer(self.layer_name, self.config)
        
        input_dim = 16
        batch_size = 4
        num_batches = 3
        
        dataloader = optimizer._create_mock_dataloader(input_dim, batch_size, num_batches)
        
        batches = list(dataloader)
        assert len(batches) == num_batches
        
        for batch in batches:
            assert batch.shape == (batch_size, input_dim)
            assert batch.dtype == torch.float32
    
    @patch('src.co_design.correlation.compute_activation_correlation')
    def test_compute_permutation_with_mock_data(self, mock_correlation):
        """Test full permutation computation with mock data."""
        # Mock correlation computation
        correlation_matrix = torch.tensor([
            [1.0, 0.7, 0.2, 0.1],
            [0.7, 1.0, 0.1, 0.2],
            [0.2, 0.1, 1.0, 0.6],
            [0.1, 0.2, 0.6, 1.0]
        ])
        mock_correlation.return_value = correlation_matrix
        
        optimizer = IASPPermutationOptimizer(self.layer_name, self.config)
        
        # Create test model
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.mixer = nn.ModuleDict({
                    'in_proj': nn.Linear(4, 8)
                })
            
            def forward(self, x):
                return self.mixer.in_proj(x)
        
        model = TestModel()
        dataloader = optimizer._create_mock_dataloader(4, 2, 3)
        
        permutation, results = optimizer.compute_permutation(model, dataloader)
        
        assert len(permutation) == 4
        assert isinstance(results, IASPResults)
        assert results.modularity >= 0.0
        assert torch.equal(results.correlation_matrix, correlation_matrix)
        
        mock_correlation.assert_called_once()
    
    def test_permutation_identity_case(self):
        """Test permutation computation with identity correlation matrix."""
        optimizer = IASPPermutationOptimizer(self.layer_name, self.config)
        
        # Identity matrix should result in identity permutation (or close to it)
        identity_matrix = torch.eye(4)
        
        permutation, modularity = optimizer._compute_permutation_from_correlation(identity_matrix)
        
        assert len(permutation) == 4
        assert set(permutation.tolist()) == {0, 1, 2, 3}
        # Modularity might be low for identity matrix
        assert modularity >= 0.0


class TestUtilityFunctions:
    """Test utility functions."""
    
    @patch.object(IASPPermutationOptimizer, 'compute_permutation')
    @patch.object(IASPPermutationOptimizer, 'apply_permutation')
    def test_optimize_memory_layout(self, mock_apply, mock_compute):
        """Test optimize_memory_layout utility function."""
        # Mock returns
        mock_permutation = torch.tensor([1, 0, 2, 3])
        mock_results = IASPResults(
            permutation=mock_permutation,
            modularity=0.8,
            num_clusters=2,
            correlation_matrix=torch.eye(4),
            optimization_time=1.0
        )
        mock_compute.return_value = (mock_permutation, mock_results)
        
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(4, 6)
        
        model = TestModel()
        dataloader = [torch.randn(2, 4)]
        
        results = optimize_memory_layout(
            model=model,
            dataloader=dataloader,
            layer_name='layer',
            num_clusters=2
        )
        
        assert results.modularity == 0.8
        assert torch.equal(results.permutation, mock_permutation)
        
        mock_compute.assert_called_once()
        mock_apply.assert_called_once_with(model, mock_permutation)
    
    @patch('src.co_design.correlation.compute_activation_correlation')
    def test_compute_memory_layout_permutation(self, mock_correlation):
        """Test compute_memory_layout_permutation utility function."""
        # Mock correlation matrix
        correlation_matrix = torch.tensor([
            [1.0, 0.5, 0.2],
            [0.5, 1.0, 0.3],
            [0.2, 0.3, 1.0]
        ])
        mock_correlation.return_value = correlation_matrix
        
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.target_layer = nn.Linear(3, 6)
        
        model = TestModel()
        dataloader = [torch.randn(2, 3)]
        
        permutation, modularity = compute_memory_layout_permutation(
            model=model,
            dataloader=dataloader,
            layer_name='target_layer',
            num_clusters=2,
            method='spectral'
        )
        
        assert len(permutation) == 3
        assert set(permutation.tolist()) == {0, 1, 2}
        assert modularity >= 0.0
        
        mock_correlation.assert_called_once()


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_layer_name(self):
        """Test handling of invalid layer names."""
        config = IASPConfig()
        optimizer = IASPPermutationOptimizer('nonexistent.layer', config)
        
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.real_layer = nn.Linear(4, 6)
        
        model = TestModel()
        
        with pytest.raises(ValueError, match="Layer .* not found"):
            optimizer._get_target_layer(model)
    
    def test_empty_dataloader(self):
        """Test handling of empty dataloader."""
        config = IASPConfig()
        optimizer = IASPPermutationOptimizer('layer', config)
        
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(4, 6)
        
        model = TestModel()
        empty_dataloader = []
        
        with pytest.raises(ValueError, match="Dataloader is empty"):
            optimizer.compute_permutation(model, empty_dataloader)
    
    def test_dimension_mismatch(self):
        """Test handling of dimension mismatches."""
        config = IASPConfig()
        optimizer = IASPPermutationOptimizer('layer', config)
        
        layer = nn.Linear(4, 6)
        wrong_permutation = torch.tensor([0, 1, 2])  # Only 3 elements for 4D input
        
        with pytest.raises(ValueError, match="Permutation length"):
            optimizer._apply_permutation_to_layer(layer, wrong_permutation)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])