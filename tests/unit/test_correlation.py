"""
Unit tests for correlation matrix computation.
"""
import pytest
import tempfile
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from src.co_design.correlation import CorrelationMatrixComputer
from src.models.permutable_model import PermutableModel
from src.utils.exceptions import IterativeCoDesignError


class MockModel(nn.Module):
    """Mock model for testing."""
    def __init__(self, input_dim=64, hidden_dim=128, output_dim=32):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class TestCorrelationMatrixComputer:
    """Test cases for CorrelationMatrixComputer."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.computer = CorrelationMatrixComputer(
            cache_dir=self.temp_dir,
            device='cpu'
        )
        
        # Create mock model
        self.mock_model = MockModel()
        self.permutable_model = PermutableModel(
            model=self.mock_model,
            model_type='test',
            model_name='test-model'
        )
    
    def test_initialization(self):
        """Test initialization of CorrelationMatrixComputer."""
        assert self.computer.cache_dir == Path(self.temp_dir)
        assert self.computer.device == 'cpu'
        assert self.computer.cache_dir.exists()
    
    def test_cache_key_generation(self):
        """Test cache key generation."""
        dataset_info = {
            'name': 'test-dataset',
            'sequence_length': 1024,
            'batch_size': 2
        }
        
        key1 = self.computer._generate_cache_key(
            model_name='test-model',
            layer_name='linear1',
            num_samples=100,
            dataset_info=dataset_info
        )
        
        key2 = self.computer._generate_cache_key(
            model_name='test-model',
            layer_name='linear1',
            num_samples=100,
            dataset_info=dataset_info
        )
        
        # Same parameters should generate same key
        assert key1 == key2
        
        # Different parameters should generate different keys
        key3 = self.computer._generate_cache_key(
            model_name='test-model',
            layer_name='linear2',  # Different layer
            num_samples=100,
            dataset_info=dataset_info
        )
        
        assert key1 != key3
    
    def test_correlation_computation(self):
        """Test correlation matrix computation."""
        # Create mock activations
        num_samples = 50
        feature_dim = 32
        activations = torch.randn(num_samples, feature_dim)
        
        # Compute correlation
        correlation = self.computer._compute_correlation(activations)
        
        # Check shape
        assert correlation.shape == (feature_dim, feature_dim)
        
        # Check diagonal is 1.0
        diagonal = torch.diag(correlation)
        assert torch.allclose(diagonal, torch.ones_like(diagonal), atol=1e-3)
        
        # Check symmetry
        assert torch.allclose(correlation, correlation.T, atol=1e-3)
        
        # Check value range
        assert correlation.min() >= -1.1
        assert correlation.max() <= 1.1
    
    def test_correlation_validation(self):
        """Test correlation matrix validation."""
        # Valid correlation matrix
        valid_corr = torch.eye(5)
        assert self.computer.validate_correlation_matrix(valid_corr)
        
        # Invalid shapes
        invalid_1d = torch.ones(5)
        assert not self.computer.validate_correlation_matrix(invalid_1d)
        
        invalid_rect = torch.ones(3, 5)
        assert not self.computer.validate_correlation_matrix(invalid_rect)
        
        # Invalid diagonal
        invalid_diag = torch.eye(3)
        invalid_diag[0, 0] = 0.5
        assert not self.computer.validate_correlation_matrix(invalid_diag)
        
        # Invalid symmetry
        invalid_sym = torch.eye(3)
        invalid_sym[0, 1] = 0.5
        invalid_sym[1, 0] = 0.3
        assert not self.computer.validate_correlation_matrix(invalid_sym)
        
        # Invalid value range
        invalid_range = torch.eye(3)
        invalid_range[0, 1] = 2.0
        invalid_range[1, 0] = 2.0
        assert not self.computer.validate_correlation_matrix(invalid_range)
    
    def test_memory_check(self):
        """Test memory usage checking."""
        layer_info = {
            'weight_shape': [128, 64],
            'has_weight': True
        }
        
        # Should not raise warning for small sizes
        self.computer._check_memory_usage(layer_info, num_samples=10)
        
        # Should warn for large sizes (if available memory is low)
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.available = 1024**3  # 1 GB available
            
            with pytest.warns(UserWarning, match="High memory usage estimated"):
                self.computer._check_memory_usage(layer_info, num_samples=100000)
    
    def test_cache_operations(self):
        """Test cache operations."""
        # Test cache info
        cache_info = self.computer.get_cache_info()
        assert 'cache_dir' in cache_info
        assert 'cached_matrices' in cache_info
        assert cache_info['cached_matrices'] == 0
        
        # Create a fake cached matrix
        fake_matrix = torch.eye(10)
        cache_file = Path(self.temp_dir) / 'test_matrix.pt'
        torch.save(fake_matrix, cache_file)
        
        # Check cache info again
        cache_info = self.computer.get_cache_info()
        assert cache_info['cached_matrices'] == 1
        assert cache_info['total_size_mb'] > 0
        
        # Clear cache
        self.computer.clear_cache()
        assert not cache_file.exists()
    
    def test_precomputed_matrix_loading(self):
        """Test loading precomputed matrices."""
        # Create a test matrix
        test_matrix = torch.eye(5)
        test_file = Path(self.temp_dir) / 'test_precomputed.pt'
        torch.save(test_matrix, test_file)
        
        # Load the matrix
        loaded_matrix = self.computer.load_precomputed_matrix(str(test_file))
        
        # Check it's the same
        assert torch.allclose(loaded_matrix, test_matrix)
        
        # Test loading non-existent file
        with pytest.raises(FileNotFoundError):
            self.computer.load_precomputed_matrix('nonexistent.pt')
    
    def test_activation_collection_mock(self):
        """Test activation collection with mock data."""
        # Create mock dataloader
        mock_dataloader = Mock()
        mock_dataloader.batch_size = 2
        mock_dataloader.dataset = Mock()
        mock_dataloader.dataset.name = 'test'
        mock_dataloader.dataset.sequence_length = 100
        
        # Create mock batches
        batch1 = torch.randn(2, 64)  # batch_size=2, input_dim=64
        batch2 = torch.randn(2, 64)
        mock_dataloader.__iter__ = Mock(return_value=iter([batch1, batch2]))
        
        # Collect activations
        activations = self.computer._collect_activations(
            model=self.permutable_model,
            dataloader=mock_dataloader,
            layer_name='linear1',
            num_samples=4
        )
        
        # Check shape
        assert activations.shape[0] == 4  # num_samples
        assert activations.shape[1] == 128  # linear1 output dim
    
    def test_full_correlation_computation_mock(self):
        """Test full correlation computation with mock data."""
        # Create mock dataloader
        mock_dataloader = Mock()
        mock_dataloader.batch_size = 2
        mock_dataloader.dataset = Mock()
        mock_dataloader.dataset.name = 'test'
        mock_dataloader.dataset.sequence_length = 100
        
        # Create mock batches
        batch1 = torch.randn(2, 64)
        batch2 = torch.randn(2, 64)
        mock_dataloader.__iter__ = Mock(return_value=iter([batch1, batch2]))
        
        # Mock the _collect_activations method to return known activations
        mock_activations = torch.randn(10, 128)
        
        with patch.object(self.computer, '_collect_activations', return_value=mock_activations):
            correlation_matrix = self.computer.compute_correlation_matrix(
                model=self.permutable_model,
                dataloader=mock_dataloader,
                layer_name='linear1',
                num_samples=10,
                force_recompute=True,
                check_memory=False
            )
        
        # Check result
        assert correlation_matrix.shape == (128, 128)
        assert self.computer.validate_correlation_matrix(correlation_matrix)
    
    def test_caching_behavior(self):
        """Test that caching works correctly."""
        mock_dataloader = Mock()
        mock_dataloader.batch_size = 2
        mock_dataloader.dataset = Mock()
        mock_dataloader.dataset.name = 'test'
        mock_dataloader.dataset.sequence_length = 100
        
        mock_activations = torch.randn(10, 128)
        
        with patch.object(self.computer, '_collect_activations', return_value=mock_activations):
            # First computation should compute and cache
            correlation1 = self.computer.compute_correlation_matrix(
                model=self.permutable_model,
                dataloader=mock_dataloader,
                layer_name='linear1',
                num_samples=10,
                force_recompute=True,
                check_memory=False
            )
            
            # Second computation should load from cache
            correlation2 = self.computer.compute_correlation_matrix(
                model=self.permutable_model,
                dataloader=mock_dataloader,
                layer_name='linear1',
                num_samples=10,
                force_recompute=False,
                check_memory=False
            )
            
            # Should be identical
            assert torch.allclose(correlation1, correlation2)
            
            # Cache should exist
            cache_info = self.computer.get_cache_info()
            assert cache_info['cached_matrices'] >= 1


if __name__ == '__main__':
    pytest.main([__file__])