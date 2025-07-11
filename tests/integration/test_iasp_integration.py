"""
Integration tests for IASP correlation matrix computation and caching.
"""
import pytest
import tempfile
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import Mock

from src.co_design.correlation import CorrelationMatrixComputer
from src.co_design.iasp import IASPPermutationOptimizer
from src.models.permutable_model import PermutableModel
from src.utils.text_dataset import TextDataset


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self, input_dim=64, hidden_dim=128):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 32)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class TestIASPIntegration:
    """Integration tests for IASP correlation and permutation optimization."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create simple model
        self.model = SimpleModel()
        self.permutable_model = PermutableModel(
            model=self.model,
            model_type='test',
            model_name='test-model'
        )
        
        # Create correlation computer
        self.correlation_computer = CorrelationMatrixComputer(
            cache_dir=self.temp_dir,
            device='cpu'
        )
        
        # Create IASP optimizer
        self.iasp_optimizer = IASPPermutationOptimizer(
            correlation_computer=self.correlation_computer,
            device='cpu'
        )
    
    def create_mock_dataloader(self, num_batches=5, batch_size=2, input_dim=64):
        """Create a mock dataloader for testing."""
        mock_dataloader = Mock()
        mock_dataloader.batch_size = batch_size
        mock_dataloader.dataset = Mock()
        mock_dataloader.dataset.name = 'test'
        mock_dataloader.dataset.sequence_length = 100
        
        # Create batches
        batches = []
        for _ in range(num_batches):
            batch = torch.randn(batch_size, input_dim)
            batches.append(batch)
        
        mock_dataloader.__iter__ = Mock(return_value=iter(batches))
        return mock_dataloader
    
    def test_full_correlation_workflow(self):
        """Test complete correlation matrix computation workflow."""
        # Create mock dataloader
        dataloader = self.create_mock_dataloader()
        
        # Compute correlation matrix
        correlation_matrix = self.correlation_computer.compute_correlation_matrix(
            model=self.permutable_model,
            dataloader=dataloader,
            layer_name='linear1',
            num_samples=10,
            force_recompute=True,
            check_memory=False
        )
        
        # Validate result
        assert correlation_matrix.shape == (128, 128)  # linear1 output dim
        assert self.correlation_computer.validate_correlation_matrix(correlation_matrix)
        
        # Check that matrix was cached
        cache_info = self.correlation_computer.get_cache_info()
        assert cache_info['cached_matrices'] >= 1
        
        # Test loading from cache
        correlation_matrix2 = self.correlation_computer.compute_correlation_matrix(
            model=self.permutable_model,
            dataloader=dataloader,
            layer_name='linear1',
            num_samples=10,
            force_recompute=False,
            check_memory=False
        )
        
        # Should be identical
        assert torch.allclose(correlation_matrix, correlation_matrix2)
    
    def test_iasp_permutation_optimization(self):
        """Test IASP permutation optimization."""
        # Create mock dataloader
        dataloader = self.create_mock_dataloader()
        
        # Test spectral method
        permutation, info = self.iasp_optimizer.optimize_permutation(
            model=self.permutable_model,
            dataloader=dataloader,
            layer_name='linear1',
            num_clusters=8,
            num_samples=10,
            method='spectral',
            force_recompute=True
        )
        
        # Validate permutation
        assert len(permutation) == 128  # linear1 output dim
        assert set(permutation) == set(range(128))  # Valid permutation
        assert 'modularity' in info
        assert 'method' in info
        assert info['method'] == 'spectral'
        
        # Test TSP method
        permutation_tsp, info_tsp = self.iasp_optimizer.optimize_permutation(
            model=self.permutable_model,
            dataloader=dataloader,
            layer_name='linear1',
            num_clusters=8,
            num_samples=10,
            method='tsp',
            force_recompute=True
        )
        
        # Validate TSP permutation
        assert len(permutation_tsp) == 128
        assert set(permutation_tsp) == set(range(128))
        assert info_tsp['method'] == 'tsp'
        
        # Test random method
        permutation_random, info_random = self.iasp_optimizer.optimize_permutation(
            model=self.permutable_model,
            dataloader=dataloader,
            layer_name='linear1',
            num_clusters=8,
            num_samples=10,
            method='random',
            force_recompute=True
        )
        
        # Validate random permutation
        assert len(permutation_random) == 128
        assert set(permutation_random) == set(range(128))
        assert info_random['method'] == 'random'
    
    def test_permutation_application(self):
        """Test applying permutations to model."""
        # Create mock dataloader
        dataloader = self.create_mock_dataloader()
        
        # Get original weights
        original_weight = self.permutable_model.get_layer('linear1').weight.data.clone()
        
        # Generate permutation
        permutation, _ = self.iasp_optimizer.optimize_permutation(
            model=self.permutable_model,
            dataloader=dataloader,
            layer_name='linear1',
            num_clusters=8,
            num_samples=10,
            method='spectral',
            force_recompute=True
        )
        
        # Apply permutation
        self.iasp_optimizer.apply_permutation(
            model=self.permutable_model,
            layer_name='linear1',
            permutation=permutation,
            dimension='output'
        )
        
        # Check that weights changed
        new_weight = self.permutable_model.get_layer('linear1').weight.data
        assert not torch.equal(original_weight, new_weight)
        
        # Check that permutation is tracked
        assert self.permutable_model.has_permutation('linear1')
    
    def test_cache_key_consistency(self):
        """Test that cache keys are consistent across runs."""
        # Create mock dataloader
        dataloader = self.create_mock_dataloader()
        
        # Generate cache key
        dataset_info = self.correlation_computer._get_dataset_info(dataloader)
        key1 = self.correlation_computer._generate_cache_key(
            model_name='test-model',
            layer_name='linear1',
            num_samples=10,
            dataset_info=dataset_info
        )
        
        # Generate same key again
        key2 = self.correlation_computer._generate_cache_key(
            model_name='test-model',
            layer_name='linear1',
            num_samples=10,
            dataset_info=dataset_info
        )
        
        # Should be identical
        assert key1 == key2
        
        # Different model should give different key
        key3 = self.correlation_computer._generate_cache_key(
            model_name='different-model',
            layer_name='linear1',
            num_samples=10,
            dataset_info=dataset_info
        )
        
        assert key1 != key3
    
    def test_correlation_matrix_properties(self):
        """Test that computed correlation matrices have correct properties."""
        # Create mock dataloader
        dataloader = self.create_mock_dataloader()
        
        # Compute correlation matrix
        correlation_matrix = self.correlation_computer.compute_correlation_matrix(
            model=self.permutable_model,
            dataloader=dataloader,
            layer_name='linear1',
            num_samples=20,
            force_recompute=True,
            check_memory=False
        )
        
        # Test mathematical properties
        
        # 1. Square matrix
        assert correlation_matrix.shape[0] == correlation_matrix.shape[1]
        
        # 2. Diagonal is 1.0
        diagonal = torch.diag(correlation_matrix)
        assert torch.allclose(diagonal, torch.ones_like(diagonal), atol=1e-3)
        
        # 3. Symmetric
        assert torch.allclose(correlation_matrix, correlation_matrix.T, atol=1e-3)
        
        # 4. Values in [-1, 1]
        assert correlation_matrix.min() >= -1.1
        assert correlation_matrix.max() <= 1.1
        
        # 5. Positive semi-definite (eigenvalues >= 0)
        eigenvalues = torch.linalg.eigvals(correlation_matrix)
        assert torch.all(eigenvalues.real >= -1e-6)  # Allow small numerical errors
    
    def test_modularity_computation(self):
        """Test modularity computation for permutations."""
        # Create a simple test correlation matrix
        correlation_matrix = torch.eye(16)  # Identity matrix
        
        # Test identity permutation
        identity_perm = np.arange(16)
        modularity_identity = self.iasp_optimizer._compute_modularity(
            correlation_matrix, identity_perm
        )
        
        # Test random permutation
        random_perm = np.random.permutation(16)
        modularity_random = self.iasp_optimizer._compute_modularity(
            correlation_matrix, random_perm
        )
        
        # Modularity should be well-defined
        assert isinstance(modularity_identity, float)
        assert isinstance(modularity_random, float)
    
    def test_permutation_evaluation(self):
        """Test permutation evaluation metrics."""
        # Create test correlation matrix
        correlation_matrix = torch.randn(32, 32)
        correlation_matrix = correlation_matrix @ correlation_matrix.T  # Make PSD
        correlation_matrix = correlation_matrix / torch.sqrt(torch.diag(correlation_matrix))[:, None]
        correlation_matrix = correlation_matrix / torch.sqrt(torch.diag(correlation_matrix))[None, :]
        correlation_matrix.fill_diagonal_(1.0)
        
        # Test permutation
        permutation = np.random.permutation(32)
        
        # Evaluate permutation
        evaluation = self.iasp_optimizer.evaluate_permutation(
            correlation_matrix, permutation
        )
        
        # Check that all metrics are present
        assert 'modularity' in evaluation
        assert 'locality' in evaluation
        assert 'block_coherence' in evaluation
        assert 'permutation_length' in evaluation
        
        # Check that values are reasonable
        assert evaluation['permutation_length'] == 32
        assert isinstance(evaluation['modularity'], float)
        assert isinstance(evaluation['locality'], float)
        assert isinstance(evaluation['block_coherence'], float)
    
    def test_error_handling(self):
        """Test error handling in correlation computation."""
        # Create mock dataloader that will cause errors
        bad_dataloader = Mock()
        bad_dataloader.batch_size = 1
        bad_dataloader.dataset = Mock()
        bad_dataloader.dataset.name = 'test'
        bad_dataloader.dataset.sequence_length = 100
        bad_dataloader.__iter__ = Mock(return_value=iter([]))  # Empty iterator
        
        # Should raise error for empty dataloader
        with pytest.raises(IterativeCoDesignError):
            self.correlation_computer.compute_correlation_matrix(
                model=self.permutable_model,
                dataloader=bad_dataloader,
                layer_name='linear1',
                num_samples=10,
                force_recompute=True,
                check_memory=False
            )
        
        # Test invalid layer name
        dataloader = self.create_mock_dataloader()
        
        with pytest.raises(ValueError):
            self.correlation_computer.compute_correlation_matrix(
                model=self.permutable_model,
                dataloader=dataloader,
                layer_name='nonexistent_layer',
                num_samples=10,
                force_recompute=True,
                check_memory=False
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])