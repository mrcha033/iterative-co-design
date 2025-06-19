import pytest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import patch, MagicMock
from co_design.iasp import (
    get_activation_correlation,
    find_permutation_from_matrix,
    find_optimal_permutation_from_matrix,
)


class SimpleTestModel(nn.Module):
    """Simple model for testing IASP functionality."""
    def __init__(self, hidden_size=32):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 10)
    
    def forward(self, input_ids):
        x = self.linear1(input_ids.float())
        x = torch.relu(x)
        output = self.linear2(x)
        return {"logits": output}


class TestIASP:
    """Unit tests for IASP module."""
    
    def test_find_permutation_from_matrix_basic(self):
        """Test basic permutation finding from correlation matrix."""
        # Create a simple correlation matrix with clear block structure
        correlation_matrix = np.array([
            [1.0, 0.8, 0.1, 0.1],
            [0.8, 1.0, 0.1, 0.1],
            [0.1, 0.1, 1.0, 0.9],
            [0.1, 0.1, 0.9, 1.0]
        ])
        
        permutation = find_permutation_from_matrix(correlation_matrix, n_clusters=2)
        
        assert isinstance(permutation, list)
        assert len(permutation) == 4
        assert set(permutation) == {0, 1, 2, 3}  # Should contain all indices
        
        # Check that strongly correlated pairs are grouped together
        # (The exact order may vary, but correlated pairs should be adjacent)
        perm_array = np.array(permutation)
        
        # Should create 2 groups
        first_two = perm_array[:2]
        last_two = perm_array[2:]
        
        # Within each group, elements should be from the same block
        assert (set(first_two) == {0, 1} and set(last_two) == {2, 3}) or \
               (set(first_two) == {2, 3} and set(last_two) == {0, 1})
    
    def test_find_permutation_from_matrix_single_cluster(self):
        """Test permutation finding with single cluster."""
        correlation_matrix = np.eye(4)  # Identity matrix
        
        permutation = find_permutation_from_matrix(correlation_matrix, n_clusters=1)
        
        assert len(permutation) == 4
        assert set(permutation) == {0, 1, 2, 3}
    
    def test_find_optimal_permutation_with_num_clusters(self):
        """Test optimal permutation finding with specified number of clusters."""
        # Create correlation matrix with known structure
        correlation_matrix = np.array([
            [1.0, 0.7, 0.2, 0.1],
            [0.7, 1.0, 0.1, 0.2],
            [0.2, 0.1, 1.0, 0.8],
            [0.1, 0.2, 0.8, 1.0]
        ])
        
        permutation = find_optimal_permutation_from_matrix(
            correlation_matrix, num_clusters=2
        )
        
        assert isinstance(permutation, list)
        assert len(permutation) == 4
        assert set(permutation) == {0, 1, 2, 3}
    
    def test_find_optimal_permutation_auto_clusters(self):
        """Test optimal permutation finding with automatic cluster detection."""
        # Larger matrix for automatic cluster detection
        size = 32
        correlation_matrix = np.eye(size) + 0.1 * np.random.rand(size, size)
        correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2  # Make symmetric
        np.fill_diagonal(correlation_matrix, 1.0)  # Ensure diagonal is 1
        
        permutation = find_optimal_permutation_from_matrix(
            correlation_matrix, clusters_range=(2, 4)
        )
        
        assert isinstance(permutation, list)
        assert len(permutation) == size
        assert set(permutation) == set(range(size))
    
    def test_get_activation_correlation_basic(self):
        """Test activation correlation computation."""
        model = SimpleTestModel(hidden_size=8)
        model.eval()
        
        # Create simple dataset
        batch_size, seq_len = 4, 6
        input_data = torch.randn(batch_size, seq_len, 8)
        dataset = TensorDataset(input_data)
        dataloader = DataLoader(dataset, batch_size=2)
        
        # Mock the dataloader to return proper format
        mock_dataloader = []
        for batch in dataloader:
            mock_dataloader.append({"input_ids": batch[0]})
        
        correlation_matrix = get_activation_correlation(
            model=model,
            dataloader=mock_dataloader,
            target_layer_name="linear1",
            max_samples=4,
            device="cpu"
        )
        
        assert isinstance(correlation_matrix, np.ndarray)
        assert correlation_matrix.shape == (8, 8)  # hidden_size x hidden_size
        assert np.allclose(np.diag(correlation_matrix), 1.0)  # Diagonal should be 1
        assert np.allclose(correlation_matrix, correlation_matrix.T)  # Should be symmetric
    
    def test_get_activation_correlation_invalid_layer(self):
        """Test error handling for invalid layer name."""
        model = SimpleTestModel()
        
        # Create minimal dataset
        input_data = torch.randn(2, 4, 32)
        dataset = TensorDataset(input_data)
        dataloader = [{"input_ids": input_data}]
        
        with pytest.raises(ValueError, match="Layer 'nonexistent_layer' not found"):
            get_activation_correlation(
                model=model,
                dataloader=dataloader,
                target_layer_name="nonexistent_layer",
                max_samples=2,
                device="cpu"
            )
    
    def test_get_activation_correlation_empty_dataloader(self):
        """Test error handling for empty dataloader."""
        model = SimpleTestModel()
        empty_dataloader = []
        
        with pytest.raises(ValueError, match="No activations were collected"):
            get_activation_correlation(
                model=model,
                dataloader=empty_dataloader,
                target_layer_name="linear1",
                max_samples=10,
                device="cpu"
            )
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_get_activation_correlation_gpu(self):
        """Test activation correlation on GPU."""
        model = SimpleTestModel(hidden_size=8)
        model.eval()
        
        # Create simple dataset
        batch_size, seq_len = 2, 4
        input_data = torch.randn(batch_size, seq_len, 8)
        dataloader = [{"input_ids": input_data}]
        
        correlation_matrix = get_activation_correlation(
            model=model,
            dataloader=dataloader,
            target_layer_name="linear1",
            max_samples=2,
            device="cuda"
        )
        
        assert isinstance(correlation_matrix, np.ndarray)
        assert correlation_matrix.shape == (8, 8)
    
    def test_find_permutation_robustness(self):
        """Test permutation finding with edge cases."""
        # Test with very small matrix
        small_matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
        
        permutation = find_permutation_from_matrix(small_matrix, n_clusters=1)
        assert len(permutation) == 2
        assert set(permutation) == {0, 1}
    
    def test_correlation_matrix_properties(self):
        """Test that computed correlation matrices have expected properties."""
        model = SimpleTestModel(hidden_size=4)
        model.eval()
        
        # Use deterministic input for reproducible test
        torch.manual_seed(42)
        input_data = torch.randn(4, 3, 4)
        dataloader = [{"input_ids": input_data}]
        
        correlation_matrix = get_activation_correlation(
            model=model,
            dataloader=dataloader,
            target_layer_name="linear1",
            max_samples=4,
            device="cpu"
        )
        
        # Test matrix properties
        assert np.all(correlation_matrix >= -1.0)  # Values should be >= -1
        assert np.all(correlation_matrix <= 1.0)   # Values should be <= 1
        assert not np.any(np.isnan(correlation_matrix))  # No NaN values
        assert not np.any(np.isinf(correlation_matrix))  # No infinite values
    
    def test_permutation_from_matrix_reproducibility(self):
        """Test that permutation finding is reproducible."""
        # Use fixed random seed in correlation matrix
        np.random.seed(42)
        size = 8
        correlation_matrix = np.random.rand(size, size)
        correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
        np.fill_diagonal(correlation_matrix, 1.0)
        
        # Should get same result with same input
        perm1 = find_permutation_from_matrix(correlation_matrix, n_clusters=3)
        perm2 = find_permutation_from_matrix(correlation_matrix, n_clusters=3)
        
        assert perm1 == perm2 