"""
Unit tests for spectral clustering implementation.
"""
import numpy as np
import pytest
import torch
from unittest.mock import patch, MagicMock

from src.co_design.spectral import SpectralClusteringOptimizer
from src.utils.exceptions import IterativeCoDesignError


class TestSpectralClusteringOptimizer:
    """Test the SpectralClusteringOptimizer class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.optimizer = SpectralClusteringOptimizer(random_state=42)
        
        # Create a simple test correlation matrix
        self.simple_corr = torch.tensor([
            [1.0, 0.8, 0.1, 0.2],
            [0.8, 1.0, 0.2, 0.1],
            [0.1, 0.2, 1.0, 0.9],
            [0.2, 0.1, 0.9, 1.0]
        ], dtype=torch.float32)
        
        # Create a larger test correlation matrix
        self.large_corr = torch.randn(100, 100)
        # Make it symmetric and positive definite
        self.large_corr = self.large_corr @ self.large_corr.T
        # Normalize to correlation matrix
        diag_sqrt = torch.sqrt(torch.diag(self.large_corr))
        self.large_corr = self.large_corr / diag_sqrt[:, None] / diag_sqrt[None, :]
    
    def test_init(self):
        """Test optimizer initialization."""
        optimizer = SpectralClusteringOptimizer(random_state=123)
        assert optimizer.random_state == 123
        
        # Test default initialization
        default_optimizer = SpectralClusteringOptimizer()
        assert default_optimizer.random_state == 42
    
    def test_construct_affinity_matrix(self):
        """Test affinity matrix construction."""
        corr_np = self.simple_corr.numpy()
        
        # Test basic construction
        W = self.optimizer._construct_affinity_matrix(corr_np, correlation_threshold=0.5)
        
        # Check shape
        assert W.shape == (4, 4)
        
        # Check diagonal is zero
        assert np.allclose(np.diag(W), 0)
        
        # Check thresholding
        assert W[0, 1] > 0  # 0.8 > 0.5
        assert W[0, 2] == 0  # 0.1 < 0.5
        assert W[2, 3] > 0  # 0.9 > 0.5
        
        # Check symmetry (should be symmetric due to abs operation)
        assert np.allclose(W, W.T)
        
        # Test with different threshold
        W_strict = self.optimizer._construct_affinity_matrix(corr_np, correlation_threshold=0.85)
        assert W_strict[0, 1] == 0  # 0.8 < 0.85
        assert W_strict[2, 3] > 0   # 0.9 > 0.85
    
    def test_compute_graph_laplacian_dense(self):
        """Test dense graph Laplacian computation."""
        # Create simple affinity matrix
        W = np.array([
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0]
        ], dtype=float)
        
        L = self.optimizer._compute_graph_laplacian(W, use_sparse=False)
        
        # Check shape
        assert L.shape == (4, 4)
        
        # Check Laplacian properties
        expected_degrees = np.array([1, 2, 2, 1])
        computed_degrees = np.diag(L)
        
        # Account for regularization
        assert np.allclose(computed_degrees, expected_degrees + 1e-8)
        
        # Check row sums are approximately zero (with regularization)
        row_sums = np.sum(L, axis=1)
        assert np.allclose(row_sums, 1e-8, atol=1e-6)
    
    def test_compute_graph_laplacian_sparse(self):
        """Test sparse graph Laplacian computation."""
        # Create simple affinity matrix
        W = np.array([
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0]
        ], dtype=float)
        
        L = self.optimizer._compute_graph_laplacian(W, use_sparse=True)
        
        # Check that it's a sparse matrix
        from scipy.sparse import issparse
        assert issparse(L)
        
        # Convert to dense for comparison
        L_dense = L.toarray()
        
        # Check shape
        assert L_dense.shape == (4, 4)
        
        # Check degrees
        expected_degrees = np.array([1, 2, 2, 1])
        computed_degrees = np.diag(L_dense)
        assert np.allclose(computed_degrees, expected_degrees)
    
    def test_compute_eigenvectors_dense(self):
        """Test dense eigenvector computation."""
        # Create a simple Laplacian matrix
        L = np.array([
            [2, -1, -1, 0],
            [-1, 2, -1, 0],
            [-1, -1, 2, 0],
            [0, 0, 0, 1]
        ], dtype=float)
        
        # Add regularization
        L += 1e-8 * np.eye(4)
        
        features = self.optimizer._compute_eigenvectors(L, num_clusters=2, use_sparse=False)
        
        # Check shape
        assert features.shape == (4, 2)
        
        # Check that features are normalized
        norms = np.linalg.norm(features, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)
    
    def test_compute_eigenvectors_sparse(self):
        """Test sparse eigenvector computation."""
        from scipy.sparse import csr_matrix
        
        # Create a simple Laplacian matrix
        L = np.array([
            [2, -1, -1, 0],
            [-1, 2, -1, 0],
            [-1, -1, 2, 0],
            [0, 0, 0, 1]
        ], dtype=float)
        
        L_sparse = csr_matrix(L)
        
        features = self.optimizer._compute_eigenvectors(L_sparse, num_clusters=2, use_sparse=True)
        
        # Check shape
        assert features.shape == (4, 2)
        
        # Check that features are normalized
        norms = np.linalg.norm(features, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)
    
    def test_compute_eigenvectors_fallback(self):
        """Test eigenvector computation fallback."""
        # Create a problematic matrix that will cause eigenvalue computation to fail
        L = np.array([[np.nan, 0], [0, np.nan]])
        
        with patch('warnings.warn') as mock_warn:
            features = self.optimizer._compute_eigenvectors(L, num_clusters=2, use_sparse=False)
            
            # Check that warning was issued
            mock_warn.assert_called_once()
            
            # Check shape
            assert features.shape == (2, 2)
    
    def test_perform_clustering(self):
        """Test k-means clustering on features."""
        # Create simple features that should cluster well
        features = np.array([
            [0, 1],
            [0, 1.1],
            [1, 0],
            [1.1, 0]
        ])
        
        cluster_labels, silhouette = self.optimizer._perform_clustering(features, num_clusters=2)
        
        # Check shape
        assert len(cluster_labels) == 4
        
        # Check that we have 2 clusters
        unique_labels = np.unique(cluster_labels)
        assert len(unique_labels) == 2
        
        # Check silhouette score
        assert isinstance(silhouette, float)
        assert silhouette > 0  # Should be positive for well-separated clusters
    
    def test_perform_clustering_single_cluster(self):
        """Test clustering with single cluster."""
        features = np.array([[0, 1], [0, 1.1], [1, 0], [1.1, 0]])
        
        cluster_labels, silhouette = self.optimizer._perform_clustering(features, num_clusters=1)
        
        # Check that all points are in the same cluster
        assert len(np.unique(cluster_labels)) == 1
        assert silhouette == 0.0
    
    def test_perform_clustering_fallback(self):
        """Test clustering fallback when k-means fails."""
        # Create features that will cause k-means to fail
        features = np.array([[np.nan, np.nan], [np.inf, np.inf]])
        
        with patch('warnings.warn') as mock_warn:
            cluster_labels, silhouette = self.optimizer._perform_clustering(features, num_clusters=2)
            
            # Check that warning was issued
            mock_warn.assert_called_once()
            
            # Check shape
            assert len(cluster_labels) == 2
            
            # Check silhouette score
            assert silhouette == 0.0
    
    def test_construct_permutation(self):
        """Test permutation construction from cluster labels."""
        # Simple case with 2 clusters
        cluster_labels = np.array([0, 0, 1, 1])
        
        permutation = self.optimizer._construct_permutation(cluster_labels, num_clusters=2, dimension=4)
        
        # Check shape
        assert len(permutation) == 4
        
        # Check that it's a valid permutation
        assert set(permutation) == {0, 1, 2, 3}
        
        # Check that cluster 0 elements come first
        first_two = permutation[:2]
        last_two = permutation[2:]
        assert set(first_two) == {0, 1}
        assert set(last_two) == {2, 3}
    
    def test_construct_permutation_with_missing_indices(self):
        """Test permutation construction with missing cluster indices."""
        # Case where some indices are missing from clustering
        cluster_labels = np.array([0, 0, 1])  # Missing index 3
        
        permutation = self.optimizer._construct_permutation(cluster_labels, num_clusters=2, dimension=4)
        
        # Check that all indices are present
        assert len(permutation) == 4
        assert set(permutation) == {0, 1, 2, 3}
    
    def test_construct_permutation_invalid(self):
        """Test permutation construction with invalid inputs."""
        # This should raise an error
        cluster_labels = np.array([0, 0, 1])
        
        with pytest.raises(IterativeCoDesignError):
            self.optimizer._construct_permutation(cluster_labels, num_clusters=2, dimension=2)
    
    def test_compute_permutation_simple(self):
        """Test end-to-end permutation computation."""
        permutation, info = self.optimizer.compute_permutation(
            self.simple_corr, num_clusters=2, correlation_threshold=0.5
        )
        
        # Check permutation
        assert len(permutation) == 4
        assert set(permutation) == {0, 1, 2, 3}
        
        # Check info
        assert 'num_clusters_used' in info
        assert 'silhouette_score' in info
        assert 'graph_edges' in info
        assert 'graph_density' in info
        assert 'dimension' in info
        assert info['dimension'] == 4
    
    def test_compute_permutation_large_dimension(self):
        """Test permutation computation with large dimension triggering block-wise."""
        # Create a large correlation matrix
        large_corr = torch.randn(5000, 5000)
        large_corr = large_corr @ large_corr.T
        diag_sqrt = torch.sqrt(torch.diag(large_corr))
        large_corr = large_corr / diag_sqrt[:, None] / diag_sqrt[None, :]
        
        permutation, info = self.optimizer.compute_permutation(
            large_corr, num_clusters=8, correlation_threshold=0.1
        )
        
        # Check permutation
        assert len(permutation) == 5000
        assert set(permutation) == set(range(5000))
        
        # Check that block-wise method was used
        assert info['block_size'] is not None
        assert 'block_info' in info
    
    def test_compute_blockwise_permutation(self):
        """Test block-wise permutation computation."""
        permutation, info = self.optimizer._compute_blockwise_permutation(
            self.large_corr.numpy(), num_clusters=4, correlation_threshold=0.1, block_size=50
        )
        
        # Check permutation
        assert len(permutation) == 100
        assert set(permutation) == set(range(100))
        
        # Check info
        assert info['method'] == 'blockwise_spectral'
        assert info['num_blocks'] == 2  # 100 / 50 = 2
        assert info['block_size'] == 50
        assert 'block_info' in info
        assert len(info['block_info']) == 2
    
    def test_validate_permutation(self):
        """Test permutation validation."""
        # Valid permutation
        valid_perm = np.array([2, 0, 1, 3])
        assert self.optimizer.validate_permutation(valid_perm) is True
        
        # Invalid permutation - duplicate indices
        invalid_perm1 = np.array([0, 0, 1, 2])
        assert self.optimizer.validate_permutation(invalid_perm1) is False
        
        # Invalid permutation - missing indices
        invalid_perm2 = np.array([0, 1, 3])
        assert self.optimizer.validate_permutation(invalid_perm2) is False
        
        # Invalid permutation - out of range
        invalid_perm3 = np.array([0, 1, 2, 5])
        assert self.optimizer.validate_permutation(invalid_perm3) is False
        
        # Empty permutation
        empty_perm = np.array([])
        assert self.optimizer.validate_permutation(empty_perm) is False
    
    def test_compute_permutation_with_sparse(self):
        """Test permutation computation with sparse matrices."""
        permutation, info = self.optimizer.compute_permutation(
            self.simple_corr, num_clusters=2, correlation_threshold=0.5, use_sparse=True
        )
        
        # Check permutation
        assert len(permutation) == 4
        assert set(permutation) == {0, 1, 2, 3}
        
        # Check that sparse flag is recorded
        assert info['use_sparse'] is True
    
    def test_compute_permutation_edge_cases(self):
        """Test permutation computation edge cases."""
        # Single dimension
        single_corr = torch.tensor([[1.0]])
        permutation, info = self.optimizer.compute_permutation(single_corr, num_clusters=1)
        assert len(permutation) == 1
        assert permutation[0] == 0
        
        # Two dimensions
        two_corr = torch.tensor([[1.0, 0.5], [0.5, 1.0]])
        permutation, info = self.optimizer.compute_permutation(two_corr, num_clusters=2)
        assert len(permutation) == 2
        assert set(permutation) == {0, 1}
    
    def test_compute_permutation_with_block_size(self):
        """Test permutation computation with explicit block size."""
        permutation, info = self.optimizer.compute_permutation(
            self.large_corr, num_clusters=4, correlation_threshold=0.1, block_size=30
        )
        
        # Check permutation
        assert len(permutation) == 100
        assert set(permutation) == set(range(100))
        
        # Check that block-wise method was used
        assert info['block_size'] == 30


if __name__ == '__main__':
    pytest.main([__file__, '-v'])