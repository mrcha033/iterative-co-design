"""
Integration tests for permutation application and correctness verification.
"""
import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import tempfile

from src.models.manager import ModelManager
from src.models.permutable_model import PermutableModel
from src.co_design.spectral import SpectralClusteringOptimizer
from src.co_design.apply import PermutationApplicator
from src.co_design.correlation import CorrelationMatrixComputer
from src.co_design.iasp import IASPPermutationOptimizer


class TestPermutationIntegration:
    """Integration tests for end-to-end permutation workflows."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.device = 'cpu'
        
        # Create a simple test model
        self.test_model = self._create_test_model()
        self.permutable_model = PermutableModel(self.test_model, 'test')
        
        # Create test data
        self.test_data = self._create_test_data()
        
        # Initialize components
        self.spectral_optimizer = SpectralClusteringOptimizer(random_state=42)
        self.applicator = PermutationApplicator(self.permutable_model)
        self.correlation_computer = CorrelationMatrixComputer(cache_dir=self.temp_dir)
        self.iasp_optimizer = IASPPermutationOptimizer(
            correlation_computer=self.correlation_computer,
            device=self.device
        )
    
    def _create_test_model(self):
        """Create a simple test model."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(8, 6)
                self.linear2 = nn.Linear(6, 4)
                self.linear3 = nn.Linear(4, 2)
                self.relu = nn.ReLU()
            
            def forward(self, x):
                x = self.relu(self.linear1(x))
                x = self.relu(self.linear2(x))
                x = self.linear3(x)
                return x
        
        return SimpleModel()
    
    def _create_test_data(self):
        """Create test data for the model."""
        # Create synthetic dataset
        X = torch.randn(100, 8)
        y = torch.randint(0, 2, (100,))
        
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False)
        
        return dataloader
    
    def test_end_to_end_permutation_workflow(self):
        """Test complete end-to-end permutation workflow."""
        layer_name = 'linear1'
        
        # Step 1: Generate permutation using IASP
        permutation, optimization_info = self.iasp_optimizer.optimize_permutation(
            model=self.permutable_model,
            dataloader=self.test_data,
            layer_name=layer_name,
            num_clusters=4,
            num_samples=50,
            method='spectral'
        )
        
        # Verify permutation properties
        assert len(permutation) == 8  # Input dimension of linear1
        assert set(permutation) == set(range(8))
        assert optimization_info['method'] == 'spectral'
        assert 'modularity' in optimization_info
        
        # Step 2: Store original weights for comparison
        original_weights = {}
        for name, param in self.test_model.named_parameters():
            original_weights[name] = param.clone()
        
        # Step 3: Apply permutation
        self.applicator.apply_permutation(
            layer_name=layer_name,
            permutation=permutation,
            dimension='input',
            validate=True
        )
        
        # Step 4: Verify permutation was applied
        assert self.applicator.has_permutation(layer_name)
        applied_perm = self.applicator.get_applied_permutations()[layer_name]
        assert np.array_equal(applied_perm, permutation)
        
        # Step 5: Verify weights were correctly permuted
        new_weight = self.test_model.linear1.weight
        expected_weight = original_weights['linear1.weight'][:, permutation]
        assert torch.allclose(new_weight, expected_weight)
        
        # Step 6: Verify model functionality is preserved
        with torch.no_grad():
            test_input = torch.randn(5, 8)
            
            # Apply same permutation to input
            permuted_input = test_input[:, permutation]
            
            # Get output with permuted model and permuted input
            output_permuted = self.test_model(permuted_input)
            
            # Create fresh model with original weights
            fresh_model = self._create_test_model()
            fresh_model.load_state_dict(original_weights)
            
            # Get output with original model and original input
            output_original = fresh_model(test_input)
            
            # Outputs should be identical
            assert torch.allclose(output_permuted, output_original, atol=1e-6)
    
    def test_spectral_clustering_permutation_quality(self):
        """Test that spectral clustering produces reasonable permutations."""
        # Create a correlation matrix with clear block structure
        corr_matrix = torch.zeros(12, 12)
        
        # Create 3 blocks of 4 elements each with high intra-block correlation
        for i in range(3):
            start = i * 4
            end = (i + 1) * 4
            corr_matrix[start:end, start:end] = 0.8
        
        # Add diagonal
        corr_matrix.fill_diagonal_(1.0)
        
        # Add some noise
        corr_matrix += 0.1 * torch.randn_like(corr_matrix)
        corr_matrix = (corr_matrix + corr_matrix.T) / 2  # Make symmetric
        corr_matrix.fill_diagonal_(1.0)
        
        # Compute permutation
        permutation, info = self.spectral_optimizer.compute_permutation(
            corr_matrix, num_clusters=3, correlation_threshold=0.5
        )
        
        # Verify permutation properties
        assert len(permutation) == 12
        assert set(permutation) == set(range(12))
        assert info['num_clusters_used'] <= 3
        assert info['silhouette_score'] >= 0  # Should be positive for good clustering
        
        # Verify block structure is preserved
        permuted_corr = corr_matrix[permutation, :][:, permutation]
        
        # Check that early indices are more correlated with each other
        # than with later indices (indicating good clustering)
        early_block = permuted_corr[:4, :4]
        late_block = permuted_corr[8:12, 8:12]
        cross_block = permuted_corr[:4, 8:12]
        
        assert torch.mean(torch.abs(early_block)) > torch.mean(torch.abs(cross_block))
        assert torch.mean(torch.abs(late_block)) > torch.mean(torch.abs(cross_block))
    
    def test_blockwise_permutation_consistency(self):
        """Test that block-wise permutation is consistent."""
        # Create a large correlation matrix
        large_corr = torch.randn(5000, 5000)
        large_corr = large_corr @ large_corr.T
        
        # Normalize to correlation matrix
        diag_sqrt = torch.sqrt(torch.diag(large_corr))
        large_corr = large_corr / diag_sqrt[:, None] / diag_sqrt[None, :]
        
        # Compute permutation (should trigger block-wise mode)
        permutation, info = self.spectral_optimizer.compute_permutation(
            large_corr, num_clusters=8, correlation_threshold=0.1
        )
        
        # Verify permutation properties
        assert len(permutation) == 5000
        assert set(permutation) == set(range(5000))
        assert info['block_size'] is not None
        assert 'block_info' in info
        
        # Verify that permutation is valid
        assert self.spectral_optimizer.validate_permutation(permutation)
    
    def test_permutation_application_error_handling(self):
        """Test error handling in permutation application."""
        # Test with invalid layer name
        with pytest.raises(Exception):
            self.applicator.apply_permutation(
                layer_name='nonexistent_layer',
                permutation=np.array([1, 0, 2, 3]),
                dimension='input'
            )
        
        # Test with invalid permutation
        with pytest.raises(Exception):
            self.applicator.apply_permutation(
                layer_name='linear1',
                permutation=np.array([1, 1, 2, 3]),  # Invalid: duplicate
                dimension='input'
            )
        
        # Test with wrong size permutation
        with pytest.raises(Exception):
            self.applicator.apply_permutation(
                layer_name='linear1',
                permutation=np.array([1, 0, 2]),  # Wrong size
                dimension='input'
            )
    
    def test_permutation_correctness_verification(self):
        """Test that permutation application preserves model correctness."""
        layer_name = 'linear1'
        permutation = np.array([7, 3, 1, 5, 0, 2, 4, 6])
        
        # Store original model state
        original_state = self.test_model.state_dict()
        
        # Apply permutation
        self.applicator.apply_permutation(
            layer_name=layer_name,
            permutation=permutation,
            dimension='input'
        )
        
        # Test with multiple inputs
        test_inputs = [
            torch.randn(1, 8),
            torch.randn(3, 8),
            torch.randn(10, 8)
        ]
        
        for test_input in test_inputs:
            # Get output from permuted model with permuted input
            permuted_input = test_input[:, permutation]
            output_permuted = self.test_model(permuted_input)
            
            # Restore original model
            self.test_model.load_state_dict(original_state)
            
            # Get output from original model with original input
            output_original = self.test_model(test_input)
            
            # Outputs should be identical
            assert torch.allclose(output_permuted, output_original, atol=1e-6)
            
            # Reapply permutation for next iteration
            self.applicator.reset_permutations()
            self.applicator.apply_permutation(
                layer_name=layer_name,
                permutation=permutation,
                dimension='input'
            )
    
    def test_multiple_layer_permutation(self):
        """Test applying permutations to multiple layers."""
        # Apply permutation to first layer
        perm1 = np.array([7, 3, 1, 5, 0, 2, 4, 6])
        self.applicator.apply_permutation('linear1', perm1, 'input')
        
        # Apply permutation to second layer
        perm2 = np.array([5, 2, 0, 4, 1, 3])
        self.applicator.apply_permutation('linear2', perm2, 'input')
        
        # Verify both permutations are tracked
        assert self.applicator.has_permutation('linear1')
        assert self.applicator.has_permutation('linear2')
        
        summary = self.applicator.get_permutation_summary()
        assert summary['num_permuted_layers'] == 2
        assert summary['total_permutations'] == 14  # 8 + 6
    
    def test_permutation_with_different_layer_types(self):
        """Test permutation application with different layer types."""
        # Create a model with different layer types
        class MixedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 3)
                self.conv1d = nn.Conv1d(4, 3, kernel_size=3)
                self.conv2d = nn.Conv2d(4, 3, kernel_size=3)
            
            def forward(self, x):
                return x  # Dummy forward
        
        mixed_model = MixedModel()
        permutable_mixed = PermutableModel(mixed_model, 'mixed')
        applicator = PermutationApplicator(permutable_mixed)
        
        # Test linear layer permutation
        perm_linear = np.array([3, 1, 0, 2])
        applicator.apply_permutation('linear', perm_linear, 'input')
        
        # Test conv1d layer permutation  
        perm_conv1d = np.array([3, 1, 0, 2])
        applicator.apply_permutation('conv1d', perm_conv1d, 'input')
        
        # Test conv2d layer permutation
        perm_conv2d = np.array([3, 1, 0, 2])
        applicator.apply_permutation('conv2d', perm_conv2d, 'input')
        
        # Verify all permutations were applied
        assert applicator.has_permutation('linear')
        assert applicator.has_permutation('conv1d')
        assert applicator.has_permutation('conv2d')
    
    def test_correlation_matrix_caching(self):
        """Test that correlation matrix caching works correctly."""
        layer_name = 'linear1'
        
        # Compute correlation matrix first time
        corr_matrix1 = self.correlation_computer.compute_correlation_matrix(
            model=self.permutable_model,
            dataloader=self.test_data,
            layer_name=layer_name,
            num_samples=50,
            force_recompute=False
        )
        
        # Compute correlation matrix second time (should use cache)
        corr_matrix2 = self.correlation_computer.compute_correlation_matrix(
            model=self.permutable_model,
            dataloader=self.test_data,
            layer_name=layer_name,
            num_samples=50,
            force_recompute=False
        )
        
        # Should be identical due to caching
        assert torch.allclose(corr_matrix1, corr_matrix2)
        
        # Force recompute should potentially give different results
        corr_matrix3 = self.correlation_computer.compute_correlation_matrix(
            model=self.permutable_model,
            dataloader=self.test_data,
            layer_name=layer_name,
            num_samples=50,
            force_recompute=True
        )
        
        # Should have same shape
        assert corr_matrix1.shape == corr_matrix3.shape
    
    def test_iasp_optimization_methods(self):
        """Test different IASP optimization methods."""
        layer_name = 'linear1'
        
        # Test spectral method
        perm_spectral, info_spectral = self.iasp_optimizer.optimize_permutation(
            model=self.permutable_model,
            dataloader=self.test_data,
            layer_name=layer_name,
            num_clusters=4,
            num_samples=50,
            method='spectral'
        )
        
        # Test TSP method
        perm_tsp, info_tsp = self.iasp_optimizer.optimize_permutation(
            model=self.permutable_model,
            dataloader=self.test_data,
            layer_name=layer_name,
            num_samples=50,
            method='tsp'
        )
        
        # Test random method
        perm_random, info_random = self.iasp_optimizer.optimize_permutation(
            model=self.permutable_model,
            dataloader=self.test_data,
            layer_name=layer_name,
            num_samples=50,
            method='random'
        )
        
        # All should produce valid permutations
        for perm in [perm_spectral, perm_tsp, perm_random]:
            assert len(perm) == 8
            assert set(perm) == set(range(8))
        
        # Spectral should generally have better modularity than random
        # (though this is stochastic, so we just check it's computed)
        assert 'modularity' in info_spectral
        assert 'modularity' in info_tsp
        assert 'modularity' in info_random
    
    def test_permutation_evaluation_metrics(self):
        """Test permutation evaluation metrics."""
        # Create a test correlation matrix
        corr_matrix = torch.randn(8, 8)
        corr_matrix = corr_matrix @ corr_matrix.T
        diag_sqrt = torch.sqrt(torch.diag(corr_matrix))
        corr_matrix = corr_matrix / diag_sqrt[:, None] / diag_sqrt[None, :]
        
        # Test different permutations
        identity_perm = np.arange(8)
        random_perm = np.random.permutation(8)
        
        # Evaluate identity permutation
        metrics_identity = self.iasp_optimizer.evaluate_permutation(
            corr_matrix, identity_perm
        )
        
        # Evaluate random permutation
        metrics_random = self.iasp_optimizer.evaluate_permutation(
            corr_matrix, random_perm
        )
        
        # Check that all metrics are computed
        for metrics in [metrics_identity, metrics_random]:
            assert 'modularity' in metrics
            assert 'locality' in metrics
            assert 'block_coherence' in metrics
            assert 'permutation_length' in metrics
            assert metrics['permutation_length'] == 8


if __name__ == '__main__':
    pytest.main([__file__, '-v'])