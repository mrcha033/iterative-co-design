"""
Unit tests for permutation application implementation.
"""
import numpy as np
import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch

from src.co_design.apply import PermutationApplicator
from src.models.permutable_model import PermutableModel
from src.utils.exceptions import PermutationApplicationError


class TestPermutationApplicator:
    """Test the PermutationApplicator class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Create a mock permutable model
        self.mock_model = Mock(spec=PermutableModel)
        self.mock_model.model_type = 'test'
        
        # Create test layers
        self.linear_layer = nn.Linear(4, 3)
        self.conv1d_layer = nn.Conv1d(4, 3, kernel_size=3)
        self.conv2d_layer = nn.Conv2d(4, 3, kernel_size=3)
        
        # Initialize weights for predictable testing
        with torch.no_grad():
            self.linear_layer.weight.fill_(1.0)
            self.linear_layer.bias.fill_(0.1)
            self.conv1d_layer.weight.fill_(2.0)
            self.conv1d_layer.bias.fill_(0.2)
            self.conv2d_layer.weight.fill_(3.0)
            self.conv2d_layer.bias.fill_(0.3)
        
        # Setup model mock methods
        self.mock_model.get_layer.return_value = self.linear_layer
        self.mock_model.get_layer_info.return_value = {'type': 'linear', 'shape': (3, 4)}
        
        self.applicator = PermutationApplicator(self.mock_model)
    
    def test_init(self):
        """Test applicator initialization."""
        assert self.applicator.model == self.mock_model
        assert len(self.applicator.applied_permutations) == 0
    
    def test_validate_permutation_valid(self):
        """Test permutation validation with valid permutation."""
        permutation = np.array([1, 0, 2, 3])
        layer_info = {'type': 'linear', 'shape': (3, 4)}
        
        result = self.applicator._validate_permutation(
            self.linear_layer, permutation, 'input', layer_info
        )
        
        assert result['valid'] is True
        assert result['expected_size'] == 4
        assert result['weight_shape'] == (3, 4)
        assert result['dimension'] == 'input'
    
    def test_validate_permutation_invalid_size(self):
        """Test permutation validation with wrong size."""
        permutation = np.array([1, 0, 2])  # Too small
        layer_info = {'type': 'linear', 'shape': (3, 4)}
        
        result = self.applicator._validate_permutation(
            self.linear_layer, permutation, 'input', layer_info
        )
        
        assert result['valid'] is False
        assert 'size' in result['error']
    
    def test_validate_permutation_invalid_indices(self):
        """Test permutation validation with invalid indices."""
        permutation = np.array([1, 1, 2, 3])  # Duplicate
        layer_info = {'type': 'linear', 'shape': (3, 4)}
        
        result = self.applicator._validate_permutation(
            self.linear_layer, permutation, 'input', layer_info
        )
        
        assert result['valid'] is False
        assert 'exactly once' in result['error']
    
    def test_validate_permutation_out_of_bounds(self):
        """Test permutation validation with out-of-bounds indices."""
        permutation = np.array([1, 0, 2, 5])  # 5 is out of bounds
        layer_info = {'type': 'linear', 'shape': (3, 4)}
        
        result = self.applicator._validate_permutation(
            self.linear_layer, permutation, 'input', layer_info
        )
        
        assert result['valid'] is False
        assert 'out-of-bounds' in result['error']
    
    def test_validate_permutation_no_weight(self):
        """Test permutation validation with layer without weight."""
        layer_without_weight = nn.ReLU()
        layer_info = {'type': 'activation', 'shape': None}
        
        result = self.applicator._validate_permutation(
            layer_without_weight, np.array([1, 0]), 'input', layer_info
        )
        
        assert result['valid'] is False
        assert 'no weight' in result['error']
    
    def test_validate_permutation_dimensions(self):
        """Test permutation validation for different dimensions."""
        permutation = np.array([1, 0, 2])
        layer_info = {'type': 'linear', 'shape': (3, 4)}
        
        # Test output dimension
        result = self.applicator._validate_permutation(
            self.linear_layer, permutation, 'output', layer_info
        )
        assert result['valid'] is True
        assert result['expected_size'] == 3
        
        # Test both dimension with non-square matrix
        result = self.applicator._validate_permutation(
            self.linear_layer, permutation, 'both', layer_info
        )
        assert result['valid'] is False
        assert 'non-square' in result['error']
        
        # Test invalid dimension
        result = self.applicator._validate_permutation(
            self.linear_layer, permutation, 'invalid', layer_info
        )
        assert result['valid'] is False
        assert 'Invalid dimension' in result['error']
    
    def test_apply_linear_permutation_input(self):
        """Test applying permutation to linear layer input dimension."""
        permutation = np.array([3, 1, 0, 2])
        original_weight = self.linear_layer.weight.clone()
        
        result = self.applicator._apply_linear_permutation(
            self.linear_layer, permutation, 'input'
        )
        
        assert result is True
        
        # Check that input features were permuted
        expected_weight = original_weight[:, permutation]
        assert torch.allclose(self.linear_layer.weight, expected_weight)
        
        # Bias should be unchanged
        assert torch.allclose(self.linear_layer.bias, torch.full((3,), 0.1))
    
    def test_apply_linear_permutation_output(self):
        """Test applying permutation to linear layer output dimension."""
        permutation = np.array([2, 0, 1])
        original_weight = self.linear_layer.weight.clone()
        original_bias = self.linear_layer.bias.clone()
        
        result = self.applicator._apply_linear_permutation(
            self.linear_layer, permutation, 'output'
        )
        
        assert result is True
        
        # Check that output features were permuted
        expected_weight = original_weight[permutation, :]
        expected_bias = original_bias[permutation]
        assert torch.allclose(self.linear_layer.weight, expected_weight)
        assert torch.allclose(self.linear_layer.bias, expected_bias)
    
    def test_apply_linear_permutation_both(self):
        """Test applying permutation to both dimensions of square linear layer."""
        # Create a square linear layer
        square_layer = nn.Linear(3, 3)
        with torch.no_grad():
            square_layer.weight.fill_(1.0)
            square_layer.bias.fill_(0.1)
        
        permutation = np.array([2, 0, 1])
        original_weight = square_layer.weight.clone()
        original_bias = square_layer.bias.clone()
        
        result = self.applicator._apply_linear_permutation(
            square_layer, permutation, 'both'
        )
        
        assert result is True
        
        # Check that both dimensions were permuted
        expected_weight = original_weight[permutation, :][:, permutation]
        expected_bias = original_bias[permutation]
        assert torch.allclose(square_layer.weight, expected_weight)
        assert torch.allclose(square_layer.bias, expected_bias)
    
    def test_apply_conv1d_permutation_input(self):
        """Test applying permutation to conv1d layer input channels."""
        permutation = np.array([3, 1, 0, 2])
        original_weight = self.conv1d_layer.weight.clone()
        
        result = self.applicator._apply_conv1d_permutation(
            self.conv1d_layer, permutation, 'input'
        )
        
        assert result is True
        
        # Check that input channels were permuted
        expected_weight = original_weight[:, permutation, :]
        assert torch.allclose(self.conv1d_layer.weight, expected_weight)
    
    def test_apply_conv1d_permutation_output(self):
        """Test applying permutation to conv1d layer output channels."""
        permutation = np.array([2, 0, 1])
        original_weight = self.conv1d_layer.weight.clone()
        original_bias = self.conv1d_layer.bias.clone()
        
        result = self.applicator._apply_conv1d_permutation(
            self.conv1d_layer, permutation, 'output'
        )
        
        assert result is True
        
        # Check that output channels were permuted
        expected_weight = original_weight[permutation, :, :]
        expected_bias = original_bias[permutation]
        assert torch.allclose(self.conv1d_layer.weight, expected_weight)
        assert torch.allclose(self.conv1d_layer.bias, expected_bias)
    
    def test_apply_conv2d_permutation_input(self):
        """Test applying permutation to conv2d layer input channels."""
        permutation = np.array([3, 1, 0, 2])
        original_weight = self.conv2d_layer.weight.clone()
        
        result = self.applicator._apply_conv2d_permutation(
            self.conv2d_layer, permutation, 'input'
        )
        
        assert result is True
        
        # Check that input channels were permuted
        expected_weight = original_weight[:, permutation, :, :]
        assert torch.allclose(self.conv2d_layer.weight, expected_weight)
    
    def test_apply_conv2d_permutation_output(self):
        """Test applying permutation to conv2d layer output channels."""
        permutation = np.array([2, 0, 1])
        original_weight = self.conv2d_layer.weight.clone()
        original_bias = self.conv2d_layer.bias.clone()
        
        result = self.applicator._apply_conv2d_permutation(
            self.conv2d_layer, permutation, 'output'
        )
        
        assert result is True
        
        # Check that output channels were permuted
        expected_weight = original_weight[permutation, :, :, :]
        expected_bias = original_bias[permutation]
        assert torch.allclose(self.conv2d_layer.weight, expected_weight)
        assert torch.allclose(self.conv2d_layer.bias, expected_bias)
    
    def test_apply_generic_permutation(self):
        """Test applying permutation to generic layer."""
        # Create a custom layer
        custom_layer = nn.Module()
        custom_layer.weight = nn.Parameter(torch.ones(3, 4))
        custom_layer.bias = nn.Parameter(torch.full((3,), 0.1))
        
        permutation = np.array([3, 1, 0, 2])
        original_weight = custom_layer.weight.clone()
        
        result = self.applicator._apply_generic_permutation(
            custom_layer, permutation, 'input'
        )
        
        assert result is True
        
        # Check that input dimension was permuted
        expected_weight = original_weight[:, permutation]
        assert torch.allclose(custom_layer.weight, expected_weight)
    
    def test_apply_generic_permutation_no_weight(self):
        """Test applying permutation to layer without weight."""
        layer_without_weight = nn.ReLU()
        
        result = self.applicator._apply_generic_permutation(
            layer_without_weight, np.array([1, 0]), 'input'
        )
        
        assert result is False
    
    def test_identify_affected_layers_mamba(self):
        """Test identifying affected layers for Mamba architecture."""
        self.mock_model.model_type = 'mamba'
        
        # Mock get_layer to return different layers
        def mock_get_layer(name):
            if 'in_proj' in name:
                return nn.Linear(4, 3)
            elif 'out_proj' in name:
                return nn.Linear(3, 4)
            else:
                raise ValueError(f"Layer {name} not found")
        
        self.mock_model.get_layer.side_effect = mock_get_layer
        
        affected = self.applicator._identify_mamba_affected_layers('layers.0.mixer.in_proj')
        
        # Should find related layers
        assert 'layers.0.in_proj' in affected
        assert 'layers.0.out_proj' in affected
    
    def test_identify_affected_layers_bert(self):
        """Test identifying affected layers for BERT architecture."""
        self.mock_model.model_type = 'bert'
        
        # Mock get_layer to return different layers
        def mock_get_layer(name):
            if any(x in name for x in ['query', 'key', 'value', 'dense']):
                return nn.Linear(4, 3)
            else:
                raise ValueError(f"Layer {name} not found")
        
        self.mock_model.get_layer.side_effect = mock_get_layer
        
        affected = self.applicator._identify_bert_affected_layers('layers.0.attention.query')
        
        # Should find related attention layers
        assert 'layers.0.query' in affected
        assert 'layers.0.key' in affected
        assert 'layers.0.value' in affected
        assert 'layers.0.dense' in affected
    
    def test_apply_permutation_success(self):
        """Test successful permutation application."""
        permutation = np.array([1, 0, 2, 3])
        layer_info = {'type': 'linear', 'shape': (3, 4)}
        
        self.mock_model.get_layer.return_value = self.linear_layer
        self.mock_model.get_layer_info.return_value = layer_info
        
        result = self.applicator.apply_permutation(
            'test_layer', permutation, 'input', validate=True, dry_run=False
        )
        
        assert result['success'] is True
        assert result['layer_name'] == 'test_layer'
        assert result['permutation_size'] == 4
        assert result['dimension'] == 'input'
        assert 'test_layer' in self.applicator.applied_permutations
    
    def test_apply_permutation_dry_run(self):
        """Test dry run permutation application."""
        permutation = np.array([1, 0, 2, 3])
        layer_info = {'type': 'linear', 'shape': (3, 4)}
        
        self.mock_model.get_layer.return_value = self.linear_layer
        self.mock_model.get_layer_info.return_value = layer_info
        
        result = self.applicator.apply_permutation(
            'test_layer', permutation, 'input', validate=True, dry_run=True
        )
        
        assert result['dry_run'] is True
        assert result['layer_name'] == 'test_layer'
        assert 'test_layer' not in self.applicator.applied_permutations
    
    def test_apply_permutation_validation_failure(self):
        """Test permutation application with validation failure."""
        permutation = np.array([1, 0, 2])  # Wrong size
        layer_info = {'type': 'linear', 'shape': (3, 4)}
        
        self.mock_model.get_layer.return_value = self.linear_layer
        self.mock_model.get_layer_info.return_value = layer_info
        
        with pytest.raises(PermutationApplicationError):
            self.applicator.apply_permutation(
                'test_layer', permutation, 'input', validate=True, dry_run=False
            )
    
    def test_apply_permutation_layer_not_found(self):
        """Test permutation application when layer is not found."""
        permutation = np.array([1, 0, 2, 3])
        
        self.mock_model.get_layer.side_effect = ValueError("Layer not found")
        
        with pytest.raises(PermutationApplicationError):
            self.applicator.apply_permutation(
                'nonexistent_layer', permutation, 'input'
            )
    
    def test_apply_permutation_tensor_input(self):
        """Test permutation application with tensor input."""
        permutation = torch.tensor([1, 0, 2, 3])
        layer_info = {'type': 'linear', 'shape': (3, 4)}
        
        self.mock_model.get_layer.return_value = self.linear_layer
        self.mock_model.get_layer_info.return_value = layer_info
        
        result = self.applicator.apply_permutation(
            'test_layer', permutation, 'input', validate=True, dry_run=False
        )
        
        assert result['success'] is True
        assert 'test_layer' in self.applicator.applied_permutations
    
    def test_get_applied_permutations(self):
        """Test getting applied permutations."""
        permutation = np.array([1, 0, 2, 3])
        self.applicator.applied_permutations['test_layer'] = permutation
        
        applied = self.applicator.get_applied_permutations()
        
        assert 'test_layer' in applied
        assert np.array_equal(applied['test_layer'], permutation)
        
        # Should be a copy
        applied['test_layer'] = np.array([0, 1, 2, 3])
        assert not np.array_equal(self.applicator.applied_permutations['test_layer'], applied['test_layer'])
    
    def test_has_permutation(self):
        """Test checking if layer has permutation."""
        permutation = np.array([1, 0, 2, 3])
        self.applicator.applied_permutations['test_layer'] = permutation
        
        assert self.applicator.has_permutation('test_layer') is True
        assert self.applicator.has_permutation('other_layer') is False
    
    def test_reset_permutations(self):
        """Test resetting applied permutations."""
        permutation = np.array([1, 0, 2, 3])
        self.applicator.applied_permutations['test_layer'] = permutation
        
        self.applicator.reset_permutations()
        
        assert len(self.applicator.applied_permutations) == 0
    
    def test_get_permutation_summary(self):
        """Test getting permutation summary."""
        permutation1 = np.array([1, 0, 2, 3])
        permutation2 = np.array([2, 1, 0])
        
        self.applicator.applied_permutations['layer1'] = permutation1
        self.applicator.applied_permutations['layer2'] = permutation2
        
        summary = self.applicator.get_permutation_summary()
        
        assert summary['num_permuted_layers'] == 2
        assert set(summary['permuted_layers']) == {'layer1', 'layer2'}
        assert summary['total_permutations'] == 7  # 4 + 3
    
    def test_apply_to_single_layer_with_warning(self):
        """Test applying permutation to single layer with shape warning."""
        # Create a layer that will change shape (shouldn't happen in practice)
        mock_layer = Mock()
        mock_layer.weight = Mock()
        mock_layer.weight.shape = (3, 4)
        
        self.mock_model.get_layer.return_value = mock_layer
        
        # Mock the apply method to change shape
        def mock_apply(layer, perm, dim):
            layer.weight.shape = (4, 3)  # Changed shape
            return True
        
        with patch.object(self.applicator, '_apply_linear_permutation', mock_apply):
            with patch('warnings.warn') as mock_warn:
                result = self.applicator._apply_to_single_layer(
                    'test_layer', np.array([1, 0, 2, 3]), 'input', {}
                )
                
                # Should warn about shape change
                mock_warn.assert_called_once()
                assert 'shape changed' in mock_warn.call_args[0][0].lower()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])