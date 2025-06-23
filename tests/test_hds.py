"""
Tests for Hardware-Native Differentiable Sparsity (HDS) module.
"""

import torch
import torch.nn as nn
import pytest
from unittest.mock import MagicMock, patch
from torch.utils.data import DataLoader, TensorDataset

from src.co_design.hds import gumbel_topk, HDSLinear, apply_hds, _replace_linear_with_hds


class TestGumbelTopK:
    """Test cases for the gumbel_topk function."""

    def test_gumbel_topk_shape(self):
        """Test that gumbel_topk returns correct shape."""
        logits = torch.randn(4, 10)
        k = 3
        result = gumbel_topk(logits, k)
        assert result.shape == logits.shape

    def test_gumbel_topk_selection_count(self):
        """Test that gumbel_topk selects exactly k items."""
        logits = torch.randn(2, 8)
        k = 3
        result = gumbel_topk(logits, k)
        
        # Count selected items (should be exactly k per row)
        selected_counts = result.sum(dim=-1)
        assert torch.allclose(selected_counts, torch.tensor(k, dtype=torch.float))

    def test_gumbel_topk_temperature_effect(self):
        """Test that temperature affects the selection distribution."""
        torch.manual_seed(42)  # For reproducibility
        logits = torch.randn(1, 10)
        k = 3
        
        # Test with different temperatures
        result_low_temp = gumbel_topk(logits, k, temperature=0.1)
        result_high_temp = gumbel_topk(logits, k, temperature=10.0)
        
        # Both should still select exactly k items
        assert torch.allclose(result_low_temp.sum(), torch.tensor(k, dtype=torch.float))
        assert torch.allclose(result_high_temp.sum(), torch.tensor(k, dtype=torch.float))

    def test_gumbel_topk_gradient_flow(self):
        """Test that gradients can flow through gumbel_topk."""
        logits = torch.randn(2, 6, requires_grad=True)
        k = 2
        result = gumbel_topk(logits, k)
        
        # Compute a dummy loss and backpropagate
        loss = result.sum()
        loss.backward()
        
        # Check that gradients exist
        assert logits.grad is not None
        assert logits.grad.shape == logits.shape


class TestHDSLinear:
    """Test cases for the HDSLinear class."""

    def test_hds_linear_initialization(self):
        """Test HDSLinear initialization with different configurations."""
        linear = nn.Linear(12, 8)
        hds_linear = HDSLinear(linear, n=2, m=4)
        
        assert hds_linear.n == 2
        assert hds_linear.m == 4
        assert hds_linear.in_features == 12
        assert hds_linear.padding == 0  # 12 is divisible by 4
        assert hds_linear.scores.shape == (8, 12)

    def test_hds_linear_initialization_with_padding(self):
        """Test HDSLinear initialization when padding is needed."""
        linear = nn.Linear(10, 6)  # 10 is not divisible by 4
        hds_linear = HDSLinear(linear, n=2, m=4)
        
        assert hds_linear.padding == 2  # Need 2 more to make 12
        assert hds_linear.scores.shape == (6, 12)  # Padded dimension

    def test_hds_linear_forward_pass(self):
        """Test forward pass through HDSLinear."""
        linear = nn.Linear(8, 4)
        hds_linear = HDSLinear(linear, n=2, m=4)
        
        x = torch.randn(3, 8)
        output = hds_linear(x)
        
        assert output.shape == (3, 4)

    def test_sparsity_mask_shape(self):
        """Test that sparsity mask has correct shape."""
        linear = nn.Linear(12, 6)
        hds_linear = HDSLinear(linear, n=3, m=4)
        
        mask = hds_linear.get_sparsity_mask()
        assert mask.shape == (6, 12)

    def test_sparsity_mask_n_m_constraint(self):
        """Test that sparsity mask satisfies N:M constraint."""
        linear = nn.Linear(8, 2)
        hds_linear = HDSLinear(linear, n=2, m=4)
        
        mask = hds_linear.get_sparsity_mask()
        
        # Reshape to check N:M constraint
        reshaped_mask = mask.view(2, 2, 4)  # (out_features, groups, m)
        selected_per_group = reshaped_mask.sum(dim=-1)
        
        # Each group should have exactly n=2 selected items
        assert torch.allclose(selected_per_group, torch.tensor(2.0))


class TestApplyHDS:
    """Test cases for apply_hds and related functions."""

    def create_dummy_model(self):
        """Create a dummy model for testing."""
        model = nn.Sequential(
            nn.Linear(10, 8),
            nn.ReLU(),
            nn.Linear(8, 6),
            nn.ReLU(),
            nn.Linear(6, 2)
        )
        return model

    def create_dummy_dataloader(self):
        """Create a dummy dataloader for testing."""
        # Create dummy data in the format expected by apply_hds
        # Use smaller, consistent dimensions for testing
        input_ids = torch.randint(0, 100, (20, 8))  # Match input dimension
        attention_mask = torch.ones(20, 8)
        labels = torch.randint(0, 2, (20, 2))  # Match output dimension
        
        # Create a custom dataset that returns dictionaries
        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self, input_ids, attention_mask, labels):
                self.input_ids = input_ids
                self.attention_mask = attention_mask
                self.labels = labels
            
            def __len__(self):
                return len(self.input_ids)
            
            def __getitem__(self, idx):
                return {
                    'input_ids': self.input_ids[idx],
                    'attention_mask': self.attention_mask[idx],
                    'labels': self.labels[idx]
                }
        
        dataset = DummyDataset(input_ids, attention_mask, labels)
        return DataLoader(dataset, batch_size=4)

    def test_replace_linear_with_hds(self):
        """Test that linear layers are correctly replaced with HDSLinear."""
        model = self.create_dummy_model()
        
        hds_config = {
            "target_layers": ["0", "2"],  # First and third layers
            "n": 2,
            "m": 4
        }
        
        _replace_linear_with_hds(model, hds_config)
        
        # Check that specified layers are now HDSLinear
        assert isinstance(model[0], HDSLinear)
        assert isinstance(model[2], HDSLinear)
        # Layer 4 should remain nn.Linear
        assert isinstance(model[4], nn.Linear)

    def test_replace_linear_with_hds_wildcard_patterns(self):
        """Test wildcard pattern matching for layer replacement."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.ModuleDict({
                    'layer1': nn.Linear(10, 8),
                    'layer2': nn.Linear(8, 6),
                })
                self.decoder = nn.Linear(6, 2)
            
            def forward(self, x):
                x = self.encoder['layer1'](x)
                x = self.encoder['layer2'](x)
                return self.decoder(x)
        
        model = TestModel()
        
        hds_config = {
            "target_layers": ["encoder.*"],  # All encoder layers
            "n": 2,
            "m": 4
        }
        
        _replace_linear_with_hds(model, hds_config)
        
        # Check that encoder layers are replaced
        assert isinstance(model.encoder['layer1'], HDSLinear)
        assert isinstance(model.encoder['layer2'], HDSLinear)
        # Decoder should remain unchanged
        assert isinstance(model.decoder, nn.Linear)

    def test_apply_hds_no_target_layers(self):
        """Test apply_hds with no target layers specified."""
        # Create a mock model that accepts the expected arguments
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(8, 2)  # Match input/output dimensions
            
            def forward(self, input_ids, attention_mask=None, labels=None):
                # Simple mock forward that returns a loss
                output = self.linear(input_ids.float())
                if labels is not None:
                    loss = nn.functional.mse_loss(output, labels.float())
                    return type('obj', (object,), {'loss': loss})()
                return output
        
        model = MockModel()
        original_param = list(model.parameters())[0].clone()
        
        config = {"hds": {}}  # No target_layers
        dataloader = self.create_dummy_dataloader()
        
        with patch('src.co_design.hds.logger') as mock_logger:
            result_model = apply_hds(model, dataloader, config)
            mock_logger.warning.assert_called_once()
        
        # Model should remain unchanged (no HDS applied)
        assert isinstance(result_model.linear, nn.Linear)  # Not HDSLinear

    @patch('src.co_design.hds.tqdm')
    def test_apply_hds_fine_tuning(self, mock_tqdm):
        """Test that apply_hds performs fine-tuning."""
        # Create a simple model with a mock forward method
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(8, 2)  # Match input/output dimensions
            
            def forward(self, input_ids, attention_mask=None, labels=None):
                # Simple mock forward that returns a loss
                output = self.linear(input_ids.float())
                if labels is not None:
                    loss = nn.functional.mse_loss(output, labels.float())
                    return type('obj', (object,), {'loss': loss})()
                return output
        
        model = MockModel()
        
        config = {
            "hds": {
                "target_layers": ["linear"],
                "fine_tuning_epochs": 1,
                "n": 1,
                "m": 2
            },
            "learning_rate": 1e-3
        }
        
        dataloader = self.create_dummy_dataloader()
        
        # Mock tqdm to avoid progress bar output
        mock_tqdm.return_value = dataloader
        
        result_model = apply_hds(model, dataloader, config)
        
        # Check that the model was modified (HDSLinear wrapper added)
        assert isinstance(result_model.linear, HDSLinear)

    def test_apply_hds_with_dataset_learning_rate(self):
        """Test that dataset-specific learning rate is used when available."""
        # Create a mock model that accepts the expected arguments
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(8, 2)  # Match input/output dimensions
            
            def forward(self, input_ids, attention_mask=None, labels=None):
                # Simple mock forward that returns a loss
                output = self.linear(input_ids.float())
                if labels is not None:
                    loss = nn.functional.mse_loss(output, labels.float())
                    return type('obj', (object,), {'loss': loss})()
                return output
        
        model = MockModel()
        
        config = {
            "hds": {
                "target_layers": ["linear"],
                "fine_tuning_epochs": 1
            },
            "learning_rate": 1e-3,
            "dataset": {
                "learning_rate": 1e-4  # This should take precedence
            }
        }
        
        dataloader = self.create_dummy_dataloader()
        
        with patch('torch.optim.AdamW') as mock_optimizer:
            apply_hds(model, dataloader, config)
            # Check that the dataset learning rate was used
            mock_optimizer.assert_called_once()
            args, kwargs = mock_optimizer.call_args
            assert kwargs['lr'] == 1e-4


if __name__ == "__main__":
    pytest.main([__file__]) 