"""Test IASP safety features and configurations."""

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.utils.config import validate_config
from src.utils.permutation import (
    safe_permute_rows,
    safe_permute_cols,
    permute_optimizer_state,
)
from src.co_design.iasp_rollback import (
    IASPRollbackManager,
    PermutationCheckpoint,
)


class TestIASPConfigSafety:
    """Test configuration safety checks."""
    
    def test_permute_gate_locked_false(self):
        """Test that permute_gate=True is rejected."""
        config = {
            'model': {'name': 'test'},
            'dataset': {'name': 'test'},
            'iasp': {
                'permute_gate': True  # Should fail
            }
        }
        
        with pytest.raises(ValueError, match="permute_gate.*must be False"):
            validate_config(config)
    
    def test_permute_gate_false_accepted(self):
        """Test that permute_gate=False is accepted."""
        config = {
            'model': {'name': 'test'},
            'dataset': {'name': 'test'},
            'iasp': {
                'permute_gate': False  # Should pass
            }
        }
        
        assert validate_config(config) is True
    
    def test_iasp_parameter_validation(self):
        """Test validation of IASP parameters."""
        # Invalid cluster_size_range
        config = {
            'model': {'name': 'test'},
            'dataset': {'name': 'test'},
            'iasp': {
                'cluster_size_range': [128, 32]  # min > max
            }
        }
        
        with pytest.raises(ValueError, match="cluster_size_range.*min < max"):
            validate_config(config)
        
        # Invalid max_samples
        config['iasp'] = {'max_samples': -1}
        with pytest.raises(ValueError, match="max_samples.*positive"):
            validate_config(config)
        
        # Invalid max_ppl_increase
        config['iasp'] = {'max_ppl_increase': 1.5}  # > 1
        with pytest.raises(ValueError, match="max_ppl_increase.*between 0 and 1"):
            validate_config(config)


class TestPermutationSafety:
    """Test permutation safety mechanisms."""
    
    def test_tensorimpl_preservation(self):
        """Test that TensorImpl is preserved during permutation."""
        # Create a parameter
        param = nn.Parameter(torch.randn(10, 20))
        original_data_ptr = param.data.data_ptr()
        
        # Apply permutation
        perm = torch.randperm(10)
        safe_permute_rows(param, perm)
        
        # Check that data_ptr is the same (same TensorImpl)
        assert param.data.data_ptr() == original_data_ptr, \
            "TensorImpl should be preserved"
        
        # Verify it's still a Parameter
        assert isinstance(param, nn.Parameter)
    
    def test_optimizer_state_preservation(self):
        """Test that optimizer state is preserved and correctly permuted."""
        # Create a simple model
        model = nn.Linear(10, 20)
        optimizer = torch.optim.Adam(model.parameters())
        
        # Run one step to create optimizer state
        x = torch.randn(5, 10)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        
        # Get state before permutation
        state_before = optimizer.state[model.weight]
        exp_avg_before = state_before['exp_avg'].clone()
        
        # Apply permutation
        perm = torch.randperm(20)
        safe_permute_rows(model.weight, perm)
        permute_optimizer_state(optimizer, model.weight, perm, axis=0)
        
        # Verify state still exists
        assert model.weight in optimizer.state
        
        # Verify state was permuted
        state_after = optimizer.state[model.weight]
        for i in range(20):
            torch.testing.assert_close(
                state_after['exp_avg'][i],
                exp_avg_before[perm[i]]
            )
    
    def test_gradient_preservation(self):
        """Test that gradients work correctly after permutation."""
        # Create tensor with gradient
        x = torch.randn(10, 20, requires_grad=True)
        
        # Compute gradient
        y = x.sum()
        y.backward()
        grad_before = x.grad.clone()
        
        # Apply permutation
        perm = torch.randperm(10)
        safe_permute_rows(x, perm)
        
        # Clear and recompute gradient
        x.grad = None
        y2 = x.sum()
        y2.backward()
        
        # Gradient should exist
        assert x.grad is not None
        assert x.requires_grad is True


class TestRollbackMechanism:
    """Test IASP rollback functionality."""
    
    @pytest.fixture
    def mock_model_and_data(self):
        """Create mock model and data for testing."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )
        
        # Mock dataloader - just returns fixed data
        class MockDataLoader:
            def __iter__(self):
                for _ in range(5):
                    yield torch.randn(32, 10), torch.randint(0, 10, (32,))
        
        return model, MockDataLoader()
    
    def test_rollback_manager_creation(self, mock_model_and_data):
        """Test creating a rollback manager."""
        model, dataloader = mock_model_and_data
        
        manager = IASPRollbackManager(
            model=model,
            eval_dataloader=dataloader,
            max_ppl_increase=0.05
        )
        
        assert manager.baseline_ppl is None
        assert len(manager.checkpoints) == 0
        assert manager.original_state is not None
    
    def test_checkpoint_addition(self, mock_model_and_data):
        """Test adding checkpoints."""
        model, dataloader = mock_model_and_data
        manager = IASPRollbackManager(model, dataloader)
        
        # Add a checkpoint
        perm = torch.randperm(20)
        manager.add_checkpoint(
            layer_name="layer1",
            permutation=perm,
            modularity=0.15,
            affected_params={"weight": model[0].weight}
        )
        
        assert len(manager.checkpoints) == 1
        checkpoint = manager.checkpoints[0]
        assert checkpoint.layer_name == "layer1"
        assert checkpoint.modularity == 0.15
        assert torch.equal(checkpoint.permutation, perm)
        assert torch.equal(checkpoint.inverse_permutation, torch.argsort(perm))
    
    def test_full_rollback(self, mock_model_and_data):
        """Test full rollback to original state."""
        model, dataloader = mock_model_and_data
        manager = IASPRollbackManager(model, dataloader)
        
        # Store original weights
        original_weight = model[0].weight.data.clone()
        
        # Modify the model
        model[0].weight.data.add_(1.0)
        
        # Verify it changed
        assert not torch.equal(model[0].weight.data, original_weight)
        
        # Rollback
        manager.full_rollback()
        
        # Verify it's restored
        torch.testing.assert_close(model[0].weight.data, original_weight)
        assert len(manager.checkpoints) == 0


class TestAxisAwarePermutation:
    """Test axis-aware permutation with validation."""
    
    def test_axis_validation(self):
        """Test axis validation for different tensor shapes."""
        from src.utils.permutation import permute_tensor_axis
        
        # 3D tensor
        tensor = torch.randn(10, 20, 30)
        perm = torch.randperm(20)
        
        # Valid axis
        permute_tensor_axis(tensor, perm, axis=1)
        
        # Invalid axis
        with pytest.raises(ValueError, match="out of bounds"):
            permute_tensor_axis(tensor, perm, axis=3)
    
    def test_custom_validator(self):
        """Test custom axis validator."""
        from src.utils.permutation import permute_tensor_axis
        
        # Validator that only allows axis 0
        def only_axis_0(tensor, axis):
            return axis == 0
        
        tensor = torch.randn(10, 20)
        perm = torch.randperm(10)
        
        # Should work for axis 0
        permute_tensor_axis(tensor, perm, axis=0, axis_validator=only_axis_0)
        
        # Should fail for axis 1
        perm2 = torch.randperm(20)
        with pytest.raises(ValueError, match="validation failed"):
            permute_tensor_axis(tensor, perm2, axis=1, axis_validator=only_axis_0) 