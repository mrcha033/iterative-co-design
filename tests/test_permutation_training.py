"""Test that permutations preserve training dynamics."""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.utils.permutation import (
    safe_permute_rows,
    safe_permute_cols,
    permute_optimizer_state,
)


class SimpleModel(nn.Module):
    """Simple model for testing permutation effects on training."""
    
    def __init__(self, input_dim=10, hidden_dim=20, output_dim=5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class TestPermutationTraining:
    """Test suite for verifying training works after permutations."""
    
    @pytest.fixture
    def setup_model_and_data(self):
        """Create a simple model and synthetic dataset."""
        torch.manual_seed(42)
        
        # Model
        model = SimpleModel()
        
        # Synthetic data
        X = torch.randn(100, 10)
        y = torch.randint(0, 5, (100,))
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
        
        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        return model, dataloader, optimizer
    
    def test_loss_decreases_after_permutation(self, setup_model_and_data):
        """Test that loss still decreases after applying permutations."""
        model, dataloader, optimizer = setup_model_and_data
        criterion = nn.CrossEntropyLoss()
        
        # Train for a few steps to build optimizer state
        initial_losses = []
        for epoch in range(2):
            for batch_idx, (data, target) in enumerate(dataloader):
                if batch_idx >= 5:  # Just a few batches
                    break
                    
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                initial_losses.append(loss.item())
        
        # Store loss before permutation
        pre_perm_loss = initial_losses[-1]
        
        # Apply permutation to hidden layer
        hidden_dim = model.fc1.out_features
        perm = torch.randperm(hidden_dim)
        perm_inv = torch.argsort(perm)
        
        # Permute fc1 output features and fc2 input features
        safe_permute_rows(model.fc1.weight, perm)
        safe_permute_cols(model.fc2.weight, perm_inv)
        
        # Permute optimizer state
        permute_optimizer_state(optimizer, model.fc1.weight, perm, axis=0)
        permute_optimizer_state(optimizer, model.fc2.weight, perm_inv, axis=1)
        
        # Continue training after permutation
        post_perm_losses = []
        for epoch in range(2):
            for batch_idx, (data, target) in enumerate(dataloader):
                if batch_idx >= 5:
                    break
                    
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                post_perm_losses.append(loss.item())
        
        # Verify loss continues to decrease
        assert post_perm_losses[-1] < pre_perm_loss, \
            f"Loss should decrease after permutation: {pre_perm_loss:.4f} -> {post_perm_losses[-1]:.4f}"
        
        # Verify gradients flow correctly
        assert all(p.grad is not None for p in model.parameters()), \
            "All parameters should have gradients"
    
    def test_optimizer_state_consistency(self, setup_model_and_data):
        """Test that optimizer state is correctly permuted."""
        model, dataloader, optimizer = setup_model_and_data
        criterion = nn.CrossEntropyLoss()
        
        # Train to build optimizer state
        for data, target in dataloader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            break  # Just one step
        
        # Check optimizer state exists
        assert model.fc1.weight in optimizer.state
        state = optimizer.state[model.fc1.weight]
        
        # For Adam, we should have exp_avg and exp_avg_sq
        assert 'exp_avg' in state
        assert 'exp_avg_sq' in state
        
        # Store original state values
        orig_exp_avg = state['exp_avg'].clone()
        orig_exp_avg_sq = state['exp_avg_sq'].clone()
        
        # Apply permutation
        perm = torch.randperm(model.fc1.out_features)
        safe_permute_rows(model.fc1.weight, perm)
        permute_optimizer_state(optimizer, model.fc1.weight, perm, axis=0)
        
        # Verify state was permuted
        for i in range(len(perm)):
            torch.testing.assert_close(
                state['exp_avg'][i],
                orig_exp_avg[perm[i]],
                msg="exp_avg not correctly permuted"
            )
            torch.testing.assert_close(
                state['exp_avg_sq'][i],
                orig_exp_avg_sq[perm[i]],
                msg="exp_avg_sq not correctly permuted"
            )
    
    def test_functional_equivalence_preserved(self, setup_model_and_data):
        """Test that model output is unchanged after consistent permutation."""
        model, _, _ = setup_model_and_data
        model.eval()
        
        # Test input
        x = torch.randn(5, 10)
        
        # Get output before permutation
        with torch.no_grad():
            output_before = model(x).clone()
        
        # Apply consistent permutation to hidden layer
        hidden_dim = model.fc1.out_features
        perm = torch.randperm(hidden_dim)
        perm_inv = torch.argsort(perm)
        
        # Permute weights and biases consistently
        safe_permute_rows(model.fc1.weight, perm)
        safe_permute_cols(model.fc2.weight, perm_inv)
        if model.fc1.bias is not None:
            model.fc1.bias.data = model.fc1.bias.data[perm]
        
        # Get output after permutation
        with torch.no_grad():
            output_after = model(x)
        
        # Outputs should be identical
        torch.testing.assert_close(
            output_before,
            output_after,
            rtol=1e-5,
            atol=1e-5,
            msg="Model output changed after permutation"
        )
    
    def test_gradient_flow_with_permutation(self):
        """Test that gradients flow correctly through permuted layers."""
        torch.manual_seed(42)
        
        # Create a deeper model to test gradient flow
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 30),
            nn.ReLU(),
            nn.Linear(30, 40),
            nn.ReLU(),
            nn.Linear(40, 5)
        )
        
        # Apply permutations to middle layers
        perm = torch.randperm(30)
        perm_inv = torch.argsort(perm)
        
        safe_permute_rows(model[2].weight, perm)  # Linear(20, 30)
        safe_permute_cols(model[4].weight, perm_inv)  # Linear(30, 40)
        
        # Forward pass
        x = torch.randn(10, 10, requires_grad=True)
        output = model(x)
        loss = output.sum()
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist and are non-zero
        for param in model.parameters():
            assert param.grad is not None, "All parameters should have gradients"
            assert param.grad.abs().sum() > 0, "Gradients should be non-zero"
        
        # Check input gradient
        assert x.grad is not None, "Input should have gradient"
        assert x.grad.abs().sum() > 0, "Input gradient should be non-zero" 