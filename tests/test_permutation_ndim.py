"""Test N-dimensional tensor support in permutation utilities."""

import pytest
import torch
import torch.nn as nn

from src.utils.permutation import (
    safe_permute_rows,
    safe_permute_cols,
    safe_permute_vector,
)


class TestNDimensionalPermutation:
    """Test suite for N-dimensional tensor permutation support."""

    def test_3d_conv_weight_permutation(self):
        """Test permuting 3D convolution weights (out_ch, in_ch, kernel_size)."""
        out_ch, in_ch, k = 64, 32, 5
        conv_weight = torch.randn(out_ch, in_ch, k)
        original_data = conv_weight.clone()
        
        # Create permutation
        perm = torch.randperm(out_ch)
        perm_inv = torch.argsort(perm)
        
        # Apply permutation
        safe_permute_rows(conv_weight, perm)
        
        # Verify shape preserved
        assert conv_weight.shape == (out_ch, in_ch, k)
        
        # Verify permutation is correct
        for i in range(out_ch):
            torch.testing.assert_close(
                conv_weight[i], 
                original_data[perm[i]],
                msg=f"Row {i} not correctly permuted"
            )
        
        # Verify inverse permutation works
        safe_permute_rows(conv_weight, perm_inv)
        torch.testing.assert_close(conv_weight, original_data)

    def test_4d_conv2d_weight_permutation(self):
        """Test permuting 4D Conv2d weights (out_ch, in_ch, H, W)."""
        out_ch, in_ch, h, w = 32, 16, 3, 3
        conv2d_weight = torch.randn(out_ch, in_ch, h, w)
        original_data = conv2d_weight.clone()
        
        # Permute output channels (rows)
        perm = torch.randperm(out_ch)
        safe_permute_rows(conv2d_weight, perm)
        
        assert conv2d_weight.shape == (out_ch, in_ch, h, w)
        
        # Verify first dimension is permuted
        for i in range(out_ch):
            torch.testing.assert_close(
                conv2d_weight[i], 
                original_data[perm[i]]
            )

    def test_column_permutation_3d(self):
        """Test column permutation on 3D tensors."""
        shape = (10, 20, 5)
        tensor = torch.randn(*shape)
        original = tensor.clone()
        
        # Permute second dimension (columns in 2D sense)
        perm = torch.randperm(20)
        safe_permute_cols(tensor, perm)
        
        assert tensor.shape == shape
        
        # Verify second dimension is permuted
        for j in range(20):
            torch.testing.assert_close(
                tensor[:, j, :],
                original[:, perm[j], :]
            )

    def test_1d_vector_unchanged(self):
        """Test that 1D vector permutation still works as before."""
        size = 100
        vec = torch.randn(size)
        original = vec.clone()
        
        perm = torch.randperm(size)
        safe_permute_vector(vec, perm)
        
        assert vec.shape == (size,)
        
        for i in range(size):
            assert vec[i] == original[perm[i]]

    def test_invalid_dimensions(self):
        """Test appropriate errors for invalid dimensions."""
        # Column permutation needs at least 2D
        with pytest.raises(AssertionError):
            vec = torch.randn(10)
            safe_permute_cols(vec, torch.randperm(10))
        
        # Vector permutation needs exactly 1D
        with pytest.raises(AssertionError):
            mat = torch.randn(10, 10)
            safe_permute_vector(mat, torch.randperm(10))

    def test_nn_parameter_support(self):
        """Test that nn.Parameter objects work correctly."""
        # Create a mock conv layer
        conv = nn.Conv1d(32, 64, kernel_size=5)
        original_weight = conv.weight.data.clone()
        
        # Permute output channels
        perm = torch.randperm(64)
        safe_permute_rows(conv.weight, perm)
        
        # Verify it's still a Parameter
        assert isinstance(conv.weight, nn.Parameter)
        
        # Verify permutation applied
        for i in range(64):
            torch.testing.assert_close(
                conv.weight.data[i],
                original_weight[perm[i]]
            )

    def test_device_consistency(self):
        """Test permutation works across different devices."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Test GPU tensor with CPU permutation indices
        gpu_tensor = torch.randn(32, 16, 5).cuda()
        cpu_perm = torch.randperm(32)  # CPU indices
        
        safe_permute_rows(gpu_tensor, cpu_perm)
        
        assert gpu_tensor.is_cuda
        assert gpu_tensor.shape == (32, 16, 5)

    def test_gradient_preservation(self):
        """Test that gradient computation still works after permutation."""
        # Create tensor that requires grad
        tensor = torch.randn(10, 20, requires_grad=True)
        original = tensor.clone()
        
        # Permute
        perm = torch.randperm(10)
        safe_permute_rows(tensor, perm)
        
        # Compute some loss and gradient
        loss = tensor.sum()
        loss.backward()
        
        # Gradient should exist and have correct shape
        assert tensor.grad is not None
        assert tensor.grad.shape == tensor.shape 