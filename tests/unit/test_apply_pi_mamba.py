#!/usr/bin/env python3
"""Unit tests for Mamba permutation application logic."""

import pytest
import torch
from unittest.mock import Mock, MagicMock, patch


def test_mamba_conv1d_permutation_rows_only():
    """Test that Mamba Conv1d permutation only reindexes rows (output channels), not cols."""
    from icd.runtime.apply_pi import apply_pi_to_mamba_hf

    # Create mock projections - these are the "A", "B", "C" modules
    mock_in_proj = Mock()
    mock_in_proj.weight = torch.randn(1024, 256)  # intermediate*2 x hidden
    mock_in_proj.bias = torch.randn(1024)

    mock_x_proj = Mock()
    mock_x_proj.weight = torch.randn(80, 512)  # (dt_rank + state*2) x intermediate
    mock_x_proj.bias = None

    mock_out_proj = Mock()
    mock_out_proj.weight = torch.randn(256, 512)  # hidden x intermediate
    mock_out_proj.bias = torch.randn(256)

    # Create mock Conv1d with depthwise shape [out_channels=512, in_channels=1, kernel=4]
    mock_conv = Mock()
    mock_conv.weight = Mock()
    mock_conv.weight.data = torch.randn(512, 1, 4)
    mock_conv.bias = Mock()
    mock_conv.bias.data = torch.randn(512)

    mock_dt_proj = Mock()
    mock_dt_proj.weight = torch.randn(512, 16)  # intermediate x dt_rank
    mock_dt_proj.bias = torch.randn(512)

    # Mock SSM parameters
    mock_A_log = torch.randn(512, 16)
    mock_D = torch.randn(512)

    # Create module reference
    mock_module_ref = Mock()
    mock_module_ref.conv1d = mock_conv
    mock_module_ref.dt_proj = mock_dt_proj
    mock_module_ref.A_log = mock_A_log
    mock_module_ref.D = mock_D

    # Mock module entry with required "A", "B", "C" structure
    mock_module_entry = {
        "A": mock_in_proj,  # in_proj
        "B": mock_x_proj,   # x_proj
        "C": mock_out_proj, # out_proj
        "_module_name": "backbone.layers.0.mixer",
        "_hf_mamba": True,
        "_module_ref": mock_module_ref,
    }

    # Create permutation for hidden_size (256)
    pi_hidden = torch.randperm(256)

    # Store original shapes
    orig_conv_shape = mock_conv.weight.data.shape

    # Apply permutation
    apply_pi_to_mamba_hf(mock_module_entry, pi_hidden)

    # Verify Conv1d shape unchanged (only rows permuted, not cols)
    assert mock_conv.weight.data.shape == orig_conv_shape, \
        f"Conv1d shape changed from {orig_conv_shape} to {mock_conv.weight.data.shape}"

    # Specifically check that in_channels=1 remained (didn't try to broadcast to 512)
    assert mock_conv.weight.data.shape[1] == 1, \
        f"Conv1d in_channels changed from 1 to {mock_conv.weight.data.shape[1]}"


def test_mamba_module_collection_logging(caplog):
    """Test that module collection has proper logging for diagnostics."""
    from icd.runtime.runners_hf import _collect_mamba_modules_from_model

    # Create mock Mamba mixer with ONLY HF structure (not A, B, C)
    mock_mixer = Mock(spec=['in_proj', 'x_proj', 'out_proj', '__class__'])
    mock_mixer.__class__.__name__ = "MambaMixer"

    # Add required attributes for HF Mamba
    mock_mixer.in_proj = Mock()
    mock_mixer.in_proj.weight = torch.randn(1024, 256)
    mock_mixer.x_proj = Mock()
    mock_mixer.x_proj.weight = torch.randn(80, 512)
    mock_mixer.out_proj = Mock()
    mock_mixer.out_proj.weight = torch.randn(256, 512)

    # Mock model with named_modules that returns the mixer
    mock_model = Mock()
    mock_model.named_modules = Mock(return_value=[
        ("backbone.layers.0.mixer", mock_mixer)
    ])

    # Collect modules
    with caplog.at_level("INFO"):
        modules = _collect_mamba_modules_from_model(mock_model)

    # Should find 1 module (or 0 if wrapping fails, which is fine for this test)
    # The important part is that logging occurred
    assert any("mamba" in record.message.lower() for record in caplog.records), \
        "Expected logging about Mamba modules"


def test_mamba_permutation_dimension_validation():
    """Test that permutation dimension is validated against model config."""
    from icd.runtime.apply_pi import apply_pi_to_mamba_hf, PermutationApplicationError

    # Create mock projections
    mock_in_proj = Mock()
    mock_in_proj.weight = torch.randn(1024, 256)  # intermediate*2 x hidden
    mock_in_proj.bias = torch.randn(1024)

    mock_x_proj = Mock()
    mock_x_proj.weight = torch.randn(80, 512)
    mock_x_proj.bias = None

    mock_out_proj = Mock()
    mock_out_proj.weight = torch.randn(256, 512)
    mock_out_proj.bias = torch.randn(256)

    # Create module reference
    mock_module_ref = Mock()
    mock_module_ref.conv1d = Mock()
    mock_module_ref.conv1d.weight = Mock()
    mock_module_ref.conv1d.weight.data = torch.randn(512, 1, 4)
    mock_module_ref.conv1d.bias = Mock()
    mock_module_ref.conv1d.bias.data = torch.randn(512)

    mock_module_ref.dt_proj = Mock()
    mock_module_ref.dt_proj.weight = torch.randn(512, 16)
    mock_module_ref.dt_proj.bias = torch.randn(512)

    mock_module_ref.A_log = torch.randn(512, 16)
    mock_module_ref.D = torch.randn(512)

    # Mock module entry with required structure
    mock_module_entry = {
        "A": mock_in_proj,
        "B": mock_x_proj,
        "C": mock_out_proj,
        "_module_name": "backbone.layers.0.mixer",
        "_hf_mamba": True,
        "_module_ref": mock_module_ref,
    }

    # Test with wrong dimension (should raise error with diagnostic message)
    wrong_pi = torch.randperm(4096)  # Sequence length, not hidden/intermediate

    with pytest.raises(PermutationApplicationError) as exc_info:
        apply_pi_to_mamba_hf(mock_module_entry, wrong_pi)

    # Check that error message is helpful
    error_msg = str(exc_info.value)
    assert "4096" in error_msg
    assert "256" in error_msg or "512" in error_msg  # Should mention expected dimensions


def test_correlation_expected_dim_parameter():
    """Test that ActivationCollector accepts and uses expected_dim parameter."""
    from icd.graph.correlation import ActivationCollector

    # Create simple mock model
    mock_model = Mock()
    mock_model.named_modules = Mock(return_value=[])

    # Create collector with expected_dim
    collector = ActivationCollector(
        model=mock_model,
        targets=[],
        dtype=torch.float32,
        expected_dim=2560
    )

    # Verify parameter is stored
    assert collector.expected_dim == 2560


def test_dimension_inference_with_mamba_config():
    """Test that feature dimension inference prefers model.config.hidden_size."""
    from icd.core.graph_pytorch import _maybe_override_feature_dim_from_config

    # Mock Mamba model config
    class MockConfig:
        hidden_size = 2560
        intermediate_size = 5120

    class MockModel:
        config = MockConfig()

    model = MockModel()

    # Test that sequence length (4096) is overridden to hidden_size (2560)
    result_dim, source = _maybe_override_feature_dim_from_config(model, 4096)
    assert result_dim == 2560, f"Expected 2560, got {result_dim}"
    assert source == "hf_config.hidden_size", f"Expected config source, got {source}"

    # Test that intermediate_size (5120) is also overridden to hidden_size
    result_dim, source = _maybe_override_feature_dim_from_config(model, 5120)
    assert result_dim == 2560, f"Expected 2560, got {result_dim}"

    # Test that 0 or invalid dimensions are overridden
    result_dim, source = _maybe_override_feature_dim_from_config(model, 0)
    assert result_dim == 2560, f"Expected 2560, got {result_dim}"


def test_reindex_rows_helper():
    """Test the reindex_rows helper function used for permutation."""
    from icd.runtime.apply_pi import reindex_rows

    # Create test tensor
    W = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ])

    # Permutation that reverses rows
    pi = torch.tensor([2, 1, 0])

    # Apply permutation
    W_permuted = reindex_rows(W, pi)

    # Check result
    expected = torch.tensor([
        [7.0, 8.0, 9.0],
        [4.0, 5.0, 6.0],
        [1.0, 2.0, 3.0],
    ])

    assert torch.allclose(W_permuted, expected), \
        f"Expected {expected}, got {W_permuted}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
