#!/usr/bin/env python3
"""Test that Mamba dimension inference fix works correctly."""

import sys

def test_config_override():
    """Test that _maybe_override_feature_dim_from_config always prefers config.hidden_size."""
    from icd.core.graph_pytorch import _maybe_override_feature_dim_from_config

    # Mock model with config
    class MockConfig:
        hidden_size = 2560
        intermediate_size = 5120

    class MockModel:
        config = MockConfig()

    model = MockModel()

    # Test cases where FX inferred wrong dimension
    test_cases = [
        (4096, 2560, "Should override 4096 to 2560"),
        (5120, 2560, "Should override 5120 to 2560"),
        (2048, 2560, "Should override 2048 to 2560"),
        (256, 2560, "Should override 256 to 2560"),
        (2560, 2560, "Should keep 2560 when already correct"),
        (0, 2560, "Should override 0 to 2560"),
        (-1, 2560, "Should override -1 to 2560"),
    ]

    print("Testing _maybe_override_feature_dim_from_config...")
    all_passed = True

    for current_dim, expected, description in test_cases:
        result_dim, source = _maybe_override_feature_dim_from_config(model, current_dim)
        if result_dim == expected and source == "hf_config.hidden_size":
            print(f"  ✓ {description}: {current_dim} -> {result_dim}")
        else:
            print(f"  ✗ {description}: expected {expected}, got {result_dim} (source: {source})")
            all_passed = False

    # Test without config
    class ModelNoConfig:
        pass

    model_no_config = ModelNoConfig()
    result_dim, source = _maybe_override_feature_dim_from_config(model_no_config, 1024)
    if result_dim == 1024 and source is None:
        print(f"  ✓ Model without config: preserves current_dim {result_dim}")
    else:
        print(f"  ✗ Model without config: expected 1024/None, got {result_dim}/{source}")
        all_passed = False

    if all_passed:
        print("\nAll tests passed!")
        return 0
    else:
        print("\nSome tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(test_config_override())