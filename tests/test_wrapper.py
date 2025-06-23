import torch
import torch.nn as nn
from models.wrapper import ModelWrapper


class SimpleModel(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        self.embedding = nn.Embedding(100, d_model)
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, 10)
        self.config = type("Config", (), {"hidden_size": d_model})()

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = torch.relu(self.linear1(x))
        return {"logits": self.linear2(x)}


class TestModelWrapper:
    def test_wrapper_initialization(self):
        """Test ModelWrapper initialization."""
        model = SimpleModel()
        wrapped = ModelWrapper(model)

        assert wrapped.model is model
        assert hasattr(wrapped, "device")

    def test_wrapper_forward(self):
        """Test that wrapped model forward works correctly."""
        model = SimpleModel()
        wrapped = ModelWrapper(model)

        input_ids = torch.randint(0, 100, (2, 10))
        output = wrapped(input_ids)

        assert "logits" in output
        assert output["logits"].shape == (2, 10, 10)

    def test_permute_model_weights(self):
        """Test weight permutation functionality."""
        model = SimpleModel(d_model=8)  # Small model for testing
        wrapped = ModelWrapper(model)

        # Get original weights
        original_linear1_weight = wrapped.model.linear1.weight.data.clone()

        # Create a simple permutation
        permutation = [7, 6, 5, 4, 3, 2, 1, 0]  # Reverse order

        # Apply permutation
        wrapped.permute_model_weights(permutation)

        # Check that weights were actually changed
        new_linear1_weight = wrapped.model.linear1.weight.data

        # The weights should be different (unless by coincidence they're symmetric)
        # Check that the permutation was applied correctly to at least one dimension
        assert not torch.allclose(original_linear1_weight, new_linear1_weight)

    def test_permute_preserves_model_functionality(self):
        """Test that permutation preserves model functionality."""
        model = SimpleModel(d_model=8)
        wrapped = ModelWrapper(model)

        # Test input
        input_ids = torch.randint(0, 100, (1, 5))

        # Get output before permutation
        output_before = wrapped(input_ids)

        # Apply permutation
        permutation = list(
            range(8)
        )  # Identity permutation (should not change anything)
        wrapped.permute_model_weights(permutation)

        # Get output after permutation
        output_after = wrapped(input_ids)

        # With identity permutation, outputs should be the same
        assert torch.allclose(
            output_before["logits"], output_after["logits"], atol=1e-6
        )

    def test_permutation_validation(self):
        """Test that invalid permutations are properly rejected."""
        model = SimpleModel(d_model=8)
        wrapped = ModelWrapper(model)

        # Test with wrong length permutation
        invalid_permutation = [0, 1, 2, 3]  # Too short for d_model=8

        # Invalid permutation should raise an error
        try:
            wrapped.permute_model_weights(invalid_permutation)
            # If we get here without an exception, the test should fail
            assert False, "Expected an error for invalid permutation length, but none was raised"
        except (IndexError, RuntimeError, ValueError):
            # This is the expected behavior for invalid permutation
            pass

    def test_device_handling(self):
        """Test device handling in ModelWrapper."""
        model = SimpleModel()
        wrapped = ModelWrapper(model)

        # Should start on CPU
        assert wrapped.device == torch.device("cpu")

        # Test CPU method
        wrapped.cpu()
        assert wrapped.device == torch.device("cpu")

        # Test CUDA if available
        if torch.cuda.is_available():
            wrapped.cuda()
            assert wrapped.device == torch.device("cuda")

    def test_wrapper_with_bias(self):
        """Test permutation with models that have bias terms."""

        class ModelWithBias(nn.Module):
            def __init__(self, d_model=8):
                super().__init__()
                self.linear = nn.Linear(d_model, d_model, bias=True)
                self.config = type("Config", (), {"hidden_size": d_model})()

            def forward(self, x):
                return self.linear(x)

        model = ModelWithBias()
        wrapped = ModelWrapper(model)

        # Store original bias
        original_bias = wrapped.model.linear.bias.data.clone()

        # Apply permutation
        permutation = [7, 6, 5, 4, 3, 2, 1, 0]
        wrapped.permute_model_weights(permutation)

        # Bias should also be permuted for output features
        assert not torch.allclose(original_bias, wrapped.model.linear.bias.data)
        new_bias = wrapped.model.linear.bias.data

        # Should be different (unless symmetric)
        # At minimum, check that the function didn't crash
        assert new_bias.shape == original_bias.shape
