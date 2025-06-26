"""
Model wrapper utilities for permutation and optimization.

This module provides a wrapper class for PyTorch models that enables weight
permutation and other model transformations required for iterative co-design.
The wrapper maintains model functionality while allowing for structural changes
to optimize memory layout and hardware efficiency.

Key components:
- ModelWrapper: Main wrapper class for model permutation operations
- Weight permutation utilities that preserve model semantics
- Device management and tensor handling for wrapped models
"""

import torch
import torch.nn as nn
from typing import List
import logging

logger = logging.getLogger(__name__)


class ModelWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.device = (
            next(model.parameters()).device
            if list(model.parameters())
            else torch.device("cpu")
        )

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def permute_model_weights(self, permutation: List[int]):
        """
        Permutes the weights of the model's linear layers and their corresponding
        biases according to the given permutation.

        Args:
            permutation: List of integers representing the new ordering of dimensions

        Raises:
            ValueError: If permutation length doesn't match any layer dimensions
        """
        perm_tensor = torch.LongTensor(permutation).to(self.device)
        d_model = len(permutation)

        # Create a map of layers to permute and validate dimensions
        layers_to_permute = {}
        valid_dimensions = set()
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                layers_to_permute[name] = ("linear", module)
                valid_dimensions.add(module.weight.shape[0])  # output features
                valid_dimensions.add(module.weight.shape[1])  # input features
            elif isinstance(module, nn.Embedding):
                layers_to_permute[name] = ("embedding", module)
                valid_dimensions.add(module.weight.shape[1])  # hidden dim
            elif isinstance(module, nn.LayerNorm):
                layers_to_permute[name] = ("layernorm", module)
                valid_dimensions.add(module.weight.shape[0])

        # Validate that permutation length matches at least one layer dimension
        if d_model not in valid_dimensions:
            raise ValueError(
                f"Permutation length {d_model} doesn't match any layer dimensions. "
                f"Valid dimensions in model: {sorted(valid_dimensions)}"
            )

        already_permuted = set()

        for name, (layer_type, layer) in layers_to_permute.items():
            # Detect shared parameters to avoid double permutation (e.g., tied
            # token embedding and LM head weight)
            param_id = id(layer.weight)
            if param_id in already_permuted:
                continue

            if layer_type == "linear":
                # Permute columns (input features)
                if layer.weight.shape[1] == d_model:
                    layer.weight.data = layer.weight.data[:, perm_tensor]
                    already_permuted.add(param_id)
                    logger.info(f"  - Permuted columns of layer: {name}")

                # Permute rows (output features)
                if layer.weight.shape[0] == d_model:
                    layer.weight.data = layer.weight.data[perm_tensor, :]
                    already_permuted.add(param_id)
                    logger.info(f"  - Permuted rows of layer: {name}")
                    # Permute corresponding bias if it exists
                    if layer.bias is not None:
                        layer.bias.data = layer.bias.data[perm_tensor]
                        already_permuted.add(id(layer.bias))
                        logger.info(f"  - Permuted bias of layer: {name}")
            elif layer_type == "embedding":
                # Weight shape (vocab, hidden) – permute hidden dimension (columns)
                if layer.weight.shape[1] == d_model:
                    layer.weight.data = layer.weight.data[:, perm_tensor]
                    already_permuted.add(param_id)
                    logger.info(f"  - Permuted columns of embedding: {name}")
            elif layer_type == "layernorm":
                # LayerNorm weight & bias are 1-D of length hidden_size
                if layer.weight.shape[0] == d_model:
                    layer.weight.data = layer.weight.data[perm_tensor]
                    layer.bias.data = layer.bias.data[perm_tensor]
                    already_permuted.add(param_id)
                    logger.info(f"  - Permuted LayerNorm parameters: {name}")

    def cuda(self, *args, **kwargs):
        self.device = torch.device("cuda")
        self.model.cuda(*args, **kwargs)
        return self

    def cpu(self, *args, **kwargs):
        self.device = torch.device("cpu")
        self.model.cpu(*args, **kwargs)
        return self

    # ------------------------------------------------------------------
    # Attribute delegation helpers
    # ------------------------------------------------------------------

    @property
    def config(self):
        """Expose the underlying model's config (e.g., for hidden_size access)."""
        return getattr(self.model, "config", None)
