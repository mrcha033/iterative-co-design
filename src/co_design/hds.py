"""
Hardware-Native Differentiable Sparsity (HDS) module.

This module implements Hardware-Native Differentiable Sparsity, which learns
N:M structured sparsity patterns using the Gumbel-Top-K reparameterization trick.
HDS enables end-to-end training while ensuring hardware compatibility and
creates opportunities for subsequent memory layout optimization.

Key components:
- gumbel_topk: Differentiable Top-K selection using Gumbel-Softmax
- HDSLinear: Linear layer wrapper with learnable N:M sparsity masks
- apply_hds: Apply HDS to model layers and fine-tune sparsity masks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import fnmatch
import logging

logger = logging.getLogger(__name__)

# Constants for HDS algorithm
DEFAULT_N_SPARSITY = 2  # N in N:M sparsity
DEFAULT_M_SPARSITY = 4  # M in N:M sparsity
DEFAULT_GUMBEL_TEMPERATURE = 1.0
DEFAULT_FINE_TUNING_EPOCHS = 1
DEFAULT_LEARNING_RATE = 1e-5
GUMBEL_EPSILON = 1e-10  # For numerical stability in Gumbel sampling


def gumbel_topk(logits: torch.Tensor, k: int, temperature: float = DEFAULT_GUMBEL_TEMPERATURE) -> torch.Tensor:
    """
    Differentiable Top-K selection using the Gumbel-Softmax trick.

    Args:
        logits: A tensor of raw scores (..., N).
        k: The number of items to select from N.
        temperature: The Gumbel-Softmax temperature. A lower temperature makes the
                     selection closer to a true one-hot encoding.

    Returns:
        A tensor of the same shape as logits with a binary mask of K selected items.
        The gradients can flow back to the original logits.
    """
    # Gumbel-Softmax sampling
    gumbels = -torch.log(-torch.log(torch.rand_like(logits) + GUMBEL_EPSILON) + GUMBEL_EPSILON)
    gumbels = (logits + gumbels) / temperature
    y_soft = F.softmax(gumbels, dim=-1)

    # Straight-Through Estimator for Top-K
    # Get the top-k indices from the softened probabilities
    _, top_k_indices = torch.topk(y_soft, k, dim=-1)
    # Create a hard, binary mask
    y_hard = torch.zeros_like(logits).scatter_(-1, top_k_indices, 1.0)

    # Combine the hard mask with the soft probabilities for gradient flow
    y_out = y_hard - y_soft.detach() + y_soft
    return y_out


class HDSLinear(nn.Module):
    """
    A wrapper for a linear layer that applies Hardware-Native Differentiable Sparsity (HDS)
    using the Gumbel-Top-K trick for N:M structured sparsity.
    """

    def __init__(
        self, linear_layer: nn.Linear, n: int = DEFAULT_N_SPARSITY, m: int = DEFAULT_M_SPARSITY, gumbel_temp: float = DEFAULT_GUMBEL_TEMPERATURE
    ):
        super().__init__()
        self.linear = linear_layer
        self.n = n
        self.m = m
        self.gumbel_temp = gumbel_temp

        # Determine padding
        self.in_features = linear_layer.in_features
        self.padding = (self.m - (self.in_features % self.m)) % self.m

        # Scores are created for the original, unpadded dimension
        self.scores = nn.Parameter(
            torch.randn(
                self.linear.out_features, self.in_features, device=self.linear.weight.device
            )
        )

    def get_sparsity_mask(self):
        """
        Generates the N:M structured sparsity mask from the learnable scores.
        """
        # Pad the scores dynamically to ensure divisibility by M
        padded_scores = F.pad(self.scores, (0, self.padding))

        # Reshape so that each group of size `m` is processed independently
        grouped_scores = padded_scores.view(self.linear.out_features, -1, self.m)

        # Use the differentiable Gumbel-TopK to obtain the mask
        mask = gumbel_topk(grouped_scores, self.n, temperature=self.gumbel_temp)

        # Reshape and crop the mask back to (out_features, in_features)
        mask = mask.view(self.linear.out_features, -1)
        return mask[:, : self.in_features]

    def forward(self, x):
        sparsity_mask = self.get_sparsity_mask()
        sparse_weight = self.linear.weight * sparsity_mask
        return F.linear(x, sparse_weight, self.linear.bias)

    # ------------------------------------------------------------------
    # Expose weight / bias so that upstream code that expects a bare
    # nn.Linear (e.g., Transformers fast-path checks) continues to work.
    # ------------------------------------------------------------------

    @property
    def weight(self):  # type: ignore
        return self.linear.weight

    @weight.setter  # type: ignore
    def weight(self, val):
        self.linear.weight = val

    @property
    def bias(self):  # type: ignore
        return self.linear.bias

    @bias.setter  # type: ignore
    def bias(self, val):
        self.linear.bias = val


def _replace_linear_with_hds(model: nn.Module, hds_config: dict):
    """
    Recursively finds and replaces nn.Linear layers with HDSLinear wrappers
    based on wildcard patterns in the configuration.
    """
    target_patterns = hds_config.get("target_layers", [])
    if not target_patterns:
        logger.warning(
            "No target_layers specified for HDS. No layers will be replaced."
        )
        return

    n = hds_config.get("n", DEFAULT_N_SPARSITY)
    m = hds_config.get("m", DEFAULT_M_SPARSITY)

    # Find all linear layers that match the target patterns and have NOT been
    # wrapped previously.  We mark wrapped layers with a private attribute so
    # that subsequent calls (e.g. iterative co-design) do not wrap again.
    layers_to_replace = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        # Skip if this linear layer is already inside an HDS wrapper
        if getattr(module, "_hds_wrapped", False):
            continue

        # Check pattern match
        if any(fnmatch.fnmatch(name, pattern) for pattern in target_patterns):
            layers_to_replace.append(name)

    for name in layers_to_replace:
        # Get the parent module and the name of the child attribute
        name_parts = name.split(".")
        parent_name = ".".join(name_parts[:-1])
        child_name = name_parts[-1]

        parent_module = model
        if parent_name:
            # Use get_submodule to access nested modules
            parent_module = model.get_submodule(parent_name)

        original_layer = getattr(parent_module, child_name)
        hds_layer = HDSLinear(original_layer, n=n, m=m)
        setattr(parent_module, child_name, hds_layer)

        # Mark the underlying linear layer so future passes ignore it
        hds_layer.linear._hds_wrapped = True

        logger.info(f"  - Wrapped layer: {name} with {n}:{m} HDSLinear")


def apply_hds(wrapped_model: "ModelWrapper", data_loader: torch.utils.data.DataLoader, config: dict):
    """
    Applies HDS to the model by replacing target linear layers and fine-tuning.
    """
    logger.info(">>> Applying HDS by replacing Linear layers and fine-tuning...")
    
    model = wrapped_model.model
    hds_config = config # config is already the 'hds' sub-config
    
    _replace_linear_with_hds(model, hds_config)

    # Freeze all model parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze only the 'scores' parameters in HDSLinear layers
    scores_params = []
    for module in model.modules():
        if isinstance(module, HDSLinear):
            module.scores.requires_grad = True
            scores_params.append(module.scores)

    if not scores_params:
        logger.warning("No HDS 'scores' parameters found to fine-tune. Skipping.")
        return wrapped_model

    optimizer = torch.optim.AdamW(scores_params, lr=hds_config.get("learning_rate", DEFAULT_LEARNING_RATE))
    num_epochs = hds_config.get("fine_tuning_epochs", DEFAULT_FINE_TUNING_EPOCHS)

    device = next(model.parameters()).device
    model.train()

    for epoch in range(num_epochs):
        logger.info(f"  - HDS Fine-tuning Epoch {epoch + 1}/{num_epochs}")
        for batch in tqdm(data_loader, desc="Fine-tuning"):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch.get("labels", input_ids).to(device)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(scores_params, max_norm=1.0)

            optimizer.step()

    model.eval()
    logger.info(">>> HDS application complete.")
    return wrapped_model
