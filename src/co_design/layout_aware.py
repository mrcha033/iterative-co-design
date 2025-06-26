"""
Layout-Aware Co-Design Module.

This module extends the HDS (Hardware-Native Differentiable Sparsity) algorithm
by incorporating a layout-aware regularization term into the fine-tuning process.
This represents the "reverse arrow" proposed in the paper, where the algorithmic
optimization (sparsity) is made aware of the hardware layout (permutation).

This functionality is separated for future work and is not used in the main
experiments of the initial paper.

Key components:
- apply_layout_aware_hds_finetuning: Fine-tunes a model with HDS wrappers using a
  layout-aware regularization loss.
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import logging

# Import components from the main HDS module
from .hds import HDSLinear, DEFAULT_FINE_TUNING_EPOCHS, DEFAULT_LEARNING_RATE

logger = logging.getLogger(__name__)


def calculate_layout_regularization(
    hds_layer: HDSLinear, permutation: torch.Tensor, cluster_size: int
) -> torch.Tensor:
    """
    Calculates a regularization term that penalizes sparsity masks that break clusters.

    The goal is to encourage the sparsity mask to preserve connections within the
    high-modularity clusters identified by IASP.

    Args:
        hds_layer: The HDSLinear layer to analyze.
        permutation: The current memory permutation from IASP.
        cluster_size: The size of clusters to preserve.

    Returns:
        A scalar tensor representing the regularization loss.
    """
    mask = hds_layer.get_sparsity_mask()

    # Permute the mask according to the memory layout
    perm_mask = mask[:, permutation]

    # Reshape the permuted mask into clusters
    num_clusters = perm_mask.shape[1] // cluster_size
    if num_clusters == 0:
        return torch.tensor(0.0, device=mask.device) # Not enough features for a full cluster

    clustered_mask = perm_mask[:, : num_clusters * cluster_size].view(
        perm_mask.shape[0], num_clusters, cluster_size
    )

    # Calculate the density of sparse elements within each cluster
    cluster_density = torch.mean(clustered_mask, dim=2)

    # The regularization loss is the negative of the mean density.
    # Minimizing this loss is equivalent to maximizing the cluster density.
    regularization_loss = -torch.mean(cluster_density)

    return regularization_loss


def apply_layout_aware_hds_finetuning(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    config: dict,
    permutation: torch.Tensor,
):
    """
    Fine-tunes a model already wrapped with HDS layers using a layout-aware
    regularization term.

    Args:
        model: The PyTorch model (must have HDSLinear layers).
        data_loader: DataLoader for fine-tuning.
        config: Configuration dictionary.
        permutation: The current memory permutation from IASP.
    """
    logger.info(">>> Fine-tuning with Layout-Aware HDS Regularization...")

    hds_config = config.get("hds", {})
    dataset_lr = config.get("dataset", {}).get("learning_rate")
    lr = dataset_lr if dataset_lr is not None else config.get("learning_rate", DEFAULT_LEARNING_RATE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    num_epochs = hds_config.get("fine_tuning_epochs", DEFAULT_FINE_TUNING_EPOCHS)
    
    # Hyperparameters for the layout-aware regularization
    reg_lambda = hds_config.get("layout_aware_lambda", 0.01)
    cluster_size = hds_config.get("cluster_size", 64)

    device = next(model.parameters()).device
    model.train()

    for epoch in range(num_epochs):
        logger.info(f"  - Layout-Aware Fine-tuning Epoch {epoch + 1}/{num_epochs}")
        for batch in tqdm(data_loader, desc="Layout-Aware Fine-tuning"):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch.get("labels", input_ids).to(device)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            task_loss = outputs.loss

            # Calculate and add the layout-aware regularization loss
            layout_reg_loss = 0
            num_hds_layers = 0
            for module in model.modules():
                if isinstance(module, HDSLinear):
                    layout_reg_loss += calculate_layout_regularization(
                        module, permutation, cluster_size
                    )
                    num_hds_layers += 1
            
            if num_hds_layers > 0:
                total_loss = task_loss + reg_lambda * (layout_reg_loss / num_hds_layers)
            else:
                total_loss = task_loss

            total_loss.backward()
            optimizer.step()

    model.eval()
    logger.info(">>> Layout-Aware HDS fine-tuning complete.")
    return model
