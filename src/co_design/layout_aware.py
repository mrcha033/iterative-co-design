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
    hds_layer: HDSLinear,
    permutation: torch.Tensor,
    penalty_fn_type: str = 'inverse' # 'inverse' or 'exp'
) -> torch.Tensor:
    """
    Calculates a regularization term based on the pairwise distance
    between permuted dimensions, directly implementing the paper's proposal.
    """
    mask = hds_layer.get_sparsity_mask() # Shape: (out_features, in_features)
    n_in = mask.shape[1]
    device = mask.device
    
    # 1. 순열에 따른 거리 행렬(distance matrix) 생성
    # inv_permutation을 사용하여 원래 인덱스가 순열 후 어디로 갔는지 찾음
    inv_permutation = torch.argsort(permutation.to(device))
    # 각 위치 i와 j 간의 순열 후 거리 |π(i) - π(j)| 계산
    pos_i = inv_permutation.unsqueeze(1).float()
    pos_j = inv_permutation.unsqueeze(0).float()
    dist_matrix = torch.abs(pos_i - pos_j) # Shape: (in_features, in_features)

    # 2. 페널티 행렬 생성
    if penalty_fn_type == 'inverse':
        # 거리가 가까울수록 페널티가 커짐 (0으로 나누는 것 방지)
        penalty_matrix = 1.0 / (dist_matrix + 1e-6)
    elif penalty_fn_type == 'exp':
        penalty_matrix = torch.exp(-dist_matrix / n_in) # 스케일링
    else:
        raise ValueError("Unknown penalty function type.")
        
    # 3. 정규화 손실 계산
    # (1 - mask)는 "연결이 끊어진 정도"를 나타냄
    # 이를 페널티 행렬과 곱하여, 가까운 연결을 끊을수록 높은 손실을 부여
    # out_features 차원에 대해 평균을 내어 최종 손실 계산
    reg_loss = torch.mean((1 - mask) @ penalty_matrix)
    
    return reg_loss


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
    lr = hds_config.get("learning_rate", DEFAULT_LEARNING_RATE)

    # Freeze all parameters first, then unfreeze only the ones we need to train.
    for param in model.parameters():
        param.requires_grad = False
    
    trainable_params = []
    for module in model.modules():
        if isinstance(module, HDSLinear):
            # For layout-aware tuning, we train both the sparsity scores
            # and the underlying weights to adapt to the layout.
            module.scores.requires_grad = True
            module.linear.weight.requires_grad = True
            trainable_params.append(module.scores)
            trainable_params.append(module.linear.weight)
            if module.linear.bias is not None:
                module.linear.bias.requires_grad = True
                trainable_params.append(module.linear.bias)

    if not trainable_params:
        logger.warning("No trainable parameters for layout-aware HDS. Skipping.")
        return model

    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    num_epochs = hds_config.get("fine_tuning_epochs", DEFAULT_FINE_TUNING_EPOCHS)
    
    # Hyperparameters for the layout-aware regularization
    reg_lambda = hds_config.get("layout_aware_lambda", 0.01)
    penalty_fn = hds_config.get("penalty_fn", "inverse")

    device = next(model.parameters()).device
    model.train()

    perm_tensor = torch.tensor(permutation, device=device, dtype=torch.long)

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
                        module, perm_tensor, penalty_fn
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
