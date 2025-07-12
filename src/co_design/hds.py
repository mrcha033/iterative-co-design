"""
Hardware-Native Differentiable Sparsity (HDS) implementation.

This module implements structured N:M sparsity using Gumbel-Top-K reparameterization
for end-to-end differentiable training, as described in the paper.
"""
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from ..models.permutable_model import PermutableModel
from ..utils.exceptions import IterativeCoDesignError
from ..utils.config import BaseConfig


@dataclass
class HDSConfig(BaseConfig):
    """Configuration for Hardware-Native Differentiable Sparsity."""
    
    # Sparsity configuration
    sparsity_ratio: str = "2:4"  # N:M sparsity pattern
    target_sparsity: float = 0.5  # Overall target sparsity level
    
    # Training configuration
    learning_rate: float = 1e-4
    num_epochs: int = 10
    warmup_epochs: int = 2
    
    # Gumbel-Softmax configuration
    gumbel_temperature: float = 1.0
    temperature_decay: float = 0.9
    min_temperature: float = 0.1
    
    # Regularization
    l1_lambda: float = 1e-5
    l2_lambda: float = 1e-4
    
    # Hardware-specific settings
    block_size: int = 16  # Hardware block size for structured sparsity
    
    # Training settings
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Layer selection
    target_layers: List[str] = None  # If None, apply to all Linear layers
    exclude_layers: List[str] = None  # Layers to exclude from sparsity
    
    # Logging and checkpointing
    log_interval: int = 10
    save_checkpoints: bool = True
    checkpoint_dir: str = "checkpoints/hds"


class GumbelTopK(nn.Module):
    """
    Gumbel-Top-K module for differentiable sparsity mask generation.
    
    This implements the Gumbel-Top-K reparameterization trick that allows
    learning binary masks in a differentiable manner.
    """
    
    def __init__(self, shape: Tuple[int, ...], k: int, temperature: float = 1.0):
        """
        Initialize Gumbel-Top-K module.
        
        Args:
            shape: Shape of the weight tensor
            k: Number of elements to keep (top-k)
            temperature: Gumbel-Softmax temperature
        """
        super().__init__()
        
        self.shape = shape
        self.k = k
        self.temperature = temperature
        
        # Learnable logits for mask generation
        self.mask_logits = nn.Parameter(torch.zeros(shape))
        
        # Initialize with small random values
        nn.init.normal_(self.mask_logits, mean=0.0, std=0.1)
    
    def forward(self, training: bool = True) -> torch.Tensor:
        """
        Generate differentiable sparsity mask.
        
        Args:
            training: Whether in training mode
            
        Returns:
            Binary mask tensor
        """
        if training:
            # Generate Gumbel noise
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(self.mask_logits) + 1e-8) + 1e-8)
            
            # Add Gumbel noise to logits
            logits_with_noise = (self.mask_logits + gumbel_noise) / self.temperature
            
            # Apply top-k selection
            mask = self._differentiable_topk(logits_with_noise, self.k)
        else:
            # During inference, use hard top-k
            mask = self._hard_topk(self.mask_logits, self.k)
        
        return mask
    
    def _differentiable_topk(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        """
        Differentiable top-k selection using Gumbel-Softmax.
        
        Args:
            logits: Input logits
            k: Number of elements to select
            
        Returns:
            Differentiable binary mask
        """
        # Flatten for top-k operations
        flat_logits = logits.view(-1)
        
        # Get top-k values and indices
        top_k_values, top_k_indices = torch.topk(flat_logits, k, dim=0)
        
        # Create mask
        mask = torch.zeros_like(flat_logits)
        mask[top_k_indices] = 1.0
        
        # Apply Gumbel-Softmax for differentiability
        mask = F.gumbel_softmax(
            torch.stack([torch.zeros_like(mask), mask], dim=-1),
            tau=self.temperature,
            hard=True
        )[..., 1]
        
        return mask.view(self.shape)
    
    def _hard_topk(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        """
        Hard top-k selection for inference.
        
        Args:
            logits: Input logits
            k: Number of elements to select
            
        Returns:
            Hard binary mask
        """
        flat_logits = logits.view(-1)
        
        # Get top-k indices
        _, top_k_indices = torch.topk(flat_logits, k, dim=0)
        
        # Create hard mask
        mask = torch.zeros_like(flat_logits)
        mask[top_k_indices] = 1.0
        
        return mask.view(self.shape)
    
    def update_temperature(self, new_temperature: float):
        """Update Gumbel-Softmax temperature."""
        self.temperature = max(new_temperature, 1e-8)


class StructuredSparsityMask(nn.Module):
    """
    Structured sparsity mask for N:M sparsity patterns.
    
    This module generates hardware-friendly structured sparsity masks
    that follow N:M patterns (e.g., 2:4 means 2 non-zero elements per 4 elements).
    """
    
    def __init__(
        self,
        weight_shape: Tuple[int, ...],
        sparsity_ratio: str = "2:4",
        temperature: float = 1.0,
        block_size: int = 16
    ):
        """
        Initialize structured sparsity mask.
        
        Args:
            weight_shape: Shape of the weight tensor
            sparsity_ratio: N:M sparsity pattern (e.g., "2:4")
            temperature: Gumbel-Softmax temperature
            block_size: Block size for structured sparsity
        """
        super().__init__()
        
        self.weight_shape = weight_shape
        self.sparsity_ratio = sparsity_ratio
        self.temperature = temperature
        self.block_size = block_size
        
        # Parse sparsity ratio
        self.n, self.m = map(int, sparsity_ratio.split(":"))
        
        # Calculate number of blocks and mask shape
        self.num_blocks = self._calculate_num_blocks()
        
        # Create Gumbel-Top-K modules for each block
        self.mask_generators = nn.ModuleList([
            GumbelTopK(shape=(self.m,), k=self.n, temperature=temperature)
            for _ in range(self.num_blocks)
        ])
    
    def _calculate_num_blocks(self) -> int:
        """Calculate number of blocks for structured sparsity."""
        total_elements = np.prod(self.weight_shape)
        return total_elements // self.m
    
    def forward(self, training: bool = True) -> torch.Tensor:
        """
        Generate structured sparsity mask.
        
        Args:
            training: Whether in training mode
            
        Returns:
            Structured sparsity mask
        """
        # Generate masks for each block
        block_masks = []
        for generator in self.mask_generators:
            block_mask = generator(training=training)
            block_masks.append(block_mask)
        
        # Concatenate and reshape to match weight shape
        full_mask = torch.cat(block_masks, dim=0)
        
        # Handle size mismatch by padding or truncating
        total_elements = np.prod(self.weight_shape)
        if full_mask.numel() > total_elements:
            full_mask = full_mask[:total_elements]
        elif full_mask.numel() < total_elements:
            # Pad with zeros
            padding = torch.zeros(total_elements - full_mask.numel(), 
                                device=full_mask.device, dtype=full_mask.dtype)
            full_mask = torch.cat([full_mask, padding], dim=0)
        
        return full_mask.view(self.weight_shape)
    
    def update_temperature(self, new_temperature: float):
        """Update temperature for all mask generators."""
        self.temperature = new_temperature
        for generator in self.mask_generators:
            generator.update_temperature(new_temperature)


class HDSLayer(nn.Module):
    """
    Hardware-Native Differentiable Sparsity layer wrapper.
    
    This wraps a regular layer and adds learnable sparsity masks.
    """
    
    def __init__(
        self,
        layer: nn.Module,
        config: HDSConfig,
        layer_name: str = ""
    ):
        """
        Initialize HDS layer.
        
        Args:
            layer: The layer to wrap
            config: HDS configuration
            layer_name: Name of the layer
        """
        super().__init__()
        
        self.layer = layer
        self.config = config
        self.layer_name = layer_name
        
        # Create sparsity masks for weights
        self.weight_masks = nn.ModuleDict()
        
        for name, param in layer.named_parameters():
            if 'weight' in name and param.requires_grad:
                mask = StructuredSparsityMask(
                    weight_shape=param.shape,
                    sparsity_ratio=config.sparsity_ratio,
                    temperature=config.gumbel_temperature,
                    block_size=config.block_size
                )
                self.weight_masks[name] = mask
        
        # Store original weights
        self.original_weights = {}
        for name, param in layer.named_parameters():
            if 'weight' in name:
                self.original_weights[name] = param.clone().detach()
    
    def forward(self, *args, **kwargs):
        """Forward pass with sparsity masks applied."""
        # Apply sparsity masks to weights
        for name, param in self.layer.named_parameters():
            if 'weight' in name and name in self.weight_masks:
                mask = self.weight_masks[name](training=self.training)
                param.data = self.original_weights[name] * mask
        
        return self.layer(*args, **kwargs)
    
    def get_sparsity_ratio(self) -> Dict[str, float]:
        """Get actual sparsity ratio for each weight."""
        sparsity_ratios = {}
        
        for name, param in self.layer.named_parameters():
            if 'weight' in name and name in self.weight_masks:
                mask = self.weight_masks[name](training=False)
                sparsity = 1.0 - mask.sum().item() / mask.numel()
                sparsity_ratios[name] = sparsity
        
        return sparsity_ratios
    
    def update_temperature(self, new_temperature: float):
        """Update temperature for all masks."""
        for mask in self.weight_masks.values():
            mask.update_temperature(new_temperature)


class HDSOptimizer:
    """
    Hardware-Native Differentiable Sparsity optimizer.
    
    This class orchestrates the HDS training process, managing sparsity masks
    and fine-tuning the model to achieve hardware-friendly sparse patterns.
    """
    
    def __init__(self, config: HDSConfig):
        """
        Initialize HDS optimizer.
        
        Args:
            config: HDS configuration
        """
        self.config = config
        self.hds_layers = {}
        self.original_model = None
        self.optimizer = None
        self.scheduler = None
        
        # Training state
        self.current_epoch = 0
        self.training_history = []
    
    def prepare_model(self, model: PermutableModel) -> PermutableModel:
        """
        Prepare model for HDS training by wrapping layers with sparsity masks.
        
        Args:
            model: The model to prepare
            
        Returns:
            Model with HDS layers
        """
        self.original_model = model
        
        # Determine layers to apply sparsity to
        target_layers = self._get_target_layers(model)
        
        # Wrap target layers with HDS layers
        for layer_name in target_layers:
            try:
                layer = model.get_layer(layer_name)
                
                # Only wrap Linear layers for now
                if isinstance(layer, nn.Linear):
                    hds_layer = HDSLayer(layer, self.config, layer_name)
                    
                    # Replace the layer in the model
                    self._replace_layer(model, layer_name, hds_layer)
                    self.hds_layers[layer_name] = hds_layer
                    
            except Exception as e:
                warnings.warn(f"Failed to wrap layer {layer_name}: {e}")
        
        print(f"Wrapped {len(self.hds_layers)} layers with HDS masks")
        
        return model
    
    def _get_target_layers(self, model: PermutableModel) -> List[str]:
        """Get list of layers to apply sparsity to."""
        if self.config.target_layers:
            return self.config.target_layers
        
        # Default: all Linear layers
        target_layers = []
        for name, layer in model.model.named_modules():
            if isinstance(layer, nn.Linear) and name:
                # Check if layer should be excluded
                if self.config.exclude_layers and name in self.config.exclude_layers:
                    continue
                target_layers.append(name)
        
        return target_layers
    
    def _replace_layer(self, model: PermutableModel, layer_name: str, new_layer: nn.Module):
        """Replace a layer in the model."""
        # Navigate to the parent module
        parts = layer_name.split('.')
        current = model.model
        
        for part in parts[:-1]:
            current = getattr(current, part)
        
        # Replace the final layer
        setattr(current, parts[-1], new_layer)
    
    def train(
        self,
        model: PermutableModel,
        dataloader: torch.utils.data.DataLoader,
        validation_dataloader: Optional[torch.utils.data.DataLoader] = None,
        device: str = 'cuda'
    ) -> Dict[str, Any]:
        """
        Train the model with HDS.
        
        Args:
            model: The model to train
            dataloader: Training data loader
            validation_dataloader: Validation data loader
            device: Device to use for training
            
        Returns:
            Training results dictionary
        """
        print(f"Starting HDS training for {self.config.num_epochs} epochs...")
        
        # Set up training
        model.to(device)
        model.train()
        
        # Create optimizer for sparsity masks only
        mask_params = []
        for hds_layer in self.hds_layers.values():
            for mask in hds_layer.weight_masks.values():
                mask_params.extend(mask.parameters())
        
        self.optimizer = torch.optim.AdamW(
            mask_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.l2_lambda
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.num_epochs,
            eta_min=self.config.learning_rate * 0.01
        )
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Update temperature (annealing)
            if epoch >= self.config.warmup_epochs:
                temperature = max(
                    self.config.gumbel_temperature * (self.config.temperature_decay ** epoch),
                    self.config.min_temperature
                )
                self._update_temperature(temperature)
            
            # Training step
            train_loss = self._train_epoch(model, dataloader, device)
            
            # Validation step
            val_loss = None
            if validation_dataloader:
                val_loss = self._validate_epoch(model, validation_dataloader, device)
            
            # Update learning rate
            self.scheduler.step()
            
            # Log progress
            if epoch % self.config.log_interval == 0:
                sparsity_info = self._get_sparsity_info()
                print(f"Epoch {epoch}/{self.config.num_epochs}: "
                      f"Train Loss = {train_loss:.4f}, "
                      f"Val Loss = {val_loss:.4f if val_loss else 'N/A'}, "
                      f"Avg Sparsity = {sparsity_info['avg_sparsity']:.2%}")
            
            # Save training history
            self.training_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'sparsity_info': self._get_sparsity_info()
            })
        
        # Finalize sparsity masks
        self._finalize_masks()
        
        return {
            'training_history': self.training_history,
            'final_sparsity': self._get_sparsity_info(),
            'num_epochs': self.config.num_epochs
        }
    
    def _train_epoch(
        self,
        model: PermutableModel,
        dataloader: torch.utils.data.DataLoader,
        device: str
    ) -> float:
        """Train for one epoch."""
        total_loss = 0.0
        num_batches = 0
        
        model.train()
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
            # Move batch to device
            if isinstance(batch, (list, tuple)):
                batch = [item.to(device) if torch.is_tensor(item) else item for item in batch]
                inputs, targets = batch[0], batch[1] if len(batch) > 1 else batch[0]
            else:
                inputs = batch.to(device)
                targets = inputs  # For unsupervised tasks
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute loss
            loss = self._compute_loss(outputs, targets)
            
            # Add sparsity regularization
            sparsity_loss = self._compute_sparsity_loss()
            total_loss_batch = loss + sparsity_loss
            
            # Backward pass
            total_loss_batch.backward()
            
            # Gradient clipping
            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for hds_layer in self.hds_layers.values() 
                     for mask in hds_layer.weight_masks.values() 
                     for p in mask.parameters()],
                    self.config.max_grad_norm
                )
            
            # Optimizer step
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            total_loss += total_loss_batch.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def _validate_epoch(
        self,
        model: PermutableModel,
        dataloader: torch.utils.data.DataLoader,
        device: str
    ) -> float:
        """Validate for one epoch."""
        total_loss = 0.0
        num_batches = 0
        
        model.eval()
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                # Move batch to device
                if isinstance(batch, (list, tuple)):
                    batch = [item.to(device) if torch.is_tensor(item) else item for item in batch]
                    inputs, targets = batch[0], batch[1] if len(batch) > 1 else batch[0]
                else:
                    inputs = batch.to(device)
                    targets = inputs
                
                # Forward pass
                outputs = model(inputs)
                
                # Compute loss
                loss = self._compute_loss(outputs, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def _compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute training loss."""
        # Default to MSE loss for simplicity
        # In practice, this would be task-specific
        return F.mse_loss(outputs, targets)
    
    def _compute_sparsity_loss(self) -> torch.Tensor:
        """Compute sparsity regularization loss."""
        sparsity_loss = 0.0
        
        for hds_layer in self.hds_layers.values():
            for mask in hds_layer.weight_masks.values():
                # L1 regularization on mask logits
                sparsity_loss += self.config.l1_lambda * torch.sum(torch.abs(mask.mask_logits))
        
        return sparsity_loss
    
    def _update_temperature(self, new_temperature: float):
        """Update temperature for all HDS layers."""
        for hds_layer in self.hds_layers.values():
            hds_layer.update_temperature(new_temperature)
    
    def _get_sparsity_info(self) -> Dict[str, Any]:
        """Get sparsity information for all layers."""
        sparsity_info = {}
        total_sparsity = 0.0
        
        for layer_name, hds_layer in self.hds_layers.items():
            layer_sparsity = hds_layer.get_sparsity_ratio()
            sparsity_info[layer_name] = layer_sparsity
            
            # Calculate average sparsity for this layer
            if layer_sparsity:
                avg_layer_sparsity = np.mean(list(layer_sparsity.values()))
                total_sparsity += avg_layer_sparsity
        
        sparsity_info['avg_sparsity'] = total_sparsity / len(self.hds_layers)
        return sparsity_info
    
    def _finalize_masks(self):
        """Finalize sparsity masks by converting to hard masks."""
        print("Finalizing sparsity masks...")
        
        for hds_layer in self.hds_layers.values():
            for name, mask in hds_layer.weight_masks.items():
                # Generate final hard mask
                with torch.no_grad():
                    final_mask = mask(training=False)
                    
                    # Apply final mask to weights
                    if name in hds_layer.original_weights:
                        param = getattr(hds_layer.layer, name)
                        param.data = hds_layer.original_weights[name] * final_mask
    
    def get_sparse_model(self) -> PermutableModel:
        """Get the final sparse model."""
        return self.original_model
    
    def save_checkpoint(self, filepath: str):
        """Save training checkpoint."""
        checkpoint = {
            'config': self.config,
            'current_epoch': self.current_epoch,
            'training_history': self.training_history,
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
        }
        
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load training checkpoint."""
        checkpoint = torch.load(filepath)
        
        self.config = checkpoint['config']
        self.current_epoch = checkpoint['current_epoch']
        self.training_history = checkpoint['training_history']
        
        if self.optimizer and checkpoint['optimizer_state_dict']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Checkpoint loaded from {filepath}")


# Utility functions
def apply_hds_to_model(
    model: PermutableModel,
    dataloader: torch.utils.data.DataLoader,
    config: Optional[HDSConfig] = None,
    validation_dataloader: Optional[torch.utils.data.DataLoader] = None,
    device: str = 'cuda'
) -> Tuple[PermutableModel, Dict[str, Any]]:
    """
    Apply HDS to a model.
    
    Args:
        model: The model to apply HDS to
        dataloader: Training data loader
        config: HDS configuration
        validation_dataloader: Validation data loader
        device: Device to use
        
    Returns:
        Tuple of (sparse_model, training_results)
    """
    if config is None:
        config = HDSConfig()
    
    # Create HDS optimizer
    hds_optimizer = HDSOptimizer(config)
    
    # Prepare model
    model = hds_optimizer.prepare_model(model)
    
    # Train with HDS
    results = hds_optimizer.train(
        model=model,
        dataloader=dataloader,
        validation_dataloader=validation_dataloader,
        device=device
    )
    
    # Get final sparse model
    sparse_model = hds_optimizer.get_sparse_model()
    
    return sparse_model, results


def validate_sparsity_pattern(
    model: PermutableModel,
    sparsity_ratio: str = "2:4"
) -> Dict[str, Any]:
    """
    Validate that the model follows the specified sparsity pattern.
    
    Args:
        model: The model to validate
        sparsity_ratio: Expected sparsity pattern
        
    Returns:
        Validation results
    """
    n, m = map(int, sparsity_ratio.split(":"))
    
    results = {
        'valid_layers': [],
        'invalid_layers': [],
        'overall_sparsity': 0.0,
        'pattern_compliance': 0.0
    }
    
    total_elements = 0
    sparse_elements = 0
    compliant_blocks = 0
    total_blocks = 0
    
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            weight = param.data
            flat_weight = weight.view(-1)
            
            # Check overall sparsity
            zero_mask = (torch.abs(flat_weight) < 1e-8)
            layer_sparse = zero_mask.sum().item()
            layer_total = flat_weight.numel()
            
            total_elements += layer_total
            sparse_elements += layer_sparse
            
            # Check N:M pattern compliance
            num_blocks = layer_total // m
            layer_compliant = 0
            
            for i in range(num_blocks):
                block = flat_weight[i*m:(i+1)*m]
                block_nonzero = (torch.abs(block) >= 1e-8).sum().item()
                
                if block_nonzero == n:
                    layer_compliant += 1
                
                total_blocks += 1
            
            compliant_blocks += layer_compliant
            
            # Record layer results
            compliance_rate = layer_compliant / num_blocks if num_blocks > 0 else 0.0
            layer_sparsity = layer_sparse / layer_total
            
            layer_info = {
                'name': name,
                'sparsity': layer_sparsity,
                'compliance': compliance_rate,
                'shape': list(weight.shape)
            }
            
            if compliance_rate > 0.8:  # 80% compliance threshold
                results['valid_layers'].append(layer_info)
            else:
                results['invalid_layers'].append(layer_info)
    
    # Calculate overall metrics
    results['overall_sparsity'] = sparse_elements / total_elements if total_elements > 0 else 0.0
    results['pattern_compliance'] = compliant_blocks / total_blocks if total_blocks > 0 else 0.0
    
    return results