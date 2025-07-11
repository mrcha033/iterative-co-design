"""
Correlation matrix computation for IASP (IO-Aware Scan Permutation).
"""
import os
import hashlib
import warnings
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import psutil

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from ..models.permutable_model import PermutableModel
from ..utils.exceptions import IterativeCoDesignError


class CorrelationMatrixComputer:
    """
    Computes and caches correlation matrices for model activations.
    
    This class implements the batched correlation computation described in the paper,
    with memory-efficient accumulation and intelligent caching.
    """
    
    def __init__(self, cache_dir: Optional[str] = None, device: str = 'cuda'):
        """
        Initialize the correlation matrix computer.
        
        Args:
            cache_dir: Directory to store cached correlation matrices
            device: Device to use for computation
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path('./data/correlation_matrices')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        
    def compute_correlation_matrix(
        self,
        model: PermutableModel,
        dataloader: torch.utils.data.DataLoader,
        layer_name: str,
        num_samples: int = 1000,
        force_recompute: bool = False,
        check_memory: bool = True
    ) -> torch.Tensor:
        """
        Compute or load cached correlation matrix for a model layer.
        
        Args:
            model: The permutable model
            dataloader: DataLoader for activation collection
            layer_name: Name of the layer to analyze
            num_samples: Number of samples to use for correlation
            force_recompute: Force recomputation even if cached
            check_memory: Check memory usage and warn if excessive
            
        Returns:
            Correlation matrix as torch.Tensor [D, D]
        """
        # Generate cache key
        cache_key = self._generate_cache_key(
            model_name=model.model_name,
            layer_name=layer_name,
            num_samples=num_samples,
            dataset_info=self._get_dataset_info(dataloader)
        )
        
        cache_path = self.cache_dir / f"{cache_key}.pt"
        
        # Check if cached version exists
        if cache_path.exists() and not force_recompute:
            print(f"Loading cached correlation matrix from {cache_path}")
            return torch.load(cache_path, map_location='cpu')
        
        print(f"Computing correlation matrix for layer '{layer_name}'...")
        
        # Get layer info
        layer = model.get_layer(layer_name)
        layer_info = model.get_layer_info(layer_name)
        
        # Estimate memory usage
        if check_memory:
            self._check_memory_usage(layer_info, num_samples)
        
        # Collect activations
        activations = self._collect_activations(
            model, dataloader, layer_name, num_samples
        )
        
        # Compute correlation matrix
        correlation_matrix = self._compute_correlation(activations)
        
        # Cache the result
        torch.save(correlation_matrix, cache_path)
        print(f"Cached correlation matrix to {cache_path}")
        
        return correlation_matrix
    
    def _generate_cache_key(
        self,
        model_name: str,
        layer_name: str,
        num_samples: int,
        dataset_info: Dict[str, Any]
    ) -> str:
        """
        Generate SHA-256 cache key for correlation matrix.
        
        Args:
            model_name: Name of the model
            layer_name: Name of the layer
            num_samples: Number of samples used
            dataset_info: Information about the dataset
            
        Returns:
            SHA-256 hash as hex string
        """
        # Create a unique string representing the computation
        key_components = [
            f"model:{model_name}",
            f"layer:{layer_name}",
            f"samples:{num_samples}",
            f"dataset:{dataset_info.get('name', 'unknown')}",
            f"seq_len:{dataset_info.get('sequence_length', 'unknown')}",
            f"batch_size:{dataset_info.get('batch_size', 'unknown')}"
        ]
        
        key_string = "|".join(key_components)
        
        # Generate SHA-256 hash
        sha256_hash = hashlib.sha256(key_string.encode()).hexdigest()
        
        return sha256_hash[:16]  # Use first 16 characters for readability
    
    def _get_dataset_info(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, Any]:
        """Extract dataset information for cache key generation."""
        dataset_info = {
            'name': getattr(dataloader.dataset, 'name', 'unknown'),
            'batch_size': dataloader.batch_size,
            'sequence_length': getattr(dataloader.dataset, 'sequence_length', 'unknown')
        }
        return dataset_info
    
    def _check_memory_usage(self, layer_info: Dict[str, Any], num_samples: int):
        """
        Check estimated memory usage and warn if excessive.
        
        Args:
            layer_info: Information about the layer
            num_samples: Number of samples to collect
        """
        # Get available RAM
        available_ram_gb = psutil.virtual_memory().available / (1024**3)
        
        # Estimate activation size
        if 'weight_shape' in layer_info:
            # For linear layers, activation size is typically [batch_size, input_features]
            # We'll estimate based on the input dimension
            input_dim = layer_info['weight_shape'][1] if len(layer_info['weight_shape']) > 1 else layer_info['weight_shape'][0]
            
            # Estimate memory usage: num_samples * input_dim * 4 bytes (float32)
            estimated_memory_gb = (num_samples * input_dim * 4) / (1024**3)
            
            # Add correlation matrix memory: input_dim^2 * 4 bytes
            correlation_memory_gb = (input_dim * input_dim * 4) / (1024**3)
            
            total_memory_gb = estimated_memory_gb + correlation_memory_gb
            
            # Warn if memory usage is high
            if total_memory_gb > available_ram_gb * 0.8:
                warnings.warn(
                    f"High memory usage estimated: {total_memory_gb:.2f} GB "
                    f"(available: {available_ram_gb:.2f} GB). "
                    f"Consider reducing num_samples from {num_samples} to "
                    f"{int(num_samples * 0.5)} or less.",
                    UserWarning
                )
            
            print(f"Estimated memory usage: {total_memory_gb:.2f} GB "
                  f"(available: {available_ram_gb:.2f} GB)")
    
    def _collect_activations(
        self,
        model: PermutableModel,
        dataloader: torch.utils.data.DataLoader,
        layer_name: str,
        num_samples: int
    ) -> torch.Tensor:
        """
        Collect activations from the specified layer.
        
        Args:
            model: The permutable model
            dataloader: DataLoader for input data
            layer_name: Name of the layer to hook
            num_samples: Number of samples to collect
            
        Returns:
            Collected activations as tensor [num_samples, feature_dim]
        """
        model.eval()
        activations = []
        samples_collected = 0
        
        # Get the target layer
        target_layer = model.get_layer(layer_name)
        
        # Hook to collect activations
        def activation_hook(module, input, output):
            if samples_collected < num_samples:
                # Handle different output formats
                if isinstance(output, tuple):
                    # For some layers, output might be a tuple
                    activation = output[0]
                else:
                    activation = output
                
                # Flatten to 2D: [batch_size, feature_dim]
                if activation.dim() > 2:
                    activation = activation.view(activation.size(0), -1)
                
                activations.append(activation.detach().cpu())
        
        # Register the hook
        hook = target_layer.register_forward_hook(activation_hook)
        
        try:
            with torch.no_grad():
                pbar = tqdm(dataloader, desc=f"Collecting activations from {layer_name}")
                
                for batch_idx, batch in enumerate(pbar):
                    if samples_collected >= num_samples:
                        break
                    
                    # Handle different batch formats
                    if isinstance(batch, dict):
                        # For datasets that return dictionaries
                        if 'input_ids' in batch:
                            inputs = batch['input_ids'].to(self.device)
                        elif 'x' in batch:
                            inputs = batch['x'].to(self.device)
                        else:
                            inputs = batch[list(batch.keys())[0]].to(self.device)
                    elif isinstance(batch, (list, tuple)):
                        inputs = batch[0].to(self.device)
                    else:
                        inputs = batch.to(self.device)
                    
                    # Forward pass
                    try:
                        if hasattr(model, 'model_type') and model.model_type == 'gcn':
                            # Special handling for GCN models
                            if isinstance(batch, dict) and 'edge_index' in batch:
                                _ = model(inputs, batch['edge_index'].to(self.device))
                            else:
                                # Create dummy edge index for GCN
                                num_nodes = inputs.size(0)
                                edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2)).to(self.device)
                                _ = model(inputs, edge_index)
                        else:
                            _ = model(inputs)
                    except Exception as e:
                        print(f"Warning: Forward pass failed for batch {batch_idx}: {e}")
                        continue
                    
                    # Update samples collected
                    if activations:
                        samples_collected += activations[-1].size(0)
                        pbar.set_postfix({'samples': samples_collected})
                
        finally:
            # Remove the hook
            hook.remove()
        
        if not activations:
            raise IterativeCoDesignError(
                f"No activations collected for layer '{layer_name}'. "
                f"Check that the layer exists and the model forward pass works."
            )
        
        # Concatenate all activations
        all_activations = torch.cat(activations, dim=0)
        
        # Limit to requested number of samples
        if all_activations.size(0) > num_samples:
            all_activations = all_activations[:num_samples]
        
        print(f"Collected {all_activations.size(0)} activation samples with "
              f"{all_activations.size(1)} features")
        
        return all_activations
    
    def _compute_correlation(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Compute Pearson correlation matrix from activations.
        
        Args:
            activations: Activation tensor [num_samples, feature_dim]
            
        Returns:
            Correlation matrix [feature_dim, feature_dim]
        """
        print(f"Computing correlation matrix for {activations.shape[1]} features...")
        
        # Center the activations (subtract mean)
        activations_centered = activations - activations.mean(dim=0, keepdim=True)
        
        # Compute covariance matrix
        # Cov = (X^T @ X) / (n - 1)
        covariance = torch.mm(activations_centered.T, activations_centered) / (activations.size(0) - 1)
        
        # Compute standard deviations
        std_devs = torch.sqrt(torch.diag(covariance))
        
        # Compute correlation matrix
        # Corr = Cov / (std_i * std_j)
        correlation = covariance / (std_devs.unsqueeze(0) * std_devs.unsqueeze(1))
        
        # Handle numerical issues (NaN, Inf)
        correlation = torch.nan_to_num(correlation, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Ensure diagonal is 1.0
        correlation.fill_diagonal_(1.0)
        
        print(f"Correlation matrix computed: {correlation.shape}")
        print(f"Correlation range: [{correlation.min():.4f}, {correlation.max():.4f}]")
        
        return correlation
    
    def load_precomputed_matrix(self, matrix_path: str) -> torch.Tensor:
        """
        Load a precomputed correlation matrix from disk.
        
        Args:
            matrix_path: Path to the .pt file
            
        Returns:
            Loaded correlation matrix
        """
        if not Path(matrix_path).exists():
            raise FileNotFoundError(f"Precomputed matrix not found: {matrix_path}")
        
        print(f"Loading precomputed correlation matrix from {matrix_path}")
        correlation_matrix = torch.load(matrix_path, map_location='cpu')
        
        # Validate the matrix
        if correlation_matrix.dim() != 2 or correlation_matrix.size(0) != correlation_matrix.size(1):
            raise ValueError(f"Invalid correlation matrix shape: {correlation_matrix.shape}")
        
        return correlation_matrix
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the correlation matrix cache."""
        if not self.cache_dir.exists():
            return {'cache_dir': str(self.cache_dir), 'cached_matrices': 0, 'total_size_mb': 0}
        
        cached_files = list(self.cache_dir.glob('*.pt'))
        total_size = sum(f.stat().st_size for f in cached_files)
        
        return {
            'cache_dir': str(self.cache_dir),
            'cached_matrices': len(cached_files),
            'total_size_mb': total_size / (1024 * 1024),
            'files': [f.name for f in cached_files]
        }
    
    def clear_cache(self, pattern: Optional[str] = None):
        """
        Clear correlation matrix cache.
        
        Args:
            pattern: Optional pattern to match files (e.g., 'mamba*')
        """
        if not self.cache_dir.exists():
            return
        
        if pattern:
            files_to_remove = list(self.cache_dir.glob(f"{pattern}.pt"))
        else:
            files_to_remove = list(self.cache_dir.glob('*.pt'))
        
        for file_path in files_to_remove:
            file_path.unlink()
        
        print(f"Removed {len(files_to_remove)} cached correlation matrices")
    
    def validate_correlation_matrix(self, correlation_matrix: torch.Tensor) -> bool:
        """
        Validate a correlation matrix.
        
        Args:
            correlation_matrix: Matrix to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Check shape
        if correlation_matrix.dim() != 2:
            return False
        
        if correlation_matrix.size(0) != correlation_matrix.size(1):
            return False
        
        # Check diagonal
        diagonal = torch.diag(correlation_matrix)
        if not torch.allclose(diagonal, torch.ones_like(diagonal), atol=1e-3):
            return False
        
        # Check symmetry
        if not torch.allclose(correlation_matrix, correlation_matrix.T, atol=1e-3):
            return False
        
        # Check value range
        if correlation_matrix.min() < -1.1 or correlation_matrix.max() > 1.1:
            return False
        
        return True