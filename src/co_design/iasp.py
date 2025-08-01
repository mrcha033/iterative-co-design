"""
IASP (IO-Aware Scan Permutation) implementation for memory layout optimization.
"""
import warnings
from typing import Optional, Tuple, List, Dict, Any

import torch
import numpy as np

from .correlation import CorrelationMatrixComputer
from .spectral import SpectralClusteringOptimizer
from .apply import PermutationApplicator
from ..models.permutable_model import PermutableModel
from ..utils.exceptions import IterativeCoDesignError


class IASPPermutationOptimizer:
    """
    IASP (IO-Aware Scan Permutation) optimizer for finding optimal memory layouts.
    
    This class implements the spectral clustering-based permutation optimization
    described in the paper, using modularity maximization for cache efficiency.
    
    This class acts as a high-level coordinator that uses dedicated modules:
    - SpectralClusteringOptimizer for spectral clustering and permutation generation
    - PermutationApplicator for safe permutation application to model weights
    - CorrelationMatrixComputer for activation correlation computation
    """
    
    def __init__(
        self,
        correlation_computer: Optional[CorrelationMatrixComputer] = None,
        spectral_optimizer: Optional[SpectralClusteringOptimizer] = None,
        permutation_applicator: Optional[PermutationApplicator] = None,
        device: str = 'cuda'
    ):
        """
        Initialize the IASP optimizer.
        
        Args:
            correlation_computer: Correlation matrix computer instance
            spectral_optimizer: Spectral clustering optimizer instance  
            permutation_applicator: Permutation applicator instance
            device: Device to use for computation
        """
        self.correlation_computer = correlation_computer or CorrelationMatrixComputer()
        self.spectral_optimizer = spectral_optimizer or SpectralClusteringOptimizer()
        self.permutation_applicator = permutation_applicator
        self.device = device
    
    def optimize_permutation(
        self,
        model: PermutableModel,
        dataloader: torch.utils.data.DataLoader,
        layer_name: str,
        num_clusters: int = 64,
        num_samples: int = 1000,
        correlation_threshold: float = 0.1,
        method: str = 'spectral',
        force_recompute: bool = False
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Optimize permutation for a model layer using IASP.
        
        Args:
            model: The permutable model
            dataloader: DataLoader for activation collection
            layer_name: Name of the layer to optimize
            num_clusters: Number of clusters for spectral clustering
            num_samples: Number of samples for correlation computation
            correlation_threshold: Threshold for correlation graph construction
            method: Optimization method ('spectral', 'tsp', 'random')
            force_recompute: Force recomputation of correlation matrix
            
        Returns:
            Tuple of (permutation, optimization_info)
        """
        print(f"Optimizing permutation for layer '{layer_name}' using {method} method...")
        
        # Step 1: Compute or load correlation matrix
        correlation_matrix = self.correlation_computer.compute_correlation_matrix(
            model=model,
            dataloader=dataloader,
            layer_name=layer_name,
            num_samples=num_samples,
            force_recompute=force_recompute
        )
        
        # Step 2: Generate permutation based on method
        if method == 'spectral':
            permutation, info = self.spectral_optimizer.compute_permutation(
                correlation_matrix, 
                num_clusters=num_clusters, 
                correlation_threshold=correlation_threshold
            )
        elif method == 'tsp':
            permutation, info = self._tsp_permutation(
                correlation_matrix, correlation_threshold
            )
        elif method == 'random':
            permutation, info = self._random_permutation(correlation_matrix.size(0))
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Step 3: Compute modularity for evaluation
        modularity = self._compute_modularity(correlation_matrix, permutation)
        
        # Combine optimization info
        optimization_info = {
            'method': method,
            'num_clusters': num_clusters,
            'correlation_threshold': correlation_threshold,
            'modularity': modularity,
            'layer_name': layer_name,
            'correlation_matrix_shape': correlation_matrix.shape,
            **info
        }
        
        print(f"Permutation optimization completed. Modularity: {modularity:.4f}")
        
        return permutation, optimization_info
    
    def apply_optimized_permutation(
        self,
        model: PermutableModel,
        layer_name: str,
        permutation: np.ndarray,
        dimension: str = 'input',
        validate: bool = True
    ) -> Dict[str, Any]:
        """
        Apply an optimized permutation to a model layer.
        
        Args:
            model: The permutable model
            layer_name: Name of the layer to permute
            permutation: Permutation to apply
            dimension: Which dimension to permute ('input', 'output', 'both')
            validate: Whether to validate permutation before applying
            
        Returns:
            Application result dictionary
        """
        print(f"Applying optimized permutation to layer '{layer_name}' ({dimension} dimension)...")
        
        # Create permutation applicator if not provided
        if self.permutation_applicator is None:
            self.permutation_applicator = PermutationApplicator(model)
        
        # Apply permutation using the dedicated module
        result = self.permutation_applicator.apply_permutation(
            layer_name=layer_name,
            permutation=permutation,
            dimension=dimension,
            validate=validate
        )
        
        print(f"Permutation applied successfully to {layer_name}")
        
        return result
    
    def _tsp_permutation(
        self,
        correlation_matrix: torch.Tensor,
        correlation_threshold: float
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generate permutation using TSP-based approach.
        
        This treats the correlation matrix as a distance matrix and finds
        a path that minimizes the total distance (maximizes correlation).
        
        Args:
            correlation_matrix: Correlation matrix [D, D]
            correlation_threshold: Threshold for graph construction
            
        Returns:
            Tuple of (permutation, info_dict)
        """
        print("Computing TSP-based permutation...")
        
        corr_np = correlation_matrix.numpy()
        D = corr_np.shape[0]
        
        # Convert correlation to distance (higher correlation = lower distance)
        # Use 1 - |correlation| as distance
        distance_matrix = 1.0 - np.abs(corr_np)
        
        # Simple greedy TSP approximation
        # Start from node 0 and always go to nearest unvisited node
        visited = set()
        current = 0
        path = [current]
        visited.add(current)
        total_distance = 0
        
        while len(visited) < D:
            # Find nearest unvisited node
            min_distance = float('inf')
            next_node = -1
            
            for i in range(D):
                if i not in visited:
                    dist = distance_matrix[current, i]
                    if dist < min_distance:
                        min_distance = dist
                        next_node = i
            
            if next_node != -1:
                path.append(next_node)
                visited.add(next_node)
                total_distance += min_distance
                current = next_node
            else:
                # Fallback: add remaining nodes in order
                remaining = [i for i in range(D) if i not in visited]
                path.extend(remaining)
                break
        
        permutation = np.array(path)
        
        info = {
            'total_distance': total_distance,
            'average_distance': total_distance / (D - 1) if D > 1 else 0,
            'path_length': len(path)
        }
        
        return permutation, info
    
    def _random_permutation(self, dimension: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generate random permutation (baseline).
        
        Args:
            dimension: Size of the permutation
            
        Returns:
            Tuple of (permutation, info_dict)
        """
        print("Generating random permutation...")
        
        permutation = np.random.permutation(dimension)
        
        info = {
            'method_note': 'Random baseline permutation'
        }
        
        return permutation, info
    
    def _compute_modularity(
        self,
        correlation_matrix: torch.Tensor,
        permutation: np.ndarray
    ) -> float:
        """
        Compute modularity of a permutation.
        
        Modularity measures how well the permutation groups highly correlated
        dimensions into contiguous blocks, which improves cache locality.
        
        Args:
            correlation_matrix: Original correlation matrix
            permutation: Permutation to evaluate
            
        Returns:
            Modularity score (higher is better)
        """
        corr_np = correlation_matrix.numpy()
        
        # Apply permutation to correlation matrix
        permuted_corr = corr_np[permutation, :][:, permutation]
        
        # Compute modularity based on block structure
        # We'll use a simple block-based modularity metric
        D = len(permutation)
        block_size = 64  # Typical cache line size consideration
        
        total_modularity = 0
        num_blocks = (D + block_size - 1) // block_size
        
        for i in range(num_blocks):
            start_i = i * block_size
            end_i = min((i + 1) * block_size, D)
            
            # Intra-block correlation (should be high)
            block_corr = permuted_corr[start_i:end_i, start_i:end_i]
            intra_block_corr = np.mean(np.abs(block_corr))
            
            # Inter-block correlation (should be low)
            inter_block_corr = 0
            count = 0
            
            for j in range(num_blocks):
                if i != j:
                    start_j = j * block_size
                    end_j = min((j + 1) * block_size, D)
                    
                    inter_corr = permuted_corr[start_i:end_i, start_j:end_j]
                    inter_block_corr += np.mean(np.abs(inter_corr))
                    count += 1
            
            if count > 0:
                inter_block_corr /= count
            
            # Modularity contribution: high intra-block, low inter-block
            block_modularity = intra_block_corr - inter_block_corr
            total_modularity += block_modularity
        
        return total_modularity / num_blocks if num_blocks > 0 else 0.0
    
    def evaluate_permutation(
        self,
        correlation_matrix: torch.Tensor,
        permutation: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate a permutation's quality.
        
        Args:
            correlation_matrix: Original correlation matrix
            permutation: Permutation to evaluate
            
        Returns:
            Dictionary with quality metrics
        """
        modularity = self._compute_modularity(correlation_matrix, permutation)
        
        # Compute additional metrics
        corr_np = correlation_matrix.numpy()
        permuted_corr = corr_np[permutation, :][:, permutation]
        
        # Locality metric: correlation between adjacent elements
        locality = 0
        for i in range(len(permutation) - 1):
            locality += abs(permuted_corr[i, i + 1])
        locality /= (len(permutation) - 1)
        
        # Block coherence: variance within blocks
        block_size = 64
        block_coherence = 0
        num_blocks = (len(permutation) + block_size - 1) // block_size
        
        for i in range(num_blocks):
            start = i * block_size
            end = min((i + 1) * block_size, len(permutation))
            
            if end > start:
                block_corr = permuted_corr[start:end, start:end]
                block_coherence += np.var(block_corr)
        
        block_coherence /= num_blocks
        
        return {
            'modularity': modularity,
            'locality': locality,
            'block_coherence': block_coherence,
            'permutation_length': len(permutation)
        }
    
    # Backward compatibility methods
    def apply_permutation(
        self,
        model: PermutableModel,
        layer_name: str,
        permutation: np.ndarray,
        dimension: str = 'input'
    ) -> None:
        """
        Apply permutation to model layer (backward compatibility).
        
        Args:
            model: The permutable model
            layer_name: Name of the layer to permute
            permutation: Permutation to apply
            dimension: Which dimension to permute ('input', 'output', 'both')
        """
        warnings.warn(
            "apply_permutation is deprecated. Use apply_optimized_permutation instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        result = self.apply_optimized_permutation(
            model=model,
            layer_name=layer_name,
            permutation=permutation,
            dimension=dimension
        )
        
        # Check for errors in the result
        if not result.get('success', False):
            raise IterativeCoDesignError(f"Failed to apply permutation to {layer_name}")
    
    def validate_permutation(self, permutation: np.ndarray) -> bool:
        """
        Validate that a permutation is valid (backward compatibility).
        
        Args:
            permutation: Permutation to validate
            
        Returns:
            True if valid, False otherwise
        """
        warnings.warn(
            "validate_permutation is deprecated. Use spectral_optimizer.validate_permutation instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        return self.spectral_optimizer.validate_permutation(permutation)