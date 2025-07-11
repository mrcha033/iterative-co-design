"""
IASP (IO-Aware Scan Permutation) implementation for memory layout optimization.
"""
import warnings
from typing import Optional, Tuple, List, Dict, Any

import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.linalg import eigh
import networkx as nx

from .correlation import CorrelationMatrixComputer
from ..models.permutable_model import PermutableModel
from ..utils.exceptions import IterativeCoDesignError


class IASPPermutationOptimizer:
    """
    IASP (IO-Aware Scan Permutation) optimizer for finding optimal memory layouts.
    
    This class implements the spectral clustering-based permutation optimization
    described in the paper, using modularity maximization for cache efficiency.
    """
    
    def __init__(
        self,
        correlation_computer: Optional[CorrelationMatrixComputer] = None,
        device: str = 'cuda'
    ):
        """
        Initialize the IASP optimizer.
        
        Args:
            correlation_computer: Correlation matrix computer instance
            device: Device to use for computation
        """
        self.correlation_computer = correlation_computer or CorrelationMatrixComputer()
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
            permutation, info = self._spectral_permutation(
                correlation_matrix, num_clusters, correlation_threshold
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
    
    def _spectral_permutation(
        self,
        correlation_matrix: torch.Tensor,
        num_clusters: int,
        correlation_threshold: float
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generate permutation using spectral clustering on correlation matrix.
        
        This implements the core IASP algorithm from the paper:
        1. Construct affinity matrix from correlation
        2. Compute graph Laplacian
        3. Find smallest eigenvectors
        4. Cluster using k-means
        5. Concatenate cluster indices
        
        Args:
            correlation_matrix: Correlation matrix [D, D]
            num_clusters: Number of clusters
            correlation_threshold: Threshold for graph construction
            
        Returns:
            Tuple of (permutation, info_dict)
        """
        print(f"Computing spectral permutation with {num_clusters} clusters...")
        
        # Convert to numpy for scipy operations
        corr_np = correlation_matrix.numpy()
        D = corr_np.shape[0]
        
        # Step 1: Construct affinity matrix (W)
        # Use absolute correlation values above threshold
        W = np.abs(corr_np)
        W[W < correlation_threshold] = 0
        
        # Step 2: Compute degree matrix (D) and graph Laplacian (L)
        degree = np.sum(W, axis=1)
        D_matrix = np.diag(degree)
        L = D_matrix - W
        
        # Add small regularization to handle isolated nodes
        L += 1e-8 * np.eye(D)
        
        # Step 3: Compute smallest eigenvectors
        try:
            # We want the smallest eigenvalues (excluding the trivial 0 eigenvalue)
            eigenvalues, eigenvectors = eigh(L, eigvals=(1, num_clusters))
            
            # Use eigenvectors as features for clustering
            features = eigenvectors  # [D, num_clusters]
            
        except Exception as e:
            warnings.warn(f"Eigenvalue computation failed: {e}. Using random initialization.")
            features = np.random.randn(D, num_clusters)
        
        # Step 4: Perform k-means clustering
        try:
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features)
            
            # Compute silhouette score for clustering quality
            if num_clusters > 1 and len(np.unique(cluster_labels)) > 1:
                silhouette = silhouette_score(features, cluster_labels)
            else:
                silhouette = 0.0
            
        except Exception as e:
            warnings.warn(f"K-means clustering failed: {e}. Using random clustering.")
            cluster_labels = np.random.randint(0, num_clusters, size=D)
            silhouette = 0.0
        
        # Step 5: Construct permutation by concatenating cluster indices
        permutation = []
        for cluster_id in range(num_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            # Sort indices within cluster for consistency
            cluster_indices = np.sort(cluster_indices)
            permutation.extend(cluster_indices)
        
        # Handle any remaining indices (shouldn't happen with proper clustering)
        all_indices = set(range(D))
        used_indices = set(permutation)
        remaining_indices = sorted(all_indices - used_indices)
        permutation.extend(remaining_indices)
        
        permutation = np.array(permutation[:D])  # Ensure correct length
        
        # Compute clustering quality metrics
        info = {
            'num_clusters_used': len(np.unique(cluster_labels)),
            'silhouette_score': silhouette,
            'graph_edges': np.sum(W > 0),
            'graph_density': np.sum(W > 0) / (D * D),
            'eigenvalues': eigenvalues.tolist() if 'eigenvalues' in locals() else [],
        }
        
        return permutation, info
    
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
    
    def apply_permutation(
        self,
        model: PermutableModel,
        layer_name: str,
        permutation: np.ndarray,
        dimension: str = 'input'
    ) -> None:
        """
        Apply permutation to model layer.
        
        Args:
            model: The permutable model
            layer_name: Name of the layer to permute
            permutation: Permutation to apply
            dimension: Which dimension to permute ('input', 'output', 'both')
        """
        print(f"Applying permutation to layer '{layer_name}' ({dimension} dimension)...")
        
        # Validate permutation
        if not self._validate_permutation(permutation):
            raise ValueError("Invalid permutation: must be a valid rearrangement of indices")
        
        # Apply permutation using the model's method
        model.apply_permutation(layer_name, permutation, dimension)
        
        print(f"Permutation applied successfully to {layer_name}")
    
    def _validate_permutation(self, permutation: np.ndarray) -> bool:
        """
        Validate that a permutation is valid.
        
        Args:
            permutation: Permutation to validate
            
        Returns:
            True if valid, False otherwise
        """
        if len(permutation) == 0:
            return False
        
        # Check if it's a valid permutation (contains each index exactly once)
        expected_indices = set(range(len(permutation)))
        actual_indices = set(permutation)
        
        return expected_indices == actual_indices
    
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