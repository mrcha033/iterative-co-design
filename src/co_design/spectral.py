"""
Spectral clustering implementation for IASP permutation optimization.
"""
import warnings
from typing import Tuple, Dict, Any, Optional

import numpy as np
import torch
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from ..utils.exceptions import IterativeCoDesignError


class SpectralClusteringOptimizer:
    """
    Spectral clustering optimizer for finding optimal permutations.
    
    This class implements the spectral clustering algorithm described in the paper:
    1. Construct affinity matrix from correlation
    2. Compute graph Laplacian
    3. Find smallest eigenvectors
    4. Cluster using k-means
    5. Concatenate cluster indices
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the spectral clustering optimizer.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
    
    def compute_permutation(
        self,
        correlation_matrix: torch.Tensor,
        num_clusters: int,
        correlation_threshold: float = 0.1,
        use_sparse: bool = False,
        block_size: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute permutation using spectral clustering.
        
        Args:
            correlation_matrix: Correlation matrix [D, D]
            num_clusters: Number of clusters
            correlation_threshold: Threshold for graph construction
            use_sparse: Whether to use sparse matrix operations
            block_size: Block size for block-wise approximation (if D > 4096)
            
        Returns:
            Tuple of (permutation, optimization_info)
        """
        corr_np = correlation_matrix.numpy()
        D = corr_np.shape[0]
        
        print(f"Computing spectral permutation for {D} dimensions with {num_clusters} clusters...")
        
        # Use block-wise approximation for large dimensions
        if D > 4096 and block_size is None:
            block_size = min(2048, D // 2)
            print(f"Using block-wise approximation with block size {block_size}")
            return self._compute_blockwise_permutation(
                corr_np, num_clusters, correlation_threshold, block_size
            )
        
        # Step 1: Construct affinity matrix (W)
        W = self._construct_affinity_matrix(corr_np, correlation_threshold)
        
        # Step 2: Compute graph Laplacian (L)
        L = self._compute_graph_laplacian(W, use_sparse)
        
        # Step 3: Compute smallest eigenvectors
        features = self._compute_eigenvectors(L, num_clusters, use_sparse)
        
        # Step 4: Perform k-means clustering
        cluster_labels, silhouette = self._perform_clustering(features, num_clusters)
        
        # Step 5: Construct permutation from clusters
        permutation = self._construct_permutation(cluster_labels, num_clusters, D)
        
        # Compute optimization info
        info = {
            'num_clusters_used': len(np.unique(cluster_labels)),
            'silhouette_score': silhouette,
            'graph_edges': np.sum(W > 0),
            'graph_density': np.sum(W > 0) / (D * D),
            'dimension': D,
            'use_sparse': use_sparse,
            'block_size': block_size
        }
        
        return permutation, info
    
    def _construct_affinity_matrix(
        self,
        correlation_matrix: np.ndarray,
        correlation_threshold: float
    ) -> np.ndarray:
        """
        Construct affinity matrix from correlation matrix.
        
        Args:
            correlation_matrix: Correlation matrix
            correlation_threshold: Threshold for graph construction
            
        Returns:
            Affinity matrix
        """
        # Use absolute correlation values above threshold
        W = np.abs(correlation_matrix)
        W[W < correlation_threshold] = 0
        
        # Ensure diagonal is zero (no self-loops)
        np.fill_diagonal(W, 0)
        
        return W
    
    def _compute_graph_laplacian(
        self,
        affinity_matrix: np.ndarray,
        use_sparse: bool = False
    ) -> np.ndarray:
        """
        Compute graph Laplacian from affinity matrix.
        
        Args:
            affinity_matrix: Affinity matrix
            use_sparse: Whether to use sparse operations
            
        Returns:
            Graph Laplacian matrix
        """
        if use_sparse:
            W_sparse = csr_matrix(affinity_matrix)
            degree = np.array(W_sparse.sum(axis=1)).flatten()
            D_sparse = csr_matrix((degree, (range(len(degree)), range(len(degree)))), 
                                 shape=affinity_matrix.shape)
            L = D_sparse - W_sparse
            return L
        else:
            # Compute degree matrix
            degree = np.sum(affinity_matrix, axis=1)
            D_matrix = np.diag(degree)
            
            # Compute Laplacian: L = D - W
            L = D_matrix - affinity_matrix
            
            # Add small regularization to handle isolated nodes
            L += 1e-8 * np.eye(L.shape[0])
            
            return L
    
    def _compute_eigenvectors(
        self,
        laplacian: np.ndarray,
        num_clusters: int,
        use_sparse: bool = False
    ) -> np.ndarray:
        """
        Compute smallest eigenvectors of the Laplacian.
        
        Args:
            laplacian: Graph Laplacian matrix
            num_clusters: Number of clusters (eigenvectors to compute)
            use_sparse: Whether to use sparse operations
            
        Returns:
            Eigenvector matrix [D, num_clusters]
        """
        try:
            if use_sparse:
                # Use sparse eigenvalue solver
                eigenvalues, eigenvectors = eigsh(
                    laplacian, k=num_clusters, which='SM', sigma=1e-6
                )
            else:
                # Use dense eigenvalue solver
                # We want the smallest eigenvalues (excluding the trivial 0 eigenvalue)
                eigenvalues, eigenvectors = eigh(
                    laplacian, eigvals=(1, num_clusters)
                )
            
            # eigenvectors shape: [D, num_clusters]
            features = eigenvectors
            
            # Normalize features (optional, but often helps with clustering)
            features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
            
            return features
            
        except Exception as e:
            warnings.warn(f"Eigenvalue computation failed: {e}. Using random initialization.")
            D = laplacian.shape[0]
            return np.random.RandomState(self.random_state).randn(D, num_clusters)
    
    def _perform_clustering(
        self,
        features: np.ndarray,
        num_clusters: int
    ) -> Tuple[np.ndarray, float]:
        """
        Perform k-means clustering on eigenvector features.
        
        Args:
            features: Eigenvector features [D, num_clusters]
            num_clusters: Number of clusters
            
        Returns:
            Tuple of (cluster_labels, silhouette_score)
        """
        try:
            # Perform k-means clustering with fixed random state
            kmeans = KMeans(
                n_clusters=num_clusters,
                random_state=self.random_state,
                n_init=10,
                max_iter=300
            )
            cluster_labels = kmeans.fit_predict(features)
            
            # Compute silhouette score for clustering quality
            if num_clusters > 1 and len(np.unique(cluster_labels)) > 1:
                silhouette = silhouette_score(features, cluster_labels)
            else:
                silhouette = 0.0
            
            return cluster_labels, silhouette
            
        except Exception as e:
            warnings.warn(f"K-means clustering failed: {e}. Using random clustering.")
            D = features.shape[0]
            cluster_labels = np.random.RandomState(self.random_state).randint(
                0, num_clusters, size=D
            )
            return cluster_labels, 0.0
    
    def _construct_permutation(
        self,
        cluster_labels: np.ndarray,
        num_clusters: int,
        dimension: int
    ) -> np.ndarray:
        """
        Construct permutation by concatenating cluster indices.
        
        Args:
            cluster_labels: Cluster assignments for each dimension
            num_clusters: Number of clusters
            dimension: Total dimension size
            
        Returns:
            Permutation array
        """
        permutation = []
        
        # Concatenate indices from each cluster
        for cluster_id in range(num_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            
            # Sort indices within cluster for consistency
            cluster_indices = np.sort(cluster_indices)
            permutation.extend(cluster_indices)
        
        # Handle any remaining indices (shouldn't happen with proper clustering)
        all_indices = set(range(dimension))
        used_indices = set(permutation)
        remaining_indices = sorted(all_indices - used_indices)
        permutation.extend(remaining_indices)
        
        # Ensure correct length
        permutation = np.array(permutation[:dimension])
        
        # Validate permutation
        if len(permutation) != dimension or set(permutation) != set(range(dimension)):
            raise IterativeCoDesignError(
                f"Invalid permutation generated: length {len(permutation)}, "
                f"expected {dimension}"
            )
        
        return permutation
    
    def _compute_blockwise_permutation(
        self,
        correlation_matrix: np.ndarray,
        num_clusters: int,
        correlation_threshold: float,
        block_size: int
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute permutation using block-wise approximation for large dimensions.
        
        This method divides the correlation matrix into blocks and optimizes
        permutations within each block, then concatenates the results.
        
        Args:
            correlation_matrix: Correlation matrix [D, D]
            num_clusters: Number of clusters
            correlation_threshold: Threshold for graph construction
            block_size: Size of each block
            
        Returns:
            Tuple of (permutation, optimization_info)
        """
        D = correlation_matrix.shape[0]
        num_blocks = (D + block_size - 1) // block_size
        
        print(f"Using block-wise approximation: {num_blocks} blocks of size {block_size}")
        
        permutation = []
        block_info = []
        
        for block_idx in range(num_blocks):
            start_idx = block_idx * block_size
            end_idx = min((block_idx + 1) * block_size, D)
            actual_block_size = end_idx - start_idx
            
            # Extract block correlation matrix
            block_corr = correlation_matrix[start_idx:end_idx, start_idx:end_idx]
            
            # Compute clusters for this block
            block_clusters = min(num_clusters, actual_block_size // 4)  # Reasonable ratio
            if block_clusters < 2:
                block_clusters = 2
            
            # Compute permutation for this block
            try:
                block_perm, block_info_dict = self.compute_permutation(
                    torch.from_numpy(block_corr),
                    block_clusters,
                    correlation_threshold,
                    use_sparse=False,
                    block_size=None  # Prevent recursive block-wise
                )
                
                # Adjust permutation indices to global indices
                global_block_perm = block_perm + start_idx
                permutation.extend(global_block_perm)
                
                block_info.append({
                    'block_idx': block_idx,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'block_size': actual_block_size,
                    'clusters': block_clusters,
                    **block_info_dict
                })
                
            except Exception as e:
                warnings.warn(f"Block {block_idx} optimization failed: {e}. Using identity.")
                identity_perm = np.arange(start_idx, end_idx)
                permutation.extend(identity_perm)
                
                block_info.append({
                    'block_idx': block_idx,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'block_size': actual_block_size,
                    'error': str(e)
                })
        
        permutation = np.array(permutation)
        
        # Compute overall info
        info = {
            'method': 'blockwise_spectral',
            'num_blocks': num_blocks,
            'block_size': block_size,
            'dimension': D,
            'block_info': block_info,
            'total_clusters': sum(info.get('clusters', 0) for info in block_info)
        }
        
        return permutation, info
    
    def validate_permutation(self, permutation: np.ndarray) -> bool:
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