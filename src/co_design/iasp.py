"""
IO-Aware Scan Permutation (IASP) module.

This module implements the IO-Aware Scan Permutation algorithm, which optimizes
memory layout by finding permutations that maximize modularity. IASP uses spectral
clustering to group correlated model dimensions into contiguous memory regions,
improving cache locality and reducing memory access latency.

Key functions:
- get_activation_correlation: Collect model activations and compute correlation matrix
- find_optimal_permutation_from_matrix: Find optimal permutation using modularity
- find_optimal_permutation: End-to-end permutation optimization for models
"""

import torch
import numpy as np
from tqdm import tqdm
from sklearn.cluster import SpectralClustering
from .modularity import calculate_modularity
from typing import List, Optional, Tuple
import torch.nn as nn
from torch.utils.data import DataLoader
import logging

logger = logging.getLogger(__name__)

# Constants for IASP algorithm
DEFAULT_MAX_ACTIVATION_SAMPLES = 512
MIN_CLUSTER_SIZE = 2
DEFAULT_MIN_CLUSTERS_DIVISOR = 128  # d_model // 128
DEFAULT_MAX_CLUSTERS_DIVISOR = 16   # d_model // 16
SPECTRAL_CLUSTERING_N_INIT = 10
SPECTRAL_CLUSTERING_RANDOM_STATE = 0  # For reproducibility
NAN_REPLACEMENT_VALUE = 0.0
DIAGONAL_CORRELATION_VALUE = 1.0
DEBUG_LAYER_DISPLAY_LIMIT = 20


def get_activation_correlation(
    model: nn.Module,
    dataloader: DataLoader,
    target_layer_names: List[str] | str | None = None,
    max_samples: int = DEFAULT_MAX_ACTIVATION_SAMPLES,
    device: Optional[str] = None,
    *,
    # Backwards-compatibility alias. If provided, `target_layer_names` is ignored
    target_layer_name: Optional[str] = None,
) -> np.ndarray:
    """
    Computes the average activation correlation matrix across multiple target layers.

    Args:
        model: The PyTorch model to analyze.
        dataloader: The dataloader providing data samples.
        target_layer_names: A list of layer names to hook for activations.
        max_samples: The maximum number of samples to use for calculation.
        device: The device to run the model on.

    Returns:
        A 2D numpy array representing the average Pearson correlation matrix.
    """
    # ------------------------------------------------------------------
    # Parameter handling
    # ------------------------------------------------------------------
    if target_layer_names is None and target_layer_name is not None:
        target_layer_names = [target_layer_name]

    if target_layer_names is None:
        raise ValueError("Either 'target_layer_names' or 'target_layer_name' must be provided.")

    if isinstance(target_layer_names, str):
        target_layer_names = [target_layer_names]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"No device specified, auto-detected: {device}")

    model.to(device)
    model.eval()

    correlation_matrices = []

    for target_layer_name in target_layer_names:
        activations = []

        def hook_fn(module, input, output):
            # Extract activation tensor from tuple outputs if necessary
            act = output[0] if isinstance(output, tuple) else output

            # --------------------------------------------------------------
            # Memory‐friendly downsampling
            # --------------------------------------------------------------
            # For sequence models (3-D activations: B × T × H), we only keep the
            # first token (e.g., CLS) to avoid storing the full sequence. This
            # reduces memory by a factor of ~T (often 512) while still capturing
            # representative hidden-state statistics for correlation analysis.
            #
            # If activations are already 2-D (B × H), we leave them unchanged.
            # --------------------------------------------------------------
            if act.ndim == 3:
                act = act[:, 0, :]  # (batch, hidden_size)

            activations.append(act.detach().cpu().numpy())

        target_layer = None
        for name, module in model.named_modules():
            if name == target_layer_name:
                target_layer = module
                break

        if target_layer is None:
            logger.error(f"Layer '{target_layer_name}' not found in the model.")
            available_layers = [name for name, module in model.named_modules() if hasattr(module, 'weight') or hasattr(module, 'bias')]
            logger.debug(f"Available layers: {available_layers[:DEBUG_LAYER_DISPLAY_LIMIT]}")
            raise ValueError(f"Layer '{target_layer_name}' not found in the model.")

        handle = target_layer.register_forward_hook(hook_fn)

        samples_collected = 0
        for batch in tqdm(dataloader, desc=f"Collecting activations for {target_layer_name}"):
            if samples_collected >= max_samples:
                break

            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            with torch.no_grad():
                _ = model(**inputs)

            first_tensor = next((v for v in inputs.values() if torch.is_tensor(v)), None)
            if first_tensor is not None:
                samples_collected += first_tensor.size(0)
            else:
                samples_collected += activations[-1].shape[0]

            if samples_collected >= max_samples:
                break

        handle.remove()

        if not activations:
            raise ValueError(f"No activations were collected for layer {target_layer_name}.")

        all_activations = np.concatenate(activations, axis=0)[:max_samples]

        if all_activations.ndim == 3:
            num_samples, seq_len, hidden_dim = all_activations.shape
            all_activations_reshaped = all_activations.reshape(num_samples * seq_len, hidden_dim)
        elif all_activations.ndim == 2:
            all_activations_reshaped = all_activations
        else:
            raise ValueError(f"Expected 2D or 3D activations, but got {all_activations.ndim}D.")

        correlation_matrix = np.corrcoef(all_activations_reshaped, rowvar=False)

        if np.any(np.isnan(correlation_matrix)):
            logger.warning(f"Found NaN values in correlation matrix for layer {target_layer_name}, replacing with identity.")
            correlation_matrix = np.nan_to_num(correlation_matrix, nan=NAN_REPLACEMENT_VALUE)
            np.fill_diagonal(correlation_matrix, DIAGONAL_CORRELATION_VALUE)
        
        correlation_matrices.append(correlation_matrix)

    if not correlation_matrices:
        raise ValueError("No correlation matrices were computed.")

    # Return the average correlation matrix
    avg_correlation_matrix = np.mean(correlation_matrices, axis=0)
    logger.info(f"Computed average correlation matrix from {len(correlation_matrices)} layers.")
    return avg_correlation_matrix


def find_permutation_from_matrix(
    correlation_matrix: np.ndarray, n_clusters: int
) -> List[int]:
    """
    Finds an optimal permutation from a correlation matrix using spectral clustering.

    This function takes a correlation matrix, computes clusters using spectral
    clustering on the absolute values of the correlations (to treat strong
    positive and negative correlations as strong connections), and then generates
    a permutation that groups nodes from the same cluster together.

    Args:
        correlation_matrix: The N x N correlation matrix.
        n_clusters: The number of clusters to form.

    Returns:
        A list of integers representing the permutation.
    """
    logger.info("Step 2a: Performing spectral clustering...")
    # We use the absolute value of correlation as affinity for clustering,
    # as strong negative correlations are also meaningful for grouping.
    affinity_matrix = np.abs(correlation_matrix)

    clustering = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        random_state=SPECTRAL_CLUSTERING_RANDOM_STATE,  # for reproducibility
    ).fit(affinity_matrix)

    labels = clustering.labels_

    logger.info("Step 2b: Constructing permutation from clusters...")
    # argsort provides a permutation that groups indices by their corresponding labels.
    # Using a stable sort for robustness.
    permutation = np.argsort(labels, kind="mergesort")

    return permutation.tolist()


def find_optimal_permutation_from_matrix(
    correlation_matrix: np.ndarray,
    clusters_range: Optional[Tuple[int, int]] = None,
    num_clusters: Optional[int] = None,
    model_type: Optional[str] = None,
) -> List[int]:
    """
    Finds the optimal permutation of neuron indices to maximize modularity.

    This function takes a correlation matrix and performs spectral clustering to
    group correlated neurons. It then orders the neurons based on these
    clusters to find a permutation that improves data locality. If num_clusters
    is not provided, it searches for the optimal number of clusters that
    maximizes the modularity score.
    
    For Mamba models, applies dimension-aware permutation that respects 
    functional block boundaries to preserve architectural integrity.

    Args:
        correlation_matrix: A 2D numpy array of activation correlations.
        clusters_range: Range of cluster numbers to search over.
        num_clusters: The number of clusters to form. If None, the optimal
            number of clusters will be determined automatically.
        model_type: Type of model (e.g., "mamba") for specialized handling.

    Returns:
        A list of integers representing the optimal permutation of indices.
    """
    d_model = correlation_matrix.shape[0]
    
    # Check if this is a Mamba model - apply dimension-aware permutation
    if model_type and "mamba" in model_type.lower():
        logger.info("🔧 Applying dimension-aware permutation for Mamba model")
        return _mamba_aware_permutation(correlation_matrix, clusters_range)
    
    # Standard IASP for other models
    if num_clusters is None:
        # Determine search bounds
        if clusters_range is None:
            min_clusters = max(MIN_CLUSTER_SIZE, d_model // DEFAULT_MIN_CLUSTERS_DIVISOR)
            max_clusters = max(MIN_CLUSTER_SIZE, d_model // DEFAULT_MAX_CLUSTERS_DIVISOR)
        else:
            min_clusters, max_clusters = clusters_range

        # Search for the best number of clusters to maximize modularity
        best_modularity = -1
        best_permutation = list(range(d_model))

        for k in range(min_clusters, max_clusters + 1):
            if k <= 1:
                continue

            logger.info(f"  - Testing with {k} clusters...")
            clustering = SpectralClustering(
                n_clusters=k,
                affinity="precomputed",
                random_state=SPECTRAL_CLUSTERING_RANDOM_STATE,
                n_init=SPECTRAL_CLUSTERING_N_INIT,
            )
            # Use absolute value of correlation as affinity for clustering,
            # consistent with find_permutation_from_matrix
            affinity_matrix = np.abs(correlation_matrix)
            labels = clustering.fit_predict(affinity_matrix)

            partition = [[] for _ in range(k)]
            for node_idx, cluster_idx in enumerate(labels):
                partition[cluster_idx].append(node_idx)

            current_modularity = calculate_modularity(correlation_matrix, partition)
            logger.info(f"    - Modularity: {current_modularity:.4f}")

            if current_modularity > best_modularity:
                best_modularity = current_modularity
                best_permutation = [node for cluster in partition for node in cluster]
                logger.info(
                    f"    - New best modularity found! Optimal clusters so far: {k}"
                )

        logger.info(f"Finished search. Best modularity {best_modularity:.4f} found.")
        return best_permutation
    else:
        clustering = SpectralClustering(
            n_clusters=num_clusters,
            affinity="precomputed",
            random_state=SPECTRAL_CLUSTERING_RANDOM_STATE,
            n_init=SPECTRAL_CLUSTERING_N_INIT,
        )
        # Use absolute value of correlation as affinity for clustering,
        # consistent with find_permutation_from_matrix
        affinity_matrix = np.abs(correlation_matrix)
        labels = clustering.fit_predict(affinity_matrix)

        partition = [[] for _ in range(num_clusters)]
        for node_idx, cluster_idx in enumerate(labels):
            partition[cluster_idx].append(node_idx)

        permutation = [node for cluster in partition for node in cluster]
        return permutation


def _mamba_aware_permutation(
    correlation_matrix: np.ndarray, 
    clusters_range: Optional[Tuple[int, int]] = None
) -> List[int]:
    """
    Applies dimension-aware permutation for Mamba models that respects
    functional block boundaries while still optimizing for modularity.
    
    Mamba's hidden dimensions may have specific functional roles:
    - Input projection blocks
    - State space computation blocks  
    - Output projection blocks
    - Gate mechanism blocks
    
    Instead of global permutation, we apply local permutation within
    reasonably-sized blocks to preserve architectural integrity.
    """
    d_model = correlation_matrix.shape[0]
    
    # Define block size for local permutation - conservative approach
    # Use smaller blocks to minimize disruption to Mamba's functional structure
    if clusters_range:
        # Use the maximum cluster size as block size, but cap it
        block_size = min(clusters_range[1], 128, d_model // 8)
    else:
        # Conservative default: 64 dimensions per block
        block_size = min(64, d_model // 8)
    
    if block_size < 8:
        logger.warning(f"Block size {block_size} too small for meaningful permutation, using identity")
        return list(range(d_model))
    
    logger.info(f"  - Using block-wise permutation with block_size={block_size}")
    logger.info(f"  - Total blocks: {d_model // block_size}")
    
    global_permutation = []
    
    # Apply IASP within each functional block
    for start_idx in range(0, d_model, block_size):
        end_idx = min(start_idx + block_size, d_model)
        block_indices = list(range(start_idx, end_idx))
        
        if len(block_indices) < 4:  # Too small to meaningfully cluster
            global_permutation.extend(block_indices)
            continue
            
        # Extract block correlation matrix
        block_corr = correlation_matrix[start_idx:end_idx, start_idx:end_idx]
        
        # Determine optimal clusters for this block
        block_size_actual = len(block_indices)
        min_clusters = max(2, block_size_actual // 32)
        max_clusters = max(2, block_size_actual // 8)
        
        # Find best permutation for this block
        best_modularity = -1
        best_block_perm = list(range(block_size_actual))
        
        for k in range(min_clusters, max_clusters + 1):
            if k <= 1 or k >= block_size_actual:
                continue
                
            try:
                clustering = SpectralClustering(
                    n_clusters=k,
                    affinity="precomputed", 
                    random_state=SPECTRAL_CLUSTERING_RANDOM_STATE,
                    n_init=SPECTRAL_CLUSTERING_N_INIT,
                )
                
                affinity_matrix = np.abs(block_corr)
                labels = clustering.fit_predict(affinity_matrix)
                
                partition = [[] for _ in range(k)]
                for node_idx, cluster_idx in enumerate(labels):
                    partition[cluster_idx].append(node_idx)
                
                current_modularity = calculate_modularity(block_corr, partition)
                
                if current_modularity > best_modularity:
                    best_modularity = current_modularity
                    best_block_perm = [node for cluster in partition for node in cluster]
                    
            except Exception as e:
                logger.warning(f"Clustering failed for block {start_idx}-{end_idx}: {e}")
                continue
        
        # Convert local permutation to global indices
        global_block_perm = [start_idx + local_idx for local_idx in best_block_perm]
        global_permutation.extend(global_block_perm)
        
        logger.info(f"    Block {start_idx}-{end_idx}: modularity={best_modularity:.4f}")
    
    logger.info(f"✅ Completed dimension-aware permutation for Mamba")
    return global_permutation


def find_optimal_permutation(
    model: nn.Module,
    data_loader: DataLoader,
    target_layer_names: Optional[List[str]] = None,
    cluster_size_range: Tuple[int, int] = (16, 128),
    device: Optional[str] = None,
    *,
    # Backwards-compatibility keyword – accepted but ignored if target_layer_names provided
    target_layer_name: Optional[str] = None,
) -> List[int]:
    """
    Compute the optimal permutation for a given model layer.

    Args:
        model: The PyTorch model to analyze.
        data_loader: The dataloader providing data samples.
        target_layer_names: The name of the layer to hook for activations.
        cluster_size_range: Tuple of (min_cluster_size, max_cluster_size).
        device: Device to run the model on. If None, defaults to 'cuda' if available, else 'cpu'.

    Returns:
        A list of integers representing the optimal permutation of indices.
    """
    # ------------------------------------------------------------------
    # Parameter handling
    # ------------------------------------------------------------------
    if target_layer_names is None and target_layer_name is not None:
        target_layer_names = [target_layer_name]

    if target_layer_names is None:
        raise ValueError("Either 'target_layer_names' or 'target_layer_name' must be provided.")

    if isinstance(target_layer_names, str):
        target_layer_names = [target_layer_names]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"No device specified, auto-detected: {device}")

    # Detect model type for specialized handling
    model_type = None
    if hasattr(model, 'config') and hasattr(model.config, 'model_type'):
        model_type = model.config.model_type
        logger.info(f"Detected model type: {model_type}")
    elif hasattr(model, '__class__'):
        model_class_name = model.__class__.__name__.lower()
        if 'mamba' in model_class_name:
            model_type = 'mamba'
            logger.info(f"Detected Mamba model from class name: {model.__class__.__name__}")

    correlation_matrix = get_activation_correlation(
        model=model,
        dataloader=data_loader,
        target_layer_names=target_layer_names,
        device=device,
        target_layer_name=target_layer_name,
    )

    d_model = correlation_matrix.shape[0]
    min_size, max_size = cluster_size_range
    min_clusters = max(MIN_CLUSTER_SIZE, d_model // max_size)
    max_clusters = max(MIN_CLUSTER_SIZE, d_model // min_size)

    return find_optimal_permutation_from_matrix(
        correlation_matrix,
        clusters_range=(min_clusters, max_clusters),
        model_type=model_type,
    )
