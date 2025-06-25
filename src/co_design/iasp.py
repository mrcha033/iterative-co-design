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
    target_layer_names: List[str],
    max_samples: int = DEFAULT_MAX_ACTIVATION_SAMPLES,
    device: Optional[str] = None,
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
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"No device specified, auto-detected: {device}")

    model.to(device)
    model.eval()

    correlation_matrices = []

    for target_layer_name in target_layer_names:
        activations = []

        def hook_fn(module, input, output):
            act = output[0] if isinstance(output, tuple) else output
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
) -> List[int]:
    """
    Finds the optimal permutation of neuron indices to maximize modularity.

    This function takes a correlation matrix and performs spectral clustering to
    group correlated neurons. It then orders the neurons based on these
    clusters to find a permutation that improves data locality. If num_clusters
    is not provided, it searches for the optimal number of clusters that
    maximizes the modularity score.

    Args:
        correlation_matrix: A 2D numpy array of activation correlations.
        num_clusters: The number of clusters to form. If None, the optimal
            number of clusters will be determined automatically.

    Returns:
        A list of integers representing the optimal permutation of indices.
    """
    if num_clusters is None:
        # Determine search bounds
        if clusters_range is None:
            min_clusters = max(MIN_CLUSTER_SIZE, correlation_matrix.shape[0] // DEFAULT_MIN_CLUSTERS_DIVISOR)
            max_clusters = max(MIN_CLUSTER_SIZE, correlation_matrix.shape[0] // DEFAULT_MAX_CLUSTERS_DIVISOR)
        else:
            min_clusters, max_clusters = clusters_range

        # Search for the best number of clusters to maximize modularity
        best_modularity = -1
        best_permutation = list(range(correlation_matrix.shape[0]))

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


def find_optimal_permutation(
    model: nn.Module,
    data_loader: DataLoader,
    target_layer_names: List[str],
    cluster_size_range: Tuple[int, int],
    device: Optional[str] = None,
) -> List[int]:
    """
    Compute the optimal permutation for a given model layer.

    Args:
        model: The PyTorch model to analyze.
        data_loader: The dataloader providing data samples.
        target_layer_name: The name of the layer to hook for activations.
        cluster_size_range: Tuple of (min_cluster_size, max_cluster_size).
        device: Device to run the model on. If None, defaults to 'cuda' if available, else 'cpu'.

    Returns:
        A list of integers representing the optimal permutation of indices.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"No device specified, auto-detected: {device}")

    correlation_matrix = get_activation_correlation(
        model=model,
        dataloader=data_loader,
        target_layer_names=target_layer_names,
        device=device,
    )

    d_model = correlation_matrix.shape[0]
    min_size, max_size = cluster_size_range
    min_clusters = max(MIN_CLUSTER_SIZE, d_model // max_size)
    max_clusters = max(MIN_CLUSTER_SIZE, d_model // min_size)

    return find_optimal_permutation_from_matrix(
        correlation_matrix,
        clusters_range=(min_clusters, max_clusters),
    )
