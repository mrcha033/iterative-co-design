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


def get_activation_correlation(
    model: nn.Module,
    dataloader: DataLoader,
    target_layer_name: str,
    max_samples: int = 512,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Computes the activation correlation matrix for a target layer's output.

    This function hooks into the specified model layer, collects activations
    as the model processes data from the dataloader, and then computes a
    Pearson correlation matrix of the activations.

    Args:
        model: The PyTorch model to analyze.
        dataloader: The dataloader providing data samples.
        target_layer_name: The name of the layer to hook for activations
            (e.g., 'bert.encoder.layer.0.output.dense').
        max_samples: The maximum number of samples to use for calculation.
        device: The device to run the model on. If None, auto-detects CUDA availability.

    Returns:
        A 2D numpy array representing the Pearson correlation matrix of the
        layer's output activations.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"No device specified, auto-detected: {device}")
    
    model.to(device)
    model.eval()

    activations = []

    def hook_fn(module, input, output):
        # We handle both tensor and tuple outputs (common in transformers)
        act = output[0] if isinstance(output, tuple) else output
        # Detach and move to CPU to avoid GPU memory buildup
        activations.append(act.detach().cpu().numpy())

    # Find the target layer and register the hook
    target_layer = None
    for name, module in model.named_modules():
        if name == target_layer_name:
            target_layer = module
            break

    if target_layer is None:
        # Print debugging information
        print(f"\nDEBUG: Layer '{target_layer_name}' not found in the model.")
        print(f"DEBUG: Model type: {type(model)}")
        print(f"DEBUG: Model class: {model.__class__}")
        print(f"DEBUG: Has 'model' attribute: {hasattr(model, 'model')}")
        if hasattr(model, 'model'):
            print(f"DEBUG: Inner model type: {type(model.model)}")
        
        print("Available layers:")
        layer_count = 0
        for name, module in model.named_modules():
            if hasattr(module, 'weight') or hasattr(module, 'bias'):  # Only show actual layers
                print(f"  - {name}: {type(module).__name__}")
                layer_count += 1
                if layer_count > 20:  # Limit output
                    print("  ... (showing first 20 layers)")
                    break
        print()
        raise ValueError(f"Layer '{target_layer_name}' not found in the model.")

    handle = target_layer.register_forward_hook(hook_fn)

    # Pass data through the model to collect activations
    samples_collected = 0
    for batch in tqdm(dataloader, desc="Collecting activations"):
        if samples_collected >= max_samples:
            break

        inputs = {
            k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()
        }
        with torch.no_grad():
            _ = model(**inputs)

        # Determine how many samples were just processed using the first tensor
        first_tensor = next((v for v in inputs.values() if torch.is_tensor(v)), None)
        if first_tensor is not None:
            samples_collected += first_tensor.size(0)
        else:
            # Fallback: use the last collected activation's batch size
            samples_collected += activations[-1].shape[0]

        if samples_collected >= max_samples:
            break

    handle.remove()  # Ensure the hook is removed to prevent memory leaks

    if not activations:
        raise ValueError(
            "No activations were collected. Check the dataloader and model."
        )

    # Concatenate all collected activations
    all_activations = np.concatenate(activations, axis=0)[:max_samples]

    # Handle both 2D and 3D activation tensors
    if all_activations.ndim == 3:
        # 3D case: (total_samples, seq_len, hidden_dim) -> (total_tokens, hidden_dim)
        num_samples, seq_len, hidden_dim = all_activations.shape
        all_activations_reshaped = all_activations.reshape(
            num_samples * seq_len, hidden_dim
        )
        logger.info(
            f"Reshaped 3D activations from {all_activations.shape} to {all_activations_reshaped.shape}"
        )
    elif all_activations.ndim == 2:
        # 2D case: (total_samples, hidden_dim) - already in correct format
        all_activations_reshaped = all_activations
        logger.info(f"Using 2D activations with shape {all_activations.shape}")
    else:
        raise ValueError(
            f"Expected 2D or 3D activations, but got {all_activations.ndim}D. "
            f"Shape: {all_activations.shape}. The hook might be on an incompatible layer."
        )

    # Compute Pearson correlation matrix
    correlation_matrix = np.corrcoef(all_activations_reshaped, rowvar=False)

    # Handle NaNs that can occur from constant activation dimensions
    if np.any(np.isnan(correlation_matrix)):
        logger.warning(
            "Found NaN values in correlation matrix, likely due to constant activations. Replacing with identity matrix."
        )
        # Replace NaN values with 0 correlations, except diagonal which should be 1
        correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
        np.fill_diagonal(correlation_matrix, 1.0)

    return correlation_matrix


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
        random_state=0,  # for reproducibility
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
            min_clusters = max(2, correlation_matrix.shape[0] // 128)
            max_clusters = max(2, correlation_matrix.shape[0] // 16)
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
                random_state=0,
                n_init=10,
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
            random_state=0,
            n_init=10,
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
    target_layer_name: str,
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
        target_layer_name=target_layer_name,
        device=device,
    )

    d_model = correlation_matrix.shape[0]
    min_size, max_size = cluster_size_range
    min_clusters = max(2, d_model // max_size)
    max_clusters = max(2, d_model // min_size)

    return find_optimal_permutation_from_matrix(
        correlation_matrix,
        clusters_range=(min_clusters, max_clusters),
    )
