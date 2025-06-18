import torch
import numpy as np
from tqdm import tqdm
from sklearn.cluster import SpectralClustering
from .modularity import calculate_modularity
from typing import List, Optional
import torch.nn as nn
from torch.utils.data import DataLoader

def get_activation_correlation(
    model: nn.Module,
    dataloader: DataLoader,
    target_layer_name: str,
    max_samples: int = 512,
    device: str = "cuda"
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
        device: The device to run the model on.

    Returns:
        A 2D numpy array representing the Pearson correlation matrix of the
        layer's output activations.
    """
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
        raise ValueError(f"Layer '{target_layer_name}' not found in the model.")

    handle = target_layer.register_forward_hook(hook_fn)

    # Pass data through the model to collect activations
    for batch in tqdm(dataloader, desc="Collecting activations"):
        inputs = {k: v.cuda() for k, v in batch.items() if torch.is_tensor(v)} if torch.cuda.is_available() else batch
        with torch.no_grad():
            _ = model(**inputs)

    handle.remove() # Ensure the hook is removed to prevent memory leaks

    if not activations:
        raise ValueError("No activations were collected. Check the dataloader and model.")

    # Concatenate all collected activations. The shape is typically (num_batches, batch_size, seq_len, hidden_dim).
    # We need to reshape it to (total_tokens, hidden_dim).
    all_activations = np.concatenate(activations, axis=0)
    
    # Reshape from (total_samples, seq_len, hidden_dim) to (total_tokens, hidden_dim)
    if all_activations.ndim != 3:
        raise ValueError(f"Expected 3D activations, but got {all_activations.ndim}D. The hook might be on an incompatible layer.")
    
    num_samples, seq_len, hidden_dim = all_activations.shape
    all_activations_reshaped = all_activations.reshape(num_samples * seq_len, hidden_dim)
    
    # Compute Pearson correlation matrix
    correlation_matrix = np.corrcoef(all_activations_reshaped, rowvar=False)
    
    return correlation_matrix


def find_permutation_from_matrix(correlation_matrix: np.ndarray, n_clusters: int) -> list[int]:
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
    print("Step 2a: Performing spectral clustering...")
    # We use the absolute value of correlation as affinity for clustering,
    # as strong negative correlations are also meaningful for grouping.
    affinity_matrix = np.abs(correlation_matrix)
    
    clustering = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        random_state=0  # for reproducibility
    ).fit(affinity_matrix)

    labels = clustering.labels_
    
    print("Step 2b: Constructing permutation from clusters...")
    # argsort provides a permutation that groups indices by their corresponding labels.
    # Using a stable sort for robustness.
    permutation = np.argsort(labels, kind='mergesort')
    
    return permutation.tolist()


def find_optimal_permutation(
    correlation_matrix: np.ndarray,
    num_clusters: Optional[int] = None
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
        # Search for the best number of clusters to maximize modularity
        best_modularity = -1
        best_permutation = list(range(correlation_matrix.shape[0]))

        min_clusters = max(2, correlation_matrix.shape[0] // 128)
        max_clusters = max(2, correlation_matrix.shape[0] // 16)

        for k in range(min_clusters, max_clusters + 1):
            if k <= 1:
                continue

            print(f"  - Testing with {k} clusters...")
            clustering = SpectralClustering(
                n_clusters=k,
                affinity='precomputed',
                random_state=0,
                n_init=10,
            )
            labels = clustering.fit_predict(correlation_matrix)

            partition = [[] for _ in range(k)]
            for node_idx, cluster_idx in enumerate(labels):
                partition[cluster_idx].append(node_idx)

            current_modularity = calculate_modularity(correlation_matrix, partition)
            print(f"    - Modularity: {current_modularity:.4f}")

            if current_modularity > best_modularity:
                best_modularity = current_modularity
                best_permutation = [node for cluster in partition for node in cluster]
                print(f"    - New best modularity found! Optimal clusters so far: {k}")

        print(f"Finished search. Best modularity {best_modularity:.4f} found.")
        return best_permutation
    else:
        clustering = SpectralClustering(
            n_clusters=num_clusters,
            affinity='precomputed',
            random_state=0,
            n_init=10,
        )
        labels = clustering.fit_predict(correlation_matrix)

        partition = [[] for _ in range(num_clusters)]
        for node_idx, cluster_idx in enumerate(labels):
            partition[cluster_idx].append(node_idx)

        permutation = [node for cluster in partition for node in cluster]
        return permutation

def find_optimal_permutation_for_config(model, data_loader, iasp_config):
    return find_optimal_permutation(
        correlation_matrix=get_activation_correlation(
            model=model,
            dataloader=data_loader,
            target_layer_name=iasp_config['target_layer_name'],
            max_samples=iasp_config['max_samples'],
            device=iasp_config['device']
        ),
        num_clusters=iasp_config['n_clusters']
    ) 