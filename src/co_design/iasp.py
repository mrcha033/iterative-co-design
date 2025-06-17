import torch
import numpy as np
from tqdm import tqdm
from sklearn.cluster import SpectralClustering
from .modularity import calculate_modularity
from typing import Tuple, List
import torch.nn as nn

def get_activation_correlation(model: nn.Module, 
                             data_loader: torch.utils.data.DataLoader, 
                             target_layer_name: str):
    """
    Computes the Pearson correlation matrix of a target layer's output activations.

    This function registers a forward hook on the specified layer, passes data through
    the model to collect the layer's output activations, and then computes the
    Pearson correlation matrix of the activation dimensions.

    Args:
        model: The PyTorch model to analyze.
        data_loader: A DataLoader providing sample data. Each batch is expected
                     to be a dictionary that can be passed to the model as `model(**batch)`.
        target_layer_name: The name of the layer to hook into (e.g., 'model.layers.0').

    Returns:
        A numpy array representing the Pearson correlation matrix of the activations.
    """
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

    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    # Pass data through the model to collect activations
    for batch in tqdm(data_loader, desc="Collecting activations"):
        inputs = {k: v.cuda() for k, v in batch.items() if torch.is_tensor(v)} if torch.cuda.is_available() else batch
        with torch.no_grad():
            _ = model(**inputs)

    handle.remove() # Ensure the hook is removed to prevent memory leaks

    if not activations:
        raise ValueError("No activations were collected. Check the data_loader and model.")

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
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    target_layer_name: str,
    *,
    n_clusters: int | None = None,
    cluster_size_range: Tuple[int, int] = (16, 128),
) -> List[int]:
    """Finds a permutation that groups strongly correlated dimensions.

    The function can either search for the best number of clusters within
    ``cluster_size_range`` or use a fixed ``n_clusters`` if provided.
    """

    print("Finding optimal permutation by maximizing modularity...")
    correlation_matrix = get_activation_correlation(model, data_loader, target_layer_name)
    d_model = correlation_matrix.shape[0]

    # If n_clusters is specified, directly compute the permutation
    if n_clusters is not None:
        clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            random_state=0,
            n_init=10,
        )
        labels = clustering.fit_predict(correlation_matrix)

        partition = [[] for _ in range(n_clusters)]
        for node_idx, cluster_idx in enumerate(labels):
            partition[cluster_idx].append(node_idx)

        permutation = [node for cluster in partition for node in cluster]
        return permutation

    best_modularity = -1
    best_permutation = list(range(d_model))

    min_clusters = max(2, d_model // cluster_size_range[1])
    max_clusters = max(2, d_model // cluster_size_range[0])

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

def find_optimal_permutation_for_config(model, data_loader, iasp_config):
    return find_optimal_permutation(
        model=model,
        data_loader=data_loader,
        target_layer_name=iasp_config['target_layer_name'],
        n_clusters=iasp_config['n_clusters']
    ) 