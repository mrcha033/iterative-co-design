"""
IO-Aware Scan Permutation (IASP) module.

This module optimizes memory layout by finding permutations of model dimensions
that maximize data locality, thereby reducing memory access latency. It is
particularly effective for models like Mamba where I/O is a bottleneck.

The main entry point for Mamba models is `run_iasp_on_mamba`.
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
import fnmatch

from numpy.linalg import LinAlgError
from omegaconf import DictConfig

# --- Setup ---
logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_MAX_SAMPLES = 4096
MIN_CLUSTER_SIZE = 2
NAN_REPLACEMENT_VALUE = 0.0
DIAGONAL_CORRELATION_VALUE = 1.0


# --- Internal Helper Functions ---

def _expand_wildcard_layer_names(model: nn.Module, target_spec) -> List[str]:
    """Expands a wildcard target layer specification into a list of layer names."""
    if not target_spec:
        return []

    all_layers = [name for name, _ in model.named_modules()]
    expanded_layers = []

    spec_list = target_spec if isinstance(target_spec, list) else [target_spec]

    for pattern in spec_list:
        if "*" in pattern:
            matched_layers = fnmatch.filter(all_layers, pattern)
            if not matched_layers:
                logger.warning(f"Wildcard '{pattern}' did not match any layers.")
            expanded_layers.extend(matched_layers)
        else:
            # If no wildcard, add it as a direct layer name
            expanded_layers.append(pattern)

    return expanded_layers


def _get_activation_correlation(
    model: nn.Module,
    dataloader: DataLoader,
    target_layer_names: List[str],
    is_mamba_in_proj: bool,
    max_samples: int = DEFAULT_MAX_SAMPLES,
    device: Optional[str] = None,
) -> np.ndarray:
    """Computes activation correlation, with special handling for Mamba's in_proj."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    activations = {name: [] for name in target_layer_names}
    hooks = []

    def create_hook(layer_name):
        def hook_fn(module, input, output):
            act = output
            if is_mamba_in_proj:
                d_inner = module.out_features // 2
                act = output[..., :d_inner] # Crucially, target the 'x' part of d_inner

            if act.ndim == 3:
                act = act.reshape(-1, act.shape[-1])
            activations[layer_name].append(act.detach().cpu())
        return hook_fn

    for name in target_layer_names:
        try:
            layer = model.get_submodule(name)
            hooks.append(layer.register_forward_hook(create_hook(name)))
        except AttributeError:
            for h in hooks:
                h.remove()
            raise ValueError(f"Layer '{name}' not found.")

    collected_tokens = 0
    pbar = tqdm(dataloader, desc="1/3: Collecting Activations", leave=False)
    for batch in pbar:
        if collected_tokens >= max_samples:
            break
        inputs = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        with torch.no_grad():
            _ = model(**inputs)
        collected_tokens += next(iter(inputs.values())).shape[0] * next(iter(inputs.values())).shape[1]
        pbar.set_postfix({"tokens": f"{collected_tokens}/{max_samples}"})
    for h in hooks:
        h.remove()

    correlation_matrices = []
    for name, acts_list in activations.items():
        if not acts_list:
            continue
        all_acts = torch.cat(acts_list, dim=0)[:max_samples]
        corr_matrix = torch.corrcoef(all_acts.T).cpu().numpy()
        if np.any(np.isnan(corr_matrix)):
            corr_matrix = np.nan_to_num(corr_matrix, nan=NAN_REPLACEMENT_VALUE)
            np.fill_diagonal(corr_matrix, DIAGONAL_CORRELATION_VALUE)
        correlation_matrices.append(corr_matrix)

    if not correlation_matrices:
        raise ValueError("Failed to compute correlation.")
    return np.mean(correlation_matrices, axis=0)


def _find_optimal_permutation(
    correlation_matrix: np.ndarray,
    clusters_range: Tuple[int, int],
    iasp_config: DictConfig,
) -> Tuple[List[int], float]:
    """Finds the permutation that maximizes modularity and returns it with the score."""
    dim = correlation_matrix.shape[0]
    best_modularity, best_permutation = -np.inf, list(range(dim))
    affinity_matrix = np.abs(correlation_matrix)

    n_init = iasp_config.get("spectral_n_init", 10)
    random_state = iasp_config.get("spectral_random_state", 0)

    search_space = range(clusters_range[0], clusters_range[1] + 1)
    pbar = tqdm(search_space, desc="2/3: Finding Optimal Permutation", leave=False)
    for k in pbar:
        if not (1 < k < dim):
            continue
        try:
            clustering = SpectralClustering(
                n_clusters=k,
                affinity="precomputed",
                random_state=random_state,
                n_init=n_init,
            ).fit(affinity_matrix)

            partition = [np.where(clustering.labels_ == i)[0] for i in range(k)]
            modularity = calculate_modularity(correlation_matrix, partition)

            if modularity > best_modularity:
                best_modularity = modularity
                best_permutation = [node for cluster in partition for node in cluster]
                pbar.set_postfix({"best_modularity": f"{best_modularity:.4f}", "k": k})
        except LinAlgError:
            logger.warning(f"Spectral clustering failed for k={k} due to a linear algebra error (e.g., matrix not positive definite). Skipping.")
            continue
        except Exception as e:
            logger.error(f"An unexpected error occurred during spectral clustering for k={k}: {e}")
            continue

    logger.info(f"Found optimal permutation with modularity: {best_modularity:.4f}")
    return best_permutation, best_modularity


def _apply_permutation_to_mamba(model: nn.Module, permutation: List[int]):
    """Applies a permutation to Mamba weights while preserving mathematical equivalence."""
    p = torch.tensor(permutation, dtype=torch.long)
    pbar = tqdm(model.named_modules(), desc="3/3: Applying Permutation", leave=False)
    for name, layer in pbar:
        if all(hasattr(layer, attr) for attr in ["in_proj", "out_proj", "dt_proj", "A_log", "D"]):
            pbar.set_postfix({"layer": name})
            d_inner = layer.out_proj.in_features
            if p.numel() != d_inner:
                continue

            dev = layer.in_proj.weight.device
            p = p.to(dev)

            with torch.no_grad():
                # Permute output of in_proj (P @ W)
                w_in, b_in = layer.in_proj.weight.data, layer.in_proj.bias.data
                layer.in_proj.weight.data.copy_(torch.cat((w_in[:d_inner][p], w_in[d_inner:][p])))
                if b_in is not None:
                    layer.in_proj.bias.data.copy_(torch.cat((b_in[:d_inner][p], b_in[d_inner:][p])))
                
                # Permute inputs of subsequent layers (W @ P^T -> W[:, p])
                layer.dt_proj.weight.data = layer.dt_proj.weight.data[:, p]
                layer.out_proj.weight.data = layer.out_proj.weight.data[:, p]
                
                # Permute parameters aligned with d_inner
                layer.A_log.data = layer.A_log.data[:, p]
                layer.D.data = layer.D.data[p]


# --- Main Public Function ---

def run_iasp_on_mamba(
    model: nn.Module,
    dataloader: DataLoader,
    iasp_config: DictConfig,
    device: Optional[str] = None,
) -> Tuple[List[int], float]:
    """
    Runs the full IASP optimization pipeline on a Mamba model.

    This function automates:
    1. Finding the Mamba 'in_proj' layers.
    2. Collecting activations for the `d_inner` dimension.
    3. Finding the optimal permutation that maximizes modularity.
    4. Applying the permutation consistently to all Mamba blocks.

    Args:
        model: The Mamba model to be optimized.
        dataloader: DataLoader for collecting sample activations.
        iasp_config: Configuration for IASP optimization.
        device: The device to run on (e.g., 'cuda:0'). Auto-detected if None.

    Returns:
        A tuple containing the optimal permutation and its modularity score.
    """
    logger.info("🚀 Starting IASP optimization pipeline for Mamba model...")

    target_layer_spec = iasp_config.get("target_layer_names")
    if target_layer_spec:
        logger.info(f"Expanding target layer spec: {target_layer_spec}")
        target_layer_names = _expand_wildcard_layer_names(model, target_layer_spec)
    else:
        logger.info("Auto-detecting Mamba 'in_proj' layers...")
        target_layer_names = [
            name for name, mod in model.named_modules()
            if name.endswith("in_proj") and isinstance(mod, nn.Linear)
        ]
    
    if not target_layer_names:
        raise ValueError("Could not find any target layers. Please check `target_layer_names` in your config.")
    logger.info(f"Found {len(target_layer_names)} target layers.")

    # Step 1: Get activation correlation for the d_inner dimension
    correlation_matrix = _get_activation_correlation(
        model, dataloader, target_layer_names, is_mamba_in_proj=True, device=device,
        max_samples=iasp_config.get("max_samples", DEFAULT_MAX_SAMPLES)
    )

    # Step 2: Find the optimal permutation
    d_inner = correlation_matrix.shape[0]
    min_size, max_size = iasp_config.cluster_size_range
    min_clusters = max(MIN_CLUSTER_SIZE, d_inner // max_size)
    max_clusters = max(MIN_CLUSTER_SIZE, d_inner // min_size)
    
    permutation, modularity = _find_optimal_permutation(
        correlation_matrix, clusters_range=(min_clusters, max_clusters), iasp_config=iasp_config
    )

    # Step 3: Apply the permutation to the model
    _apply_permutation_to_mamba(model, permutation)

    logger.info("✅ IASP optimization for Mamba completed successfully.")
    return permutation, modularity


def _apply_permutation_to_bert_ffn(model: nn.Module, permutation: List[int], target_layer_names: List[str]):
    """Applies a permutation to the specified FFN layers of a BERT-like model."""
    p = torch.tensor(permutation, dtype=torch.long)
    d_ffn = len(permutation)
    
    pbar = tqdm(target_layer_names, desc="3/3: Applying Permutation to BERT FFN", leave=False)
    for layer_name in pbar:
        # Get the parent module of the 'intermediate.dense' layer
        parent_module_name = ".".join(layer_name.split('.')[:-2])
        parent_module = model.get_submodule(parent_module_name)

        if hasattr(parent_module, 'intermediate') and hasattr(parent_module, 'output'):
            if hasattr(parent_module.intermediate, 'dense') and hasattr(parent_module.output, 'dense'):
                up_proj = parent_module.intermediate.dense
                down_proj = parent_module.output.dense
                
                # Verify that the dimensions match our permutation
                if up_proj.out_features != d_ffn or down_proj.in_features != d_ffn:
                    continue

                pbar.set_postfix({"layer": layer_name})
                dev = up_proj.weight.device
                p = p.to(dev)

                with torch.no_grad():
                    # Permute the output of the 'up' projection (rows) -> P @ W
                    up_proj.weight.data = up_proj.weight.data[p]
                    if up_proj.bias is not None:
                        up_proj.bias.data = up_proj.bias.data[p]

                    # Permute the input of the 'down' projection (columns) -> W @ P^T
                    down_proj.weight.data = down_proj.weight.data[:, p]


def run_iasp_on_bert(
    model: nn.Module,
    dataloader: DataLoader,
    iasp_config: DictConfig,
    device: Optional[str] = None,
) -> Tuple[List[int], float]:
    """
    Runs the full IASP optimization pipeline on a BERT-like model's FFN layers.

    This function automates:
    1. Finding the FFN 'up-projection' layers (e.g., `intermediate.dense`).
    2. Collecting activations for the FFN intermediate dimension.
    3. Finding the optimal permutation that maximizes modularity.
    4. Applying the permutation consistently to all FFN blocks.

    Args:
        model: The BERT-like model to be optimized.
        dataloader: DataLoader for collecting sample activations.
        iasp_config: Configuration for IASP optimization.
        device: The device to run on. Auto-detected if None.

    Returns:
        A tuple containing the optimal permutation and its modularity score.
    """
    logger.info("🚀 Starting IASP optimization pipeline for BERT model...")

    target_layer_spec = iasp_config.get("target_layer_names")
    if target_layer_spec:
        logger.info(f"Expanding target layer spec: {target_layer_spec}")
        target_layer_names = _expand_wildcard_layer_names(model, target_layer_spec)
    else:
        logger.info("Auto-detecting BERT FFN 'up-projection' layers...")
        target_layer_names = [
            name for name, mod in model.named_modules()
            if name.endswith("intermediate.dense") and isinstance(mod, nn.Linear)
        ]

    if not target_layer_names:
        raise ValueError("Could not find any target layers ('*.intermediate.dense'). Please specify them manually.")
    logger.info(f"Found {len(target_layer_names)} target layers.")

    # Step 1: Get activation correlation for the FFN intermediate dimension
    correlation_matrix = _get_activation_correlation(
        model, dataloader, target_layer_names, is_mamba_in_proj=False, device=device,
        max_samples=iasp_config.get("max_samples", DEFAULT_MAX_SAMPLES)
    )

    # Step 2: Find the optimal permutation
    d_ffn = correlation_matrix.shape[0]
    min_size, max_size = iasp_config.cluster_size_range
    min_clusters = max(MIN_CLUSTER_SIZE, d_ffn // max_size)
    max_clusters = max(MIN_CLUSTER_SIZE, d_ffn // min_size)
    
    permutation, modularity = _find_optimal_permutation(
        correlation_matrix, clusters_range=(min_clusters, max_clusters), iasp_config=iasp_config
    )

    # Step 3: Apply the permutation to the model
    _apply_permutation_to_bert_ffn(model, permutation, target_layer_names)

    logger.info("✅ IASP optimization for BERT completed successfully.")
    return permutation, modularity