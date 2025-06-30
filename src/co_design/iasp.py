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
from utils.permutation import permute_rows, permute_cols, permute_vector, permute_in_proj_split

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
    sample_stride: int = 1,
    device: Optional[str] = None,
) -> np.ndarray:
    """Computes activation correlation, with special handling for Mamba's in_proj."""
    model.eval()  # Ensure model is in eval mode, but don't move device here
    device = next(model.parameters()).device  # Infer device from model

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
            # Asynchronous copy to CPU with fp16 for memory efficiency
            activations[layer_name].append(act.to("cpu", dtype=torch.float16, non_blocking=True))
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
    pbar = tqdm(dataloader, desc="1/3: Collecting Activations", miniters=1, ncols=100, leave=True)
    for i, batch in enumerate(pbar):
        if collected_tokens >= max_samples:
            break
        # Apply sample stride
        if i % sample_stride != 0:
            continue

        inputs = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        with torch.no_grad():
            _ = model(**inputs)
        
        # More accurate token counting from input_ids
        num_new_tokens = inputs['input_ids'].numel()
        collected_tokens += num_new_tokens
        pbar.set_postfix({"tokens": f"{collected_tokens}/{max_samples}"})

        # Enhanced logging every 10 steps
        if (pbar.n + 1) % 10 == 0:
            logger.debug(f"Step {pbar.n+1}: Collected {collected_tokens} tokens so far.")

    for h in hooks:
        h.remove()

    correlation_matrices = []
    for name, acts_list in activations.items():
        if not acts_list:
            continue
        
        # PyTorch < 2.2 may not have a stable float16/bfloat16 kernel for corrcoef
        if torch.__version__ < "2.2.0":
            all_acts_fp32 = torch.cat(acts_list, dim=0)[:max_samples].to(torch.float32)
            corr_matrix = torch.corrcoef(all_acts_fp32.T).cpu().numpy()
        else:
            # Use float16 for memory efficiency on newer PyTorch versions
            all_acts_fp16 = torch.cat(acts_list, dim=0)[:max_samples].to(torch.float16)
            # Calculate correlation and cast back to float32 for stability
            corr_matrix = torch.corrcoef(all_acts_fp16.T).float().cpu().numpy()

        correlation_matrices.append(corr_matrix)

    if not correlation_matrices:
        raise ValueError("Failed to compute correlation.")
    
    final_corr = np.mean(correlation_matrices, axis=0)
    # Ensure diagonal is always 1.0, as self-correlation should be perfect.
    np.fill_diagonal(final_corr, 1.0)
    return final_corr


def _find_optimal_permutation(
    correlation_matrix: np.ndarray,
    clusters_range: Tuple[int, int],
    iasp_config: DictConfig,
) -> Tuple[List[int], float]:
    """Finds the permutation that maximizes modularity and returns it with the score."""
    dim = correlation_matrix.shape[0]
    best_modularity, best_permutation = -np.inf, list(range(dim))

    # --- Robust Preprocessing Step ---
    # 1. Ensure the correlation matrix is fully finite before proceeding.
    #    Use np.where to preserve sign information while handling NaNs.
    if not np.all(np.isfinite(correlation_matrix)):
        logger.warning("Correlation matrix contains non-finite values (NaN or Inf). Cleaning up...")
        correlation_matrix = np.where(np.isnan(correlation_matrix), 0.0, correlation_matrix)
        correlation_matrix = np.clip(correlation_matrix, -1.0, 1.0) # Clamp infs

    # 2. Create a positive-only affinity matrix to group co-activated neurons.
    affinity_matrix = np.clip(correlation_matrix, 0, None).astype(np.float64)

    # 3. Enforce perfect symmetry to prevent sklearn warnings.
    affinity_matrix = (affinity_matrix + affinity_matrix.T) / 2

    # 4. Set the diagonal to 0 to remove self-loops.
    np.fill_diagonal(affinity_matrix, 0)
    # --- End of Preprocessing ---

    n_init = iasp_config.get("spectral_n_init", 10)
    random_state = iasp_config.get("spectral_random_state", 0)

    search_space = range(clusters_range[0], clusters_range[1] + 1)
    pbar = tqdm(search_space, desc="2/3: Finding Optimal Permutation", leave=False)
    for k in pbar:
        if not (1 < k < dim):
            continue
        try:
            # Pass the fully preprocessed matrix to clustering.
            clustering = SpectralClustering(
                n_clusters=k,
                affinity="precomputed",
                random_state=random_state,
                n_init=n_init,
                assign_labels="kmeans", # K-means is more stable for a wide range of k
                eigen_tol=1e-5,
            ).fit(affinity_matrix)

            partition = [np.where(clustering.labels_ == i)[0] for i in range(k)]
            modularity = calculate_modularity(correlation_matrix, partition)

            if modularity > best_modularity:
                best_modularity = modularity
                best_permutation = [node for cluster in partition for node in cluster]
                pbar.set_postfix({"best_modularity": f"{best_modularity:.4f}", "k": k})
        except Exception as e:
            # Catch any exception during clustering for a specific k and log it,
            # allowing the search for a valid k to continue.
            logger.warning(f"Clustering failed for k={k} with error: {e}. Skipping.")
            continue

    logger.info(f"Found optimal permutation with modularity: {best_modularity:.4f}")
    return best_permutation, best_modularity


def _apply_permutation_to_mamba(mamba_mixers: List[nn.Module], permutation: List[int]):
    """Applies a permutation to a list of Mamba mixer blocks consistently."""
    if not mamba_mixers:
        return

    # Create permutation tensor once on the correct device for robustness.
    dev = mamba_mixers[0].in_proj.weight.device
    p = torch.tensor(permutation, dtype=torch.long, device=dev)

    pbar = tqdm(mamba_mixers, desc="3/3: Applying Permutation to Mamba Mixers", leave=False)
    for layer in pbar:
        d_inner = layer.out_proj.in_features
        assert p.numel() == d_inner, f"Permutation length {p.numel()} mismatch with d_inner {d_inner}"

        with torch.no_grad():
            # Use grad-safe helpers to reconstruct autograd graph
            if layer.in_proj.bias is not None:
                layer.in_proj.weight, layer.in_proj.bias = permute_in_proj_split(
                    layer.in_proj.weight, p, layer.in_proj.bias
                )
            else:
                layer.in_proj.weight = permute_in_proj_split(layer.in_proj.weight, p)

            # Handle potential API drift in mamba-ssm (conv1d vs conv1d_proj)
            conv_layer = None
            if hasattr(layer, "conv1d_proj"):
                conv_layer = layer.conv1d_proj
            elif hasattr(layer, "conv1d"):
                conv_layer = layer.conv1d

            if conv_layer:
                w = conv_layer.weight
                if w.shape[0] == d_inner:
                    conv_layer.weight = permute_rows(w, p)
                elif w.shape[1] == d_inner:
                    conv_layer.weight = permute_cols(w, p)
                
                if conv_layer.bias is not None and conv_layer.bias.numel() == d_inner:
                    conv_layer.bias = permute_vector(conv_layer.bias, p)
            
            if hasattr(layer, 'x_proj'):
                 layer.x_proj.weight = permute_cols(layer.x_proj.weight, p)

            if hasattr(layer, 'dt_proj'):
                 W = layer.dt_proj.weight
                 if W.shape[0] == d_inner:
                     layer.dt_proj.weight = permute_rows(W, p)
                 elif W.shape[1] == d_inner:
                     layer.dt_proj.weight = permute_cols(W, p)

            layer.A_log = permute_rows(layer.A_log, p)
            layer.D = permute_vector(layer.D, p)
            layer.out_proj.weight = permute_cols(layer.out_proj.weight, p)


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

    # 1. 순열을 적용할 실제 mixer 모듈 객체들을 찾습니다.
    #    target_layer_names는 '...in_proj' 이므로, 부모 모듈이 mixer입니다.
    mamba_mixers_to_permute = []
    for name in target_layer_names:
        # Get the parent module (the mixer) of the in_proj layer
        parent_module_name = ".".join(name.split('.')[:-1])
        if not parent_module_name:
             logger.warning(f"Could not determine parent module for top-level layer '{name}', skipping.")
             continue
        try:
            parent_module = model.get_submodule(parent_module_name)
            # Basic validation to ensure it's a Mamba mixer
            if "MambaMixer" in parent_module.__class__.__name__ and hasattr(parent_module, 'in_proj'):
                mamba_mixers_to_permute.append(parent_module)
            else:
                logger.warning(f"Parent module '{parent_module_name}' for '{name}' is not a valid MambaMixer, skipping.")

        except AttributeError:
            logger.warning(f"Could not find parent module for {name}, skipping.")
            continue
    
    if not mamba_mixers_to_permute:
        raise ValueError("Found no valid Mamba mixer blocks to apply permutation.")
    logger.info(f"Identified {len(mamba_mixers_to_permute)} Mamba mixer blocks to permute.")

    # Step 1: Get activation correlation for the d_inner dimension
    correlation_matrix = _get_activation_correlation(
        model, dataloader, target_layer_names, is_mamba_in_proj=True, device=device,
        max_samples=iasp_config.get("max_samples", DEFAULT_MAX_SAMPLES),
        sample_stride=iasp_config.get("sample_stride", 1)
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
    _apply_permutation_to_mamba(mamba_mixers_to_permute, permutation)

    logger.info("✅ IASP optimization for Mamba completed successfully.")
    return permutation, modularity


def _apply_permutation_to_bert_ffn(model: nn.Module, permutation: List[int], target_layer_names: List[str]):
    """Applies a permutation to the specified FFN layers of a BERT-like model."""
    if not target_layer_names:
        return
        
    # Create permutation tensor once on the correct device.
    dev = model.get_submodule(target_layer_names[0]).weight.device
    p = torch.tensor(permutation, dtype=torch.long, device=dev)
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

                with torch.no_grad():
                    # Re-wrap in nn.Parameter to be gradient-safe
                    up_proj.weight = nn.Parameter(up_proj.weight.data[p])
                    if up_proj.bias is not None:
                        up_proj.bias = nn.Parameter(up_proj.bias.data[p])

                    down_proj.weight = nn.Parameter(down_proj.weight.data[:, p])
                
                # The LayerNorm after the FFN's residual connection acts on the un-permuted
                # hidden state, so its parameters should NOT be permuted.
                if hasattr(parent_module, 'output') and hasattr(parent_module.output, 'LayerNorm'):
                    pass # Explicitly do nothing
    
    # After permuting the FFNs, we need to handle the very final LayerNorm,
    # if the model has one and it acts on the hidden dimension that was NOT permuted.
    # In standard BERT, the final LN is outside the encoder layers.
    # For now, we assume the permutation is self-contained within the FFN.

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
        max_samples=iasp_config.get("max_samples", DEFAULT_MAX_SAMPLES),
        sample_stride=iasp_config.get("sample_stride", 1)
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