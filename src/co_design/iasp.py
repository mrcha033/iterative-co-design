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
from utils.permutation import (
    inplace_permute_rows,
    inplace_permute_cols,
    inplace_permute_vector,
    inplace_permute_in_proj_split,
    alias_free_rows_slice,
)

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
            # Keep activations on GPU in fp16 for fast correlation calculation.
            # The non_blocking copy to CPU was moved to the GPU for the heavy lifting.
            activations[layer_name].append(act.detach().to(torch.float16))
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
        
        # Concatenate all activation batches (stored on GPU in fp16).
        all_acts = torch.cat(acts_list, dim=0)[:max_samples]

        # UP-CAST to float32 for stable correlation calculation to prevent NaN/Inf.
        all_acts_fp32 = all_acts.to(torch.float32)

        # Pre-process to remove zero-variance columns that cause NaNs in corrcoef
        std_devs = all_acts_fp32.std(dim=0)
        valid_indices = std_devs > 1e-6
        if valid_indices.sum() < all_acts_fp32.shape[1]:
            logger.warning(f"Removed {all_acts_fp32.shape[1] - valid_indices.sum()} zero-variance columns before calculating correlation.")
        
        # If all columns have zero variance, skip
        if valid_indices.sum() == 0:
            logger.error("All activation channels have zero variance. Cannot compute correlation.")
            continue

        acts_for_corr = all_acts_fp32[:, valid_indices]

        # Now compute correlation on the stable, valid tensor.
        corr_matrix = torch.corrcoef(acts_for_corr.T).cpu().numpy()

        correlation_matrices.append((corr_matrix, valid_indices))

    if not correlation_matrices:
        raise ValueError("Failed to compute any valid correlation matrices.")
    
    # For now, we use the first valid correlation matrix found.
    # A more advanced approach might average them if they share the same valid indices.
    final_corr, final_valid_indices = correlation_matrices[0]
    
    # Ensure diagonal is always 1.0, as self-correlation should be perfect.
    np.fill_diagonal(final_corr, 1.0)
    return final_corr, final_valid_indices


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

    # 2. Create a k-NN graph to ensure connectivity and focus on strong connections.
    #    This is more robust than simple thresholding.
    affinity_matrix = (correlation_matrix + 1) / 2
    
    # Get k for k-NN from config, with a reasonable default.
    k = iasp_config.get("knn_k", 128)
    if not (0 < k < affinity_matrix.shape[0]):
        logger.warning(f"k={k} for k-NN is out of valid range, defaulting to 128.")
        k = 128
        
    knn_graph = np.zeros_like(affinity_matrix)
    for i in range(affinity_matrix.shape[0]):
        # Get the indices of the top k neighbors for node i
        top_k_indices = np.argpartition(affinity_matrix[i, :], -k)[-k:]
        # Keep only the edges to these top k neighbors
        knn_graph[i, top_k_indices] = affinity_matrix[i, top_k_indices]
        
    # Symmetrize the graph to ensure undirected edges for spectral clustering
    affinity_matrix = np.maximum(knn_graph, knn_graph.T)

    # 3. Enforce perfect symmetry to prevent sklearn warnings (final check).
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


def _permute_params_in_module(module: nn.Module, p: torch.Tensor, p_inv: torch.Tensor):
    """
    Permutes parameters within a given module based on their shape, skipping known special cases.
    """
    d_inner = p.numel()
    for name, param in module.named_parameters(recurse=False):
        # This function handles generic parameters. Special modules like in_proj
        # or weight_normed conv1d are handled separately.
        if name.endswith("out_proj.weight"): # Special handling for out_proj
            if param.shape[1] == d_inner:
                inplace_permute_cols(param, p_inv)
            elif param.shape[0] == d_inner: # Handle transposed layout
                inplace_permute_rows(param, p)
            continue

        if param.ndim == 1 and param.numel() == d_inner:
            inplace_permute_vector(param, p)
        elif param.ndim == 2:
            if param.shape[0] == d_inner:
                inplace_permute_rows(param, p)
            elif param.shape[1] == d_inner:
                inplace_permute_cols(param, p_inv)
        elif param.ndim == 3 and param.shape[0] == d_inner:
             # Handle 3D tensors like non-weight-normed Conv1d
             logger.debug(f"Auto-permuting 3D rows of '{name}' in {module.__class__.__name__}")
             flat_param = param.data.view(d_inner, -1)
             new_flat_data = flat_param.index_select(0, p).clone()
             param.data.copy_(new_flat_data.view_as(param))


def _apply_permutation_to_mamba(mamba_mixers: List[nn.Module], permutation: List[int]):
    """Applies a permutation to a list of Mamba mixer blocks consistently."""
    if not mamba_mixers:
        return

    dev = mamba_mixers[0].in_proj.weight.device
    p = torch.tensor(permutation, dtype=torch.long, device=dev)
    p_inv = torch.empty_like(p)
    p_inv[p] = torch.arange(p.numel(), device=dev)
    d_inner = p.numel()
    
    pbar = tqdm(mamba_mixers, desc="3/3: Applying Permutation to Mamba Mixers", leave=False)
    for layer in pbar:
        seen_modules = set() # Track handled modules to prevent double permutation

        # --- Step 1: Handle Special Cases First ---
        
        # 1a. Handle the split in_proj layer
        if hasattr(layer, 'in_proj'):
            inplace_permute_in_proj_split(layer.in_proj.weight, p, getattr(layer.in_proj, 'bias', None))
            seen_modules.add('in_proj')
        
        # 1b. The final norm in the block should not be permuted as it acts on a different dimension
        # or has a different context than the inner state. Add to seen to prevent blanket pass.
        if hasattr(layer, 'norm_f'):
            seen_modules.add('norm_f')

        # 1c. Handle all conv variants (conv1d, conv1d_proj)
        for conv_name in ('conv1d', 'conv1d_proj'):
            if hasattr(layer, conv_name):
                conv_layer = getattr(layer, conv_name)
                if hasattr(conv_layer, 'weight_v') and hasattr(conv_layer, 'weight_g'):
                    logger.debug(f"Permuting special case: weight-normalized {conv_name}")
                    out_ch = conv_layer.weight_v.shape[0]
                    if out_ch == d_inner: # Value-only path
                        inplace_permute_rows(conv_layer.weight_v, p)
                        inplace_permute_vector(conv_layer.weight_g, p)
                    elif out_ch == 2 * d_inner: # Value and gate path
                        logger.debug(f"Permuting only the first half of weight-normalized {conv_name}")
                        # Permute only the value part (first d_inner), leave the gate part untouched
                        alias_free_rows_slice(conv_layer.weight_v, p, 0, d_inner)
                        # The gate part (second d_inner) is not permuted.
                        
                        # We assume weight_g corresponds to the value part only in this case.
                        # This might need adjustment if a model uses a different convention.
                        if conv_layer.weight_g.numel() == d_inner:
                             inplace_permute_vector(conv_layer.weight_g, p)
                        elif conv_layer.weight_g.numel() == 2* d_inner:
                             #Also permute only the first half of the g vector
                             inplace_permute_vector(conv_layer.weight_g[:d_inner], p)
                    
                    if hasattr(conv_layer, 'bias') and conv_layer.bias is not None:
                        # Bias handling for weight-normed conv needs similar logic
                        if conv_layer.bias.numel() == d_inner:
                            inplace_permute_vector(conv_layer.bias, p)
                        elif conv_layer.bias.numel() == 2 * d_inner:
                            logger.debug("Permuting only the first half of double-width bias for weight-normed conv")
                            bias_view = conv_layer.bias.view(2, d_inner).contiguous()
                            inplace_permute_vector(bias_view[0], p)

                else: # Plain convolution (can be 2D or 3D)
                    # Handle double-width conv as a special case first due to its unique bias logic
                    if conv_layer.weight.shape[0] == 2 * d_inner:
                         logger.debug(f"Permuting ONLY VALUE part of double-width {conv_name}")
                         # --- VALUE PART PERMUTED ---
                         alias_free_rows_slice(conv_layer.weight, p, 0, d_inner)
                         # --- GATE PART UNTOUCHED ---
                         
                         if hasattr(conv_layer, 'bias') and conv_layer.bias is not None:
                            if conv_layer.bias.numel() == 2 * d_inner:
                                logger.debug("Permuting ONLY VALUE part of double-width conv bias")
                                bias_view = conv_layer.bias.view(2, d_inner).contiguous()
                                inplace_permute_vector(bias_view[0], p)
                            elif conv_layer.bias.numel() == d_inner:
                                inplace_permute_vector(conv_layer.bias, p)
                    else:
                        # Handle all other generic conv layers via the blanket function
                        _permute_params_in_module(conv_layer, p, p_inv)
                seen_modules.add(conv_name)

        # --- Step 2: Handle all other direct child parameters automatically ---
        for name, child_module in layer.named_children():
            # Skip special modules that were already handled
            if name in seen_modules:
                continue
            _permute_params_in_module(child_module, p, p_inv)
            
        # --- Step 3: Handle parameters that are direct attributes of the layer ---
        _permute_params_in_module(layer, p, p_inv)


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
    # Infer device from model if not provided, ensuring consistency.
    device = device or next(model.parameters()).device

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
    correlation_matrix_tuple = _get_activation_correlation(
        model, dataloader, target_layer_names, is_mamba_in_proj=True, device=device,
        max_samples=iasp_config.get("max_samples", DEFAULT_MAX_SAMPLES),
        sample_stride=iasp_config.get("sample_stride", 1)
    )
    if not correlation_matrix_tuple:
        logger.error("Could not compute a valid correlation matrix for Mamba. Skipping IASP.")
        # Try to infer d_inner for a valid identity permutation
        try:
            first_mixer = model.backbone.layers[0].mixer
            dim = first_mixer.out_proj.in_features
        except (AttributeError, IndexError):
            dim = getattr(model.config, 'd_inner', 2048) # Fallback
        return list(range(dim)), 0.0

    correlation_matrix, valid_indices = correlation_matrix_tuple

    # Step 2: Find the optimal permutation for the valid subset of channels
    d_inner_valid = correlation_matrix.shape[0]
    min_size, max_size = iasp_config.cluster_size_range
    min_clusters = max(MIN_CLUSTER_SIZE, d_inner_valid // max_size)
    max_clusters = max(MIN_CLUSTER_SIZE, d_inner_valid // min_size)
    
    perm_valid, modularity = _find_optimal_permutation(
        correlation_matrix, clusters_range=(min_clusters, max_clusters), iasp_config=iasp_config
    )

    # Step 3: Map the permutation of valid channels back to the full dimension
    d_inner_full = valid_indices.numel()
    valid_indices_dev = valid_indices.to(device)

    original_indices = torch.arange(d_inner_full, device=device)[valid_indices_dev]
    permuted_original_indices = original_indices[torch.tensor(perm_valid, device=device)]
    
    full_permutation = torch.arange(d_inner_full, device=device)
    
    # Get the indices of zero-variance channels that were dropped
    dropped_indices = torch.where(~valid_indices_dev)[0]
    
    # Apply permutation to valid channels and keep dropped channels in their original places
    full_permutation[valid_indices_dev] = permuted_original_indices
    # Ensure dropped channels map to themselves
    if dropped_indices.numel() > 0:
        full_permutation[dropped_indices] = dropped_indices

    # Final sanity check to ensure the permutation has the correct full dimension
    assert full_permutation.numel() == d_inner_full, "Full permutation length mismatch!"

    # Step 4: Apply the full permutation to the model mixer blocks
    _apply_permutation_to_mamba(mamba_mixers_to_permute, full_permutation.tolist())
    
    logger.info("✅ IASP optimization for Mamba completed successfully.")
    return full_permutation.tolist(), modularity


def _apply_permutation_to_bert_ffn(model: nn.Module, permutation: List[int], target_layer_names: List[str]):
    """Applies a permutation to the specified FFN layers of a BERT-like model."""
    if not target_layer_names:
        return
        
    # Create permutation tensors once on the correct device.
    dev = model.get_submodule(target_layer_names[0]).weight.device
    p = torch.tensor(permutation, dtype=torch.long, device=dev)
    p_inv = torch.empty_like(p)
    p_inv[p] = torch.arange(p.numel(), device=dev)
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
                    logger.warning(f"Skipping FFN layer {layer_name} due to dimension mismatch.")
                    continue

                pbar.set_postfix({"layer": layer_name})

                # Permute the up-projection (output is permuted)
                inplace_permute_rows(up_proj.weight, p)
                if up_proj.bias is not None:
                    inplace_permute_vector(up_proj.bias, p)

                # Permute the down-projection (input is permuted)
                inplace_permute_cols(down_proj.weight, p_inv)
                # Down-projection bias is not permuted as it's added after the matmul
                
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
    # Infer device from model if not provided, ensuring consistency.
    device = device or next(model.parameters()).device

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
    correlation_matrix_tuple = _get_activation_correlation(
        model, dataloader, target_layer_names, is_mamba_in_proj=False, device=device,
        max_samples=iasp_config.get("max_samples", DEFAULT_MAX_SAMPLES),
        sample_stride=iasp_config.get("sample_stride", 1)
    )
    if not correlation_matrix_tuple:
        logger.error("Could not compute a valid correlation matrix for BERT FFN. Skipping IASP.")
        dim = model.config.intermediate_size
        return list(range(dim)), 0.0

    correlation_matrix, valid_indices = correlation_matrix_tuple

    # Step 2: Find the optimal permutation for the valid subset of channels
    d_ffn_valid = correlation_matrix.shape[0]
    min_size, max_size = iasp_config.cluster_size_range
    min_clusters = max(MIN_CLUSTER_SIZE, d_ffn_valid // max_size)
    max_clusters = max(MIN_CLUSTER_SIZE, d_ffn_valid // min_size)
    
    perm_valid, modularity = _find_optimal_permutation(
        correlation_matrix, clusters_range=(min_clusters, max_clusters), iasp_config=iasp_config
    )

    # Step 3: Map the permutation of valid channels back to the full dimension
    d_ffn_full = valid_indices.numel()
    valid_indices_dev = valid_indices.to(device)

    original_indices = torch.arange(d_ffn_full, device=device)[valid_indices_dev]
    permuted_original_indices = original_indices[torch.tensor(perm_valid, device=device)]
    
    full_permutation = torch.arange(d_ffn_full, device=device)
    
    # Handle dropped indices for BERT as well
    dropped_indices = torch.where(~valid_indices_dev)[0]
    full_permutation[valid_indices_dev] = permuted_original_indices
    if dropped_indices.numel() > 0:
        full_permutation[dropped_indices] = dropped_indices
    
    # Final sanity check for BERT as well
    assert full_permutation.numel() == d_ffn_full, "Full permutation length mismatch for BERT!"

    # Step 4: Apply the permutation to the model
    _apply_permutation_to_bert_ffn(model, full_permutation.tolist(), target_layer_names)

    logger.info("✅ IASP optimization for BERT completed successfully.")
    return full_permutation.tolist(), modularity