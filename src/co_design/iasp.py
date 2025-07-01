"""
IO‑Aware Scan Permutation (IASP)
================================
A robust, layer‑wise permutation pipeline that preserves perplexity while
improving memory locality on Mamba‑style architectures.

Major guarantees
----------------
* **Layer isolation** – every `*.in_proj` gets its own fp32 correlation matrix.
* **Value‑only permutation** – gate path stays identity by default (configurable).
* **Full parameter coverage** – `dt_proj.weight`, conv, SSM params, etc.
* **Device‑safe** – works whether model is on CPU or (multi)‑GPU.

Usage example
-------------
```python
from omegaconf import OmegaConf
perm, q = run_iasp_on_mamba(model, dl, OmegaConf.create({
    "max_samples": 8192,
    "sample_stride": 2,
    "knn_k": 128,
    "cluster_size_range": [32, 128],
}))
```
"""

from __future__ import annotations

import fnmatch
import logging
from types import SimpleNamespace
from typing import Iterable, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import joblib
from joblib import Parallel, delayed
from numpy.linalg import LinAlgError  # needed for SpectralClustering errors
from omegaconf import DictConfig
from sklearn.cluster import SpectralClustering
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .modularity import calculate_modularity
from utils.permutation import (
    alias_free_rows_slice,
    alias_free_vector_slice,
    inplace_permute_cols,
    inplace_permute_in_proj_split,
    inplace_permute_rows,
    inplace_permute_vector,
)

# ---------------------------------------------------------------------------
# Version Guard
# ---------------------------------------------------------------------------
if torch.__version__ < "2.0":
    raise RuntimeError("IASP's use of `torch.corrcoef` on GPU requires PyTorch >= 2.0.")

# ---------------------------------------------------------------------------
# logging / defaults
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

DEFAULTS = SimpleNamespace(
    max_samples=1_024,
    sample_stride=2,
    knn_k=128,
    cluster_size_range=(32, 128),  # (min, max) cluster size
    permute_gate=False,
        spectral_n_init=10,
        spectral_random_state=42,
    min_cluster_size=2,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _expand_wildcard(model: nn.Module, spec: Union[str, Iterable[str]]) -> List[str]:
    """Expand wildcard patterns (e.g. `backbone.*.in_proj`)."""
    if not spec:
        return []

    names = [n for n, _ in model.named_modules()]
    patterns = spec if isinstance(spec, (list, tuple)) else [spec]
    out: List[str] = []
    for pat in patterns:
        if "*" in pat:
            m = fnmatch.filter(names, pat)
            if not m:
                logger.warning("pattern %s matched nothing", pat)
            out.extend(m)
        else:
            out.append(pat)
    return out


def _collect_layer_corr(
    model: nn.Module,
    dataloader: DataLoader,
    layer: str,
    max_samples: int,
    stride: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return fp32 correlation **GPU tensor** & valid‑channel mask."""
    model.eval()
    device = next(model.parameters()).device

    acts: list[torch.Tensor] = []

    def _hook(_, __, out):
        d_inner = out.size(-1) // 2
        # Gate is first half, Value is second. Permute based on Value stats.
        # Keep in full precision to avoid numerical issues with std calculation.
        x = out[..., d_inner:]
        if x.ndim == 3:
            # Flatten (batch, seq, dim) to (batch*seq, dim)
            x = x.reshape(-1, d_inner)
        acts.append(x)  # Keep on device

    h = model.get_submodule(layer).register_forward_hook(_hook)
    collected = 0
    try:
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if collected >= max_samples:
                    break
                
                # Flexible batch handling for different dataset structures
                if isinstance(batch, dict):
                    inp = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
                elif isinstance(batch, (list, tuple)) and torch.is_tensor(batch[0]):
                    # Assume standard (input_ids, labels, ...) tuple format
                    inp = {"input_ids": batch[0].to(device)}
                    if len(batch) > 1 and torch.is_tensor(batch[1]):
                        # Pass labels if they exist, for model compatibility
                        inp["labels"] = batch[1].to(device)
                else:
                    raise TypeError(f"Unsupported batch type for IASP data collection: {type(batch)}")

                model(**inp)
                # Sample-level stride and counting
                samples_in_batch = acts[-1]
                # Apply stride and cap at max_samples
                samples_to_take = samples_in_batch[::stride]
                if collected + samples_to_take.size(0) > max_samples:
                    needed = max_samples - collected
                    samples_to_take = samples_to_take[:needed]
                
                # Replace last activation with the strided/capped version
                acts[-1] = samples_to_take
                collected += samples_to_take.size(0)

    finally:
        h.remove()

    if not acts:
        raise RuntimeError(f"no activations captured for {layer}")

    # Move correlation calculation entirely to GPU
    X = torch.cat(acts, 0)[:max_samples]
    # Upcast to float32 for stable correlation calculation
    X = X.to(torch.float32)

    std = X.std(0)
    mask = std > 1e-6
    X_masked = X[:, mask]

    # Use torch.corrcoef for GPU-accelerated computation
    corr = torch.corrcoef(X_masked.T)
    # Ensure diagonal is exactly 1.0 after potential floating point inaccuracies
    corr.fill_diagonal_(1.0)

    # Keep on device as float32 for numerical stability in clustering
    return corr, mask


def _spectral_perm_numpy(corr_np: np.ndarray, cfg: DictConfig) -> Tuple[List[int], float]:
    """Return permutation list & modularity score. (NumPy/CPU version)"""
    dim = corr_np.shape[0]
    best_q, best_perm = -np.inf, list(range(dim))

    # Guardrail: Skip spectral clustering for dimensions that are too large,
    # as it's O(D^3) and will likely cause OOM or timeout.
    max_dim_spectral = cfg.get("max_dim_for_spectral", 4096)
    if dim > max_dim_spectral:
        logger.warning(
            f"Skipping spectral clustering for dim={dim} (>{max_dim_spectral}). "
            f"This is too large and would be slow/memory-intensive. "
            f"Consider using a faster clustering method or reducing `d_inner`."
        )
        return best_perm, -1.0

    # build symmetric k‑NN affinity
    A = (corr_np + 1) / 2
    
    # Robust k calculation
    k_base = min(cfg.knn_k, dim // 4)
    k = max(4, k_base)
    # Heuristic to prevent OOM on large dimensions
    k = min(k, int(np.sqrt(dim)))
    k = min(k, dim - 1)  # clamp to valid range

    if k < 2:  # SpectralClustering needs at least 2 clusters, k must be >= 2 for that
        return best_perm, best_q

    G = np.zeros_like(A)
    for i in range(dim):
        idx = np.argpartition(A[i], -k)[-k:]
        G[i, idx] = A[i, idx]
    A = np.maximum(G, G.T)
    np.fill_diagonal(A, 0)

    # cluster search range
    cmin, cmax = cfg.cluster_size_range
    k_min = max(DEFAULTS.min_cluster_size, dim // cmax)
    k_max = max(DEFAULTS.min_cluster_size, dim // cmin)

    for n_cl in range(k_min, k_max + 1):
        if not (1 < n_cl < dim):
            continue
        try:
            labels = SpectralClustering(
                n_clusters=n_cl,
                affinity="precomputed",
                n_init=cfg.spectral_n_init,
                random_state=cfg.spectral_random_state,
                assign_labels="kmeans",
            ).fit_predict(A)
        except LinAlgError:
            continue
        parts = [np.where(labels == j)[0] for j in range(n_cl)]
        q = calculate_modularity(corr_np, parts)
        if q > best_q:
            best_q = q
            best_perm = [idx for part in parts for idx in part]
    return best_perm, best_q


def _spectral_perm(*args, **kwargs):
    raise NotImplementedError("This function is deprecated. Call _spectral_perm_numpy")

# ---------------------------------------------------------------------------
# permutation application
# ---------------------------------------------------------------------------

def _permute_module_generic(
    mod: nn.Module,
    p_val: torch.Tensor,
    p_inv_val: torch.Tensor,
    p_full: torch.Tensor,
    p_inv_full: torch.Tensor,
):
    d_val = p_val.numel()
    d_full = p_full.numel()
    for _, param in mod.named_parameters(recurse=False):
        if param.ndim == 1:
            if param.numel() == d_val:
                    inplace_permute_vector(param, p_val)
            elif param.numel() == d_full:
                    inplace_permute_vector(param, p_full)
        elif param.ndim == 2:
            r, c = param.shape
            if r == d_val:
                inplace_permute_rows(param, p_val)
            if c == d_val:
                inplace_permute_cols(param, p_inv_val)
            if c == d_full:
                inplace_permute_cols(param, p_inv_full)
        elif param.ndim == 3 and param.shape[0] == d_val:
            flat = param.data.view(d_val, -1).index_select(0, p_val)
            param.data.copy_(flat.view_as(param))


def _apply_perm_to_mixer(mixer: nn.Module, p_full: torch.Tensor):
    d_full = p_full.numel()
    d_val = d_full // 2
    p_gate, p_val = p_full.split(d_val)
    p_inv_val = torch.argsort(p_val)
    p_inv_full = torch.argsort(p_full)

    # --- in_proj ---
    # weight shape: [2*d_val, d_model], bias shape: [2*d_val]
    # First half of output channels is gate, second is value.
    alias_free_rows_slice(mixer.in_proj.weight, p_gate, 0, d_val)
    alias_free_rows_slice(mixer.in_proj.weight, p_val, d_val, 2 * d_val)
    if mixer.in_proj.bias is not None:
        alias_free_vector_slice(mixer.in_proj.bias, p_gate, 0, d_val)
        alias_free_vector_slice(mixer.in_proj.bias, p_val, d_val, 2 * d_val)

    # SSM / dt_proj
    if getattr(mixer, "A_log", None) is not None and mixer.A_log.size(0) == d_val:
        inplace_permute_rows(mixer.A_log, p_val)
    if getattr(mixer, "D", None) is not None and mixer.D.numel() == d_val:
        inplace_permute_vector(mixer.D, p_val)
    if hasattr(mixer, "dt_proj"):
        if getattr(mixer.dt_proj, "weight", None) is not None and mixer.dt_proj.weight.numel() == d_val:
            inplace_permute_vector(mixer.dt_proj.weight, p_val)
        if getattr(mixer.dt_proj, "bias", None) is not None and mixer.dt_proj.bias.numel() == d_val:
            inplace_permute_vector(mixer.dt_proj.bias, p_val)

    # children except specials
    specials = {"conv1d", "conv1d_proj", "x_proj", "out_proj"}
    for n, child in mixer.named_children():
        if n in specials:
            continue
        _permute_module_generic(child, p_val, p_inv_val, p_full, p_inv_full)

    # conv variations
    for cname in ("conv1d", "conv1d_proj"):
        if not hasattr(mixer, cname):
            continue
        conv = getattr(mixer, cname)
        if hasattr(conv, "weight_v") and conv.weight_v.size(0) == d_val:
            assert conv.weight_v.shape[1] in (d_val, d_full), (
                f"Unexpected shape for {cname}.weight_v: {conv.weight_v.shape}"
            )
            inplace_permute_rows(conv.weight_v, p_val)
            if conv.weight_v.size(1) == d_full:
                inplace_permute_cols(conv.weight_v, p_inv_full)
        if hasattr(conv, "weight_g") and conv.weight_g.numel() == d_val:
            inplace_permute_vector(conv.weight_g, p_val)
        if getattr(conv, "bias", None) is not None and conv.bias.numel() == d_val:
            inplace_permute_vector(conv.bias, p_val)

    # x_proj can have multiple configurations
    w = mixer.x_proj.weight
    if w.size(0) == 2 * d_val:
        # This case is ambiguous, but we assume the two outputs are gate/value
        alias_free_rows_slice(w, p_gate, 0, d_val)
        alias_free_rows_slice(w, p_val, d_val, 2 * d_val)
    elif w.size(0) == d_val:
        inplace_permute_rows(w, p_val)
    
    # Correctly handle input permutations
    if w.size(1) == d_full:
        inplace_permute_cols(w, p_inv_full)
    elif w.size(1) == d_val:
        inplace_permute_cols(w, p_inv_val)

    # Correctly handle gate/value split in bias
    if mixer.x_proj.bias is not None:
        if mixer.x_proj.bias.numel() == 2 * d_val:
            bias_as_rows = mixer.x_proj.bias.view(2, d_val)
            # Permute value bias (row 1)
            inplace_permute_vector(bias_as_rows[1], p_val)
            # Gate bias (row 0) should remain identity, so no-op.
        elif mixer.x_proj.bias.numel() == d_val:
            # If bias is only d_val, assume it's for the value path
            inplace_permute_vector(mixer.x_proj.bias, p_val)

    # out_proj columns
    if mixer.out_proj.weight.size(1) == d_full:
        inplace_permute_cols(mixer.out_proj.weight, p_inv_full)
    elif mixer.out_proj.weight.size(1) == d_val:
        inplace_permute_cols(mixer.out_proj.weight, p_inv_val)

# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def run_iasp_on_mamba(
    model: nn.Module,
    dataloader: DataLoader,
    cfg: DictConfig,
) -> Tuple[List[int], float]:
    """
    Apply layer-wise IASP to a Mamba model.

    This version uses a safe GPU-serial, CPU-parallel pipeline:
    1. Serially collect all correlation matrices on the GPU.
    2. In parallel on the CPU, run spectral clustering on each matrix.
    3. Serially apply the resulting permutations back on the GPU.
    """
    device = next(model.parameters()).device
    permute_gate = bool(cfg.get("permute_gate", False))
    if permute_gate:
        raise NotImplementedError(
            "Gate permutation is currently unsupported for Mamba models "
            "as it typically harms perplexity."
        )

    # Get config params with defaults
    n_jobs = cfg.get("jobs", 1)
    modularity_skip_threshold = cfg.get("modularity_skip_threshold", 0.0)
    
    # Add a fallback for target_layers and ensure it's not empty
    target_layers = _expand_wildcard(model, cfg.get("target_layers") or "*.in_proj")
    if not target_layers:
        raise RuntimeError("IASP: No target layers were found for the model. Please check `iasp.target_layers` in your config.")

    # --- 1. Serial GPU pass: Collect all correlation matrices ---
    logger.info(f"Collecting correlation matrices for {len(target_layers)} layers (GPU)...")
    corr_data = {}
    for layer_name in tqdm(target_layers, desc="Collecting layer correlations"):
        try:
            corr_gpu, mask = _collect_layer_corr(
                model, dataloader, layer_name,
                max_samples=cfg.max_samples, stride=cfg.sample_stride
            )
            # Move to CPU for parallel processing. corr_gpu is already float32.
            corr_np = corr_gpu.cpu().numpy()
            corr_data[layer_name] = (corr_np, mask)
        except Exception as e:
            logger.error(f"Failed to collect correlation for {layer_name}: {e}")
            corr_data[layer_name] = None

    # --- 2. Parallel CPU pass: Spectral Clustering ---
    def _run_spectral_for_layer(data):
        if data is None:
            return None, -1.0
        corr_np, _ = data
        return _spectral_perm_numpy(corr_np, cfg)

    layer_names = list(corr_data.keys())
    spectral_inputs = [corr_data[name] for name in layer_names]
    
    logger.info(f"Running spectral clustering for {len(spectral_inputs)} matrices using {n_jobs} parallel jobs (CPU)...")
    # Use 'threading' backend for GIL-releasing libraries like NumPy
    # to avoid the massive memory overhead of 'loky' (process-based).
    with joblib.parallel_backend("threading"):
        results = Parallel(n_jobs=n_jobs)(
            delayed(_run_spectral_for_layer)(inp) for inp in tqdm(spectral_inputs, desc="Spectral clustering")
        )
    
    # --- 3. Serial GPU pass: Apply permutations ---
    logger.info("Applying permutations to model layers (GPU)...")
    permutations = dict(zip(layer_names, results))
    
    all_q_scores = []
    last_perm = []

    for layer_name, (perm_masked, q) in permutations.items():
        if perm_masked is None or not perm_masked:
            continue
    
        all_q_scores.append(q)
        logger.info(f"Layer {layer_name}: modularity={q:.4f}, permutation found={bool(perm_masked and q > modularity_skip_threshold)}")
        if q < modularity_skip_threshold:
            continue

        try:
            _, mask = corr_data[layer_name]
            
            # --- Verification Setup ---
            parent_name = layer_name.rsplit('.', 1)[0]
            mixer = model.get_submodule(parent_name)
            d_inner_full = mask.size(0)
            # Make a copy of the original gate weights for verification
            orig_gate_weight = mixer.in_proj.weight[:d_inner_full].clone()
            
            # Project masked permutation back to the original dimension
            p_full = torch.arange(d_inner_full, device=device)
            kept_indices = torch.where(mask)[0]
            p_masked_tensor = torch.tensor(perm_masked, dtype=torch.long, device=kept_indices.device)
            permuted_kept_indices = kept_indices[p_masked_tensor]
            p_full[mask] = permuted_kept_indices

            # Bijectivity verification
            if p_full.unique().numel() != p_full.numel():
                raise ValueError(f"Non-bijective permutation generated for layer {layer_name}")

            # Final permutation: gate is identity, value is permuted
            # p_full must be 0-based as _apply_perm_to_mixer handles slicing offsets.
            p_final = torch.cat([
                torch.arange(d_inner_full, device=device), # gate (identity)
                p_full                                     # value (permuted, 0-based)
            ])
            
            # Apply permutation
            _apply_perm_to_mixer(mixer, p_final)

            # --- Verification Step ---
            assert torch.allclose(
                mixer.in_proj.weight[:d_inner_full], orig_gate_weight, atol=1e-6
            ), f"Gate weights mutated for layer {layer_name}!"

            logger.info(f"Applied permutation to {layer_name} with modularity {q:.4f}")
            last_perm = p_full.cpu().tolist() # Save the most recent valid perm
        
        except Exception as e:
            logger.error(f"Failed to apply permutation to {layer_name}: {e}", exc_info=True)

    avg_q = np.mean(all_q_scores) if all_q_scores else 0.0
    logger.info(f"IASP finished. Applied {len(all_q_scores)} permutations with average modularity: {avg_q:.4f}")

    if not last_perm:
        d_model = model.config.d_model if hasattr(model.config, "d_model") else 2048
        return list(range(d_model)), avg_q

    return last_perm, avg_q


def _apply_perm_to_bert_ffn(
    model: nn.Module,
    perm: list[int],
    layer_names: list[str]
):
    """
    Apply same permutation to all FFN layers in a BERT model.
    Permutes columns of the up-projection and rows of the down-projection.
    """
    p = torch.tensor(perm, device=next(model.parameters()).device)
    p_inv = torch.argsort(p)
    
    for intermediate_layer_name in layer_names:
        # e.g., "bert.encoder.layer.0.intermediate.dense"
        up_proj = model.get_submodule(intermediate_layer_name)
        
        # The corresponding output layer
        # e.g., "bert.encoder.layer.0.output.dense"
        output_layer_name = intermediate_layer_name.replace("intermediate.dense", "output.dense")
        down_proj = model.get_submodule(output_layer_name)

        # Permute columns of the first dense layer (up-projection)
        inplace_permute_cols(up_proj.weight, p_inv)
        inplace_permute_vector(up_proj.bias, p_inv)
        
        # Permute rows of the second dense layer (down-projection)
        inplace_permute_rows(down_proj.weight, p)
        # The bias of the down-projection is not permuted.


def run_iasp_on_bert(
    model: nn.Module,
    dataloader: DataLoader,
    cfg: DictConfig,
) -> Tuple[List[int], float]:
    """
    Run layer-wise IASP on a BERT model's FFNs.

    This now uses a robust, per-layer permutation strategy similar to the Mamba
    implementation. A permutation is calculated for each FFN's intermediate
    output, and only applied if it improves modularity beyond a threshold.
    """
    target_layers = _expand_wildcard(model, cfg.get("target_layers") or "*.intermediate.dense")
    if not target_layers:
        raise ValueError("IASP on BERT requires 'target_layers' to be specified, e.g., '*.intermediate.dense'.")
    
    n_jobs = cfg.get("jobs", 1)
    modularity_skip_threshold = cfg.get("modularity_skip_threshold", 0.0)
    logger.info(f"Starting IASP for {len(target_layers)} BERT FFN layers.")

    total_q = 0.0
    applied_count = 0
    last_perm = []

    for layer_name in tqdm(target_layers, desc="Processing BERT Layers"):
        try:
            logger.info(f"Analyzing layer: {layer_name}")
            # 1. Collect correlation matrix
            corr_gpu, mask = _collect_layer_corr(
                model,
                dataloader,
                layer_name,
                max_samples=cfg.max_samples,
                stride=cfg.sample_stride,
            )
            
            # The mask should be all True for BERT FFNs, but we check just in case.
            if not mask.all():
                logger.warning(f"Layer {layer_name} has dead neurons. Skipping permutation.")
                continue

            # 2. Run spectral clustering
            corr_np = corr_gpu.cpu().numpy()
            perm, q = _spectral_perm_numpy(corr_np, cfg)

            logger.info(f"Layer {layer_name}: modularity={q:.4f}, permutation found={bool(perm and q > modularity_skip_threshold)}")

            # 3. Apply permutation if it's good enough
            if perm and q > modularity_skip_threshold:
                _apply_perm_to_bert_ffn(model, perm, [layer_name])
                logger.info(f"Applied permutation to {layer_name} with modularity {q:.4f}")
                total_q += q
                applied_count += 1
                last_perm = perm
        except Exception as e:
            logger.error(f"Failed to process layer {layer_name}: {e}", exc_info=True)

    avg_q = (total_q / applied_count) if applied_count > 0 else 0.0
    logger.info(f"IASP for BERT finished. Applied {applied_count} permutations with average modularity: {avg_q:.4f}")

    if not last_perm:
        d_ffn = model.config.intermediate_size
        return list(range(d_ffn)), avg_q
        
    return last_perm, avg_q
