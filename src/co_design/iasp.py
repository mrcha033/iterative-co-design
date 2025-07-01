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
import os
import hashlib
from types import SimpleNamespace
from typing import Iterable, List, Tuple, Union

# Set OMP_NUM_THREADS to 1 to avoid MKL contention in joblib's threading backend.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
import joblib
from joblib import Parallel, delayed
from numpy.linalg import LinAlgError  # needed for SpectralClustering errors
from omegaconf import DictConfig
from sklearn.cluster import SpectralClustering
from sklearn.utils.extmath import randomized_svd
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# IASP 2.0 specific imports
try:
    import faiss
    import community as community_louvain
    import igraph as ig
    FAISS_LOUVAIN_AVAILABLE = True
except ImportError:
    FAISS_LOUVAIN_AVAILABLE = False
    faiss = community_louvain = ig = None # for type checkers

from .modularity import calculate_modularity
from src.utils.permutation import (
    alias_free_rows_slice,
    alias_free_vector_slice,
    inplace_permute_cols,
    inplace_permute_in_proj_split,
    inplace_permute_rows,
    inplace_permute_vector,
    safe_permute_rows,
    safe_permute_cols,
    safe_permute_vector,
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
    max_samples=2_048,
    sample_stride=2,
    knn_k=128,
    cluster_size_range=(32, 128),  # (min, max) cluster size
    permute_gate=False,
    spectral_n_init=10,
    spectral_random_state=42,
    min_cluster_size=2,
    cluster_algo="auto",
    max_dim_for_spectral=4096,
)

# ---------------------------------------------------------------------------
# IASP 2.0 Helpers
# ---------------------------------------------------------------------------

def _get_param_sha1(param: torch.Tensor) -> str:
    """
    Computes SHA1 hash of a tensor's data to verify integrity.

    Uses .detach() to avoid gradient tracking, .cpu() to move to host,
    and .numpy() to convert to raw bytes.
    """
    if not isinstance(param, torch.Tensor):
        param = torch.as_tensor(param)
    return hashlib.sha1(param.detach().cpu().numpy().tobytes()).hexdigest()


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

    if mask.sum() < 2:
        logger.warning(f"Layer has < 2 active channels for correlation. Skipping.")
        return torch.empty((0, 0), dtype=torch.float32, device=X.device), mask
    
    X_masked = X[:, mask]

    if X_masked.size(0) < X_masked.size(1):
        logger.warning(
            f"Number of samples ({X_masked.size(0)}) is less than number of features "
            f"({X_masked.size(1)}). Correlation is ill-conditioned. Skipping."
        )
        return torch.empty((0, 0), dtype=torch.float32, device=X.device), mask

    # Use torch.corrcoef for GPU-accelerated computation
    corr = torch.corrcoef(X_masked.T)
    # Ensure diagonal is exactly 1.0 after potential floating point inaccuracies
    corr.fill_diagonal_(1.0)

    # Keep on device as float32 for numerical stability in clustering
    return corr, mask


def _louvain_perm_numpy(corr_np: np.ndarray, cfg: DictConfig) -> Tuple[List[int], float]:
    """
    Computes permutation using Faiss for k-NN graph and Louvain for clustering.
    This is significantly faster and more memory-efficient for large dimensions.
    """
    if not FAISS_LOUVAIN_AVAILABLE:
        raise ImportError("Faiss and python-louvain are required for 'louvain' clustering.")

    dim = corr_np.shape[0]
    
    # 1. Embed correlation matrix into low-dim vectors using Randomized SVD
    # This avoids the O(D^2) memory of Cholesky on the full correlation matrix.
    n_components = min(cfg.get("svd_components", 256), dim // 4 or 1, dim)
    U, S, _ = randomized_svd(
        corr_np, n_components=n_components, random_state=cfg.get("spectral_random_state", 42)
    )
    # Use explicit broadcasting and L2 normalization for robust vectors
    vectors = (U * np.sqrt(S)[None, :]).astype('float32')
    vectors /= (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)
        
    # 2. Build a memory-efficient k-NN graph index (HNSW)
    k = min(cfg.get("knn_k", DEFAULTS.knn_k), dim - 1)
    index = faiss.IndexHNSWFlat(n_components, 32)
    index.add(vectors)
    _, I = index.search(vectors, k + 1)

    # 3. Convert to igraph for Louvain, pruning low-weight edges
    A = (corr_np + 1) / 2
    edge_thresh = cfg.get("edge_thresh", 1e-3)
    edges = [
        (i, j, float(A[i,j])) for i in range(dim)
        for j in I[i,1:] if i < j and A[i,j] > edge_thresh
    ]
    g = ig.Graph.TupleList(edges, weights=True, directed=False)
    
    # 4. Run Louvain community detection using igraph's native implementation
    # This is more efficient and avoids TypeError from community_louvain expecting a networkx graph.
    labels = np.array(g.community_multilevel(weights='weight').membership)
    
    # 5. Create permutation from community labels and calculate modularity
    perm = np.argsort(labels)
    parts = [np.where(labels == j)[0] for j in range(labels.max() + 1)]
    q = calculate_modularity(corr_np, parts)

    return perm.tolist(), q


def _spectral_perm_numpy(corr_np: np.ndarray, cfg: DictConfig) -> Tuple[List[int], float]:
    """Return permutation list & modularity score. (NumPy/CPU version)"""
    dim = corr_np.shape[0]
    best_q, best_perm = -np.inf, list(range(dim))

    # Guardrail: Skip spectral clustering for dimensions that are too large,
    # as it's O(D^3) and will likely cause OOM or timeout.
    # NOTE: This check is now handled by the dispatcher `_cluster_and_permute`.
    
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
                n_init=cfg.get("spectral_n_init", DEFAULTS.spectral_n_init),
                random_state=cfg.get("spectral_random_state", DEFAULTS.spectral_random_state),
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


def _cluster_and_permute(corr_np: np.ndarray, cfg: DictConfig) -> Tuple[List[int], float]:
    """
    Dispatcher: automatically selects the best clustering algorithm.
    """
    dim = corr_np.shape[0]
    cluster_algo = cfg.get("cluster_algo", DEFAULTS.cluster_algo)
    
    # Auto-select algorithm based on dimension
    if cluster_algo == "auto":
        max_dim_spectral = cfg.get("max_dim_for_spectral", DEFAULTS.max_dim_for_spectral)
        if dim > max_dim_spectral:
            if not FAISS_LOUVAIN_AVAILABLE:
                logger.warning(
                    f"dim={dim} > {max_dim_spectral}, but Faiss/Louvain not installed. "
                    "Skipping permutation. Install with: `pip install faiss-cpu python-louvain`"
                )
                return [], -1.0
            logger.info(f"Dimension ({dim}) is large, using fast Faiss+Louvain clustering.")
            cluster_algo = "louvain"
        else:
            cluster_algo = "spectral"

    logger.info(f"Using '{cluster_algo}' clustering for dim={dim}.")
    if cluster_algo == "spectral":
        return _spectral_perm_numpy(corr_np, cfg)
    elif cluster_algo == "louvain":
        return _louvain_perm_numpy(corr_np, cfg)
    else:
        raise ValueError(f"Unknown clustering algorithm: '{cluster_algo}'")


def _spectral_perm(*args, **kwargs):
    raise NotImplementedError("This function is deprecated. Call _spectral_perm_numpy")

# ---------------------------------------------------------------------------
# permutation application
# ---------------------------------------------------------------------------

def _apply_perm_to_mixer(mixer: nn.Module, p_val: torch.Tensor):
    """
    Safely applies value-path permutations to a Mamba-style mixer block
    using out-of-place tensor reconstruction.
    """
    d_val = p_val.numel()
    p_inv_val = torch.argsort(p_val)

    # --- 1. in_proj (Gate and Value) ---
    # Reconstruct the weight and bias tensors out-of-place
    with torch.no_grad():
        gate_w, val_w_orig = mixer.in_proj.weight.data.split(d_val)
        val_w_permuted = val_w_orig.index_select(0, p_val)
        mixer.in_proj.weight.data = torch.cat([gate_w, val_w_permuted], dim=0)

        if mixer.in_proj.bias is not None:
            gate_b, val_b_orig = mixer.in_proj.bias.data.split(d_val)
            val_b_permuted = val_b_orig.index_select(0, p_val)
            mixer.in_proj.bias.data = torch.cat([gate_b, val_b_permuted], dim=0)

    # --- 2. SSM Parameters (A_log, D, dt_proj) ---
    if hasattr(mixer, "A_log") and mixer.A_log is not None and mixer.A_log.size(0) == d_val:
        safe_permute_rows(mixer.A_log, p_val)
    if hasattr(mixer, "D") and mixer.D is not None and mixer.D.numel() == d_val:
        safe_permute_vector(mixer.D, p_val)
    if hasattr(mixer, "dt_proj"):
        if getattr(mixer.dt_proj, "weight", None) is not None and mixer.dt_proj.weight.numel() == d_val:
            safe_permute_vector(mixer.dt_proj.weight, p_val)
        if getattr(mixer.dt_proj, "bias", None) is not None and mixer.dt_proj.bias.numel() == d_val:
            safe_permute_vector(mixer.dt_proj.bias, p_val)

    # --- 3. Convolution (conv1d) ---
    if hasattr(mixer, "conv1d"):
        conv = mixer.conv1d
        # Permute output channels (rows) of the convolution's weight and bias
        safe_permute_rows(conv.weight, p_val)
        if conv.weight.size(1) == d_val: # Handle Mamba variants with d_val input channels
            safe_permute_cols(conv.weight, p_inv_val)
        if conv.bias is not None:
            safe_permute_vector(conv.bias, p_val)

    # --- 4. out_proj ---
    # Permute input channels (columns) of the output projection
    safe_permute_cols(mixer.out_proj.weight, p_inv_val)
    # Bias of out_proj is not affected.


def run_iasp_on_mamba(
    model: nn.Module,
    dataloader: DataLoader,
    cfg: DictConfig,
) -> Tuple[List[int], float]:
    """
    Apply layer-wise IASP 2.0 to a Mamba model.
    """
    device = next(model.parameters()).device
    if cfg.get("permute_gate", False):
        raise NotImplementedError(
            "IASP 2.0 design explicitly forbids gate permutation for safety. "
            "Use the experimental `gate_sparsity` flag if needed."
        )

    # Get config params with IASP 2.0 defaults
    n_jobs = cfg.get("jobs", 1)
    min_modularity = cfg.get("min_modularity", 0.0)
    
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
                max_samples=cfg.get("max_samples", DEFAULTS.max_samples),
                stride=cfg.get("sample_stride", DEFAULTS.sample_stride),
            )
            # Move to CPU for parallel processing. corr_gpu is already float32.
            if corr_gpu.numel() > 0:
                corr_np = corr_gpu.cpu().numpy()
                corr_data[layer_name] = (corr_np, mask)
        except Exception as e:
            logger.error(f"Failed to collect correlation for {layer_name}: {e}")
            corr_data[layer_name] = None

    # --- 2. Parallel CPU pass: Clustering ---
    layer_names = list(corr_data.keys())
    clustering_inputs = [corr_data[name][0] for name in layer_names if corr_data[name] is not None]
    
    logger.info(f"Running clustering for {len(clustering_inputs)} matrices using {n_jobs} parallel jobs (CPU)...")
    # Use 'threading' backend for GIL-releasing libraries like NumPy/Faiss
    with joblib.parallel_backend("threading"):
        results = Parallel(n_jobs=n_jobs)(
            delayed(_cluster_and_permute)(inp, cfg) for inp in tqdm(clustering_inputs, desc="Clustering")
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
        logger.info(f"Layer {layer_name}: modularity={q:.4f}")
        if q < min_modularity:
            logger.info(f"  -> Skipping permutation, modularity is below threshold ({min_modularity}).")
            continue

        try:
            _, mask = corr_data[layer_name]
            
            # --- Gate Safety & Verification ---
            parent_name = layer_name.rsplit('.', 1)[0]
            mixer = model.get_submodule(parent_name)
            d_inner_full = mask.size(0)
            
            # Store hash of gate weights BEFORE permutation
            gate_weight = mixer.in_proj.weight[:d_inner_full]
            gate_hash_before = _get_param_sha1(gate_weight)
            
            # Project masked permutation back to the original dimension
            p_full = torch.arange(d_inner_full, device=device)
            kept_indices = torch.where(mask)[0]
            p_masked_tensor = torch.tensor(perm_masked, dtype=torch.long, device=kept_indices.device)
            permuted_kept_indices = kept_indices[p_masked_tensor]
            p_full[mask] = permuted_kept_indices

            # Bijectivity verification - reverted to safer unique() check
            if p_full.unique().numel() != p_full.numel():
                raise ValueError(f"Non-bijective permutation generated for layer {layer_name}")

            # Apply permutation using the safe, out-of-place function
            _apply_perm_to_mixer(mixer, p_full)

            # --- Verification Step ---
            gate_hash_after = _get_param_sha1(mixer.in_proj.weight[:d_inner_full])
            if gate_hash_before != gate_hash_after:
                raise RuntimeError(
                    f"FATAL: Gate weights were modified for layer {layer_name} "
                    f"despite safety checks! Hash before: {gate_hash_before}, "
                    f"hash after: {gate_hash_after}"
                )

            logger.info(f"  -> Applied permutation. Gate preserved (SHA1: {gate_hash_after[:8]}).")
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

        # Permute columns of the first dense layer (up-projection) using safe methods
        safe_permute_cols(up_proj.weight, p_inv)
        safe_permute_vector(up_proj.bias, p_inv)
        
        # Permute rows of the second dense layer (down-projection)
        safe_permute_rows(down_proj.weight, p)
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
    min_modularity = cfg.get("min_modularity", 0.0)
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
                max_samples=cfg.get("max_samples", DEFAULTS.max_samples),
                stride=cfg.get("sample_stride", DEFAULTS.sample_stride),
            )
            
            # The mask should be all True for BERT FFNs, but we check just in case.
            if not mask.all():
                logger.warning(f"Layer {layer_name} has dead neurons. Skipping permutation.")
                continue

            # 2. Run clustering
            corr_np = corr_gpu.cpu().numpy()
            perm, q = _cluster_and_permute(corr_np, cfg)

            logger.info(f"Layer {layer_name}: modularity={q:.4f}")

            # 3. Apply permutation if it's good enough
            if perm and q > min_modularity:
                _apply_perm_to_bert_ffn(model, perm, [layer_name])
                logger.info(f"  -> Applied permutation with modularity {q:.4f}")
                total_q += q
                applied_count += 1
                last_perm = perm
        except Exception as e:
            logger.error(f"Failed to process layer {layer_name}: {e}", exc_info=True)

    avg_q = (total_q / applied_count) if applied_count > 0 else 0.0
    logger.info(f"IASP for BERT finished. Applied {applied_count} permutations with average modularity: {avg_q:.4f}")

    if not last_perm:
        try:
            # Read d_ffn from the model's actual layer shape for robustness
            first_layer = model.get_submodule(target_layers[0])
            d_ffn = first_layer.weight.size(0)
        except (IndexError, AttributeError):
            # Fallback for safety
            d_ffn = model.config.intermediate_size
        return list(range(d_ffn)), avg_q
        
    return last_perm, avg_q
