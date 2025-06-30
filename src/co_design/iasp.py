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
from typing import Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
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

def _expand_wildcard(model: nn.Module, spec: str | Iterable[str]) -> List[str]:
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
) -> Tuple[np.ndarray, torch.Tensor]:
    """Return fp32 correlation matrix & valid‑channel mask for **one** layer."""
    model.eval()
    device = next(model.parameters()).device

    acts: list[torch.Tensor] = []

    def _hook(_, __, out):
        d_inner = out.size(-1) // 2
        # Use .half() for faster collection, then upcast for precision later
        x = out[..., :d_inner].half()
        if x.ndim == 3:
            # Flatten (batch, seq, dim) to (batch*seq, dim)
            x = x.reshape(-1, d_inner)
        acts.append(x)  # Keep on device

    h = model.get_submodule(layer).register_forward_hook(_hook)

    collected = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if collected >= max_samples:
                break
            if i % stride:
                continue
            inp = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            model(**inp)
            # Use .shape[0] on the flattened tensor for accurate sample counting
            collected += acts[-1].shape[0]

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
    
    # Return numpy array for compatibility with scikit-learn
    return corr.cpu().numpy(), mask


def _spectral_perm(corr: np.ndarray, cfg: DictConfig) -> Tuple[List[int], float]:
    """Return permutation list & modularity score."""
    dim = corr.shape[0]
    best_q, best_perm = -np.inf, list(range(dim))

    # build symmetric k‑NN affinity
    A = (corr + 1) / 2
    k = max(1, min(cfg.knn_k, dim - 1))
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
        q = calculate_modularity(corr, parts)
        if q > best_q:
            best_q = q
            best_perm = [idx for part in parts for idx in part]
    return best_perm, best_q

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

    # x_proj
    if hasattr(mixer, "x_proj"):
        w = mixer.x_proj.weight
        if w.size(0) == 2 * d_val:
            alias_free_rows_slice(w, p_val, 0, d_val)
        elif w.size(0) == d_val:
            inplace_permute_rows(w, p_val)
        if w.size(1) == d_full:
            inplace_permute_cols(w, p_inv_full)
        elif w.size(1) == d_val:
            inplace_permute_cols(w, p_inv_val)
        if mixer.x_proj.bias is not None and mixer.x_proj.bias.numel() == 2 * d_val:
            inplace_permute_vector(mixer.x_proj.bias.view(2, d_val)[0], p_val)

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
    """Run IASP and return (last layer permutation, modularity)."""

    cfg = DictConfig({**DEFAULTS.__dict__, **cfg})
    device = next(model.parameters()).device

    layers = _expand_wildcard(model, cfg.get("target_layer_names")) or [
        n for n, m in model.named_modules() if n.endswith("in_proj") and isinstance(m, nn.Linear)
    ]
    if not layers:
        raise RuntimeError("no in_proj layers found")

    logger.info("IASP: %d target layers", len(layers))
    last_perm, last_q = None, 0.0

    for lyr in tqdm(layers, desc="IASP", ncols=95):
        corr, mask = _collect_layer_corr(
            model,
            dataloader,
            lyr,
            max_samples=cfg.max_samples,
            stride=cfg.sample_stride,
        )
        sub_perm, q = _spectral_perm(corr, cfg)

        # --- build value-only permutation ---
        d_val = mask.numel()
        p_val = torch.arange(d_val, device=device)
        # The permutation from spectral clustering is for the *masked* subspace.
        # We assign the permuted indices of the valid channels back into the
        # full-sized permutation tensor.
        valid_indices = torch.where(mask)[0]
        p_val[valid_indices] = valid_indices[torch.tensor(sub_perm, device=device)]

        # --- decide what to do with gate channels ---
        if cfg.get("permute_gate", False):
            # mirror the same order for gates
            p_gate = p_val.clone()
        else:
            # identity – keep original gate order
            p_gate = torch.arange(d_val, device=device)

        # [gate, value]
        p_full = torch.cat([p_gate, p_val])

        mixer_parent = ".".join(lyr.split(".")[:-1])
        mixer = model.get_submodule(mixer_parent)
        _apply_perm_to_mixer(mixer, p_full)

        last_perm, last_q = p_full.tolist(), float(q)

    return last_perm, last_q


def _apply_perm_to_bert_ffn(
    model: nn.Module,
    perm: list[int],
    layer_names: list[str]
):
    """Apply permutation to all FFN layers in a BERT-style model."""
    dev = next(model.parameters()).device
    p = torch.tensor(perm, dtype=torch.long, device=dev)
    p_inv = torch.argsort(p)
    d_ffn = len(perm)

    for name in layer_names:
        parent_name = ".".join(name.split(".")[:-2])
        parent_mod = model.get_submodule(parent_name)

        up_proj = parent_mod.intermediate.dense
        down_proj = parent_mod.output.dense

        if up_proj.out_features != d_ffn or down_proj.in_features != d_ffn:
            logger.warning("Skipping FFN layer %s due to shape mismatch", name)
            continue
        
        # up-projection: permute output rows and bias
        inplace_permute_rows(up_proj.weight, p)
        if up_proj.bias is not None:
            inplace_permute_vector(up_proj.bias, p)

        # down-projection: permute input columns
        inplace_permute_cols(down_proj.weight, p_inv)


def run_iasp_on_bert(
    model: nn.Module,
    dataloader: DataLoader,
    cfg: DictConfig,
) -> Tuple[List[int], float]:
    """Run IASP on BERT, returning final permutation and modularity."""
    cfg = DictConfig({**DEFAULTS.__dict__, **cfg})
    device = next(model.parameters()).device

    # 1. Find all target FFN layers
    layers = _expand_wildcard(model, cfg.get("target_layer_names")) or [
        n for n, m in model.named_modules() if n.endswith("intermediate.dense")
    ]
    if not layers:
        raise RuntimeError("No BERT FFN layers found (*.intermediate.dense)")

    # 2. Collect activations from ALL target layers to build ONE correlation matrix
    all_acts = []
    handles = []

    def _hook(_, __, out):
        # out is (B, T, C)
        all_acts.append(out.reshape(-1, out.size(-1)).detach().to(torch.float32))

    for name in layers:
        handles.append(model.get_submodule(name).register_forward_hook(_hook))
    
    collected = 0
    pbar = tqdm(dataloader, desc="IASP: BERT activations", ncols=95)
    for i, batch in pbar:
        if collected >= cfg.max_samples:
            break
        if i % cfg.sample_stride:
            continue
        inp = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        with torch.no_grad():
            model(**inp)
        collected += inp["input_ids"].numel()
        pbar.set_postfix({"tokens": f"{collected}/{cfg.max_samples}"})

    for h in handles:
        h.remove()
    
    if not all_acts:
        raise RuntimeError("No activations captured for BERT FFN layers")

    # 3. Compute global correlation
    X = torch.cat(all_acts, 0)[:cfg.max_samples]
    std = X.std(0)
    mask = std > 1e-6
    X = X[:, mask]
    corr = torch.corrcoef(X.T).cpu().numpy()
    np.fill_diagonal(corr, 1.0)
    
    # 4. Find optimal permutation on the valid subspace
    sub_perm, q = _spectral_perm(corr, cfg)

    # 5. Map permutation back to full dimension
    d_ffn = mask.size(0)
    p_full = torch.arange(d_ffn, device=device)
    valid_indices = torch.where(mask)[0]
    p_full[valid_indices] = valid_indices[torch.tensor(sub_perm, device=device)]

    # 6. Apply permutation to all layers
    _apply_perm_to_bert_ffn(model, p_full.tolist(), layers)
    
    logger.info("IASP for BERT done – modularity %.4f", q)
    return p_full.tolist(), q
