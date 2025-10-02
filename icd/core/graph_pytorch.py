from __future__ import annotations

"""
Minimal PyTorch trace → W builder (v0).

Notes
- Requires torch; this module is optional and not imported by default.
- Heuristic: accumulate co-access weights along the last feature dimension for
  supported ops (Linear/Matmul/Add/Reshape/Permute) using byte-based weights.
- Returns CSRMatrix compatible with icd.core.graph.CSRMatrix.
"""

from collections import Counter
from typing import Any, Dict, List, Tuple

try:
    import torch  # type: ignore
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    try:
        import torch.nn as nn  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        nn = None  # type: ignore
else:  # pragma: no cover - optional dependency
    nn = None  # type: ignore

from .graph import CSRMatrix


def _is_valid_dim(val: Any) -> bool:
    try:
        return isinstance(val, int) and val > 0
    except Exception:
        return False


def _maybe_override_feature_dim_from_config(model: Any, current_dim: int) -> tuple[int, str | None]:
    """Use configuration metadata to refine the inferred feature dimension.

    HuggingFace Mamba checkpoints expose both ``config.d_model`` (state width)
    and ``hidden_size`` (projection width). The permutation operates on the
    projection width, so prefer ``hidden_size`` when present and fall back to
    ``d_model`` only when ``hidden_size`` is missing.
    """

    config = getattr(model, "config", None)
    if config is None:
        return current_dim, None

    def _cfg(attr: str) -> int | None:
        try:
            val = getattr(config, attr, None)
        except Exception:
            return None
        return int(val) if _is_valid_dim(val) else None

    model_type = getattr(config, "model_type", None)
    hidden_size = _cfg("hidden_size")
    d_model = _cfg("d_model") or _cfg("model_dim")

    if model_type == "mamba":
        # HuggingFace Mamba checkpoints expose ``d_model`` (state width) alongside
        # ``hidden_size`` (projection width). The permutation operates over the
        # projection dimension, which maps to ``hidden_size`` in the mixer
        # modules. Prefer ``hidden_size`` when it is available so the graph
        # representation matches the tensors we later permute. Fall back to
        # ``d_model`` for older checkpoints where ``hidden_size`` might be
        # missing.
        if _is_valid_dim(hidden_size):
            return int(hidden_size), "hf_config.hidden_size"
        if _is_valid_dim(d_model):
            return int(d_model), "hf_config.d_model"
        return current_dim, None
    if _is_valid_dim(hidden_size):
        return int(hidden_size), "hf_config.hidden_size"
    if _is_valid_dim(d_model):
        return int(d_model), "hf_config.d_model"

    return current_dim, None


def _infer_seq_len_from_tensor(t: Any) -> int:
    try:
        if hasattr(t, "dim") and hasattr(t, "shape") and t.dim() >= 2:
            return int(t.shape[-2])
    except Exception:
        pass
    return 0


def _infer_feature_dim_from_fx(gm: Any, fallback: int = 0) -> int:
    """Infer feature dimension from FX trace metadata.

    Falls back to the provided ``fallback`` when inference fails. Prioritises
    nn.Linear ``out_features`` metadata, then matmul-style ops, and finally
    generic last-dimension shape votes to avoid being dominated by Conv1d
    traces that expose the sequence length as the final dimension.
    """

    if gm is None:
        return fallback

    dims_last: List[int] = []
    dims_linear: List[int] = []
    dims_matmul: List[int] = []

    def _accept_dim(val: int) -> bool:
        # Clamp to a plausible hidden-width window so LM heads (|V|) or long
        # sequence axes do not dominate the vote tally.
        return 8 <= val <= 32768
    graph = getattr(gm, "graph", None)
    nodes = getattr(graph, "nodes", []) if graph is not None else []
    get_submodule = getattr(gm, "get_submodule", None)
    for node in nodes:
        meta = getattr(node, "meta", {})
        tensor_meta = meta.get("tensor_meta") if isinstance(meta, dict) else getattr(meta, "tensor_meta", None)
        shape = getattr(tensor_meta, "shape", None)
        # 1) Prefer explicit nn.Linear out_features metadata when available
        if (
            getattr(node, "op", "") == "call_module"
            and callable(get_submodule)
            and nn is not None
        ):
            try:
                sub = get_submodule(node.target)
                if isinstance(sub, nn.Linear):
                    of = int(getattr(sub, "out_features", 0) or 0)
                    if of > 0 and _accept_dim(of):
                        dims_linear.append(of)
                    continue
            except Exception:
                pass

        # 2) Trust matrix multiplications that preserve feature dimension ordering
        if getattr(node, "op", "") == "call_function":
            fn = node.target
            fn_name = getattr(fn, "__name__", str(fn))
            if fn_name in {"addmm", "mm", "matmul", "bmm"}:
                if shape is not None and hasattr(shape, "__len__") and len(shape) >= 2:
                    try:
                        last = int(shape[-1])
                        if last > 0 and _accept_dim(last):
                            dims_matmul.append(last)
                        continue
                    except Exception:
                        pass

        # 3) Fall back to last-dimension shape votes (may be polluted by Conv1d)
        if shape is not None and hasattr(shape, "__len__") and len(shape) >= 2:
            try:
                last = int(shape[-1])
                if last > 0 and _accept_dim(last):
                    dims_last.append(last)
            except Exception:
                pass

    def _score(val: int) -> int:
        if isinstance(fallback, int) and fallback > 0:
            return -abs(val - fallback)
        return -val

    for votes in (dims_linear, dims_matmul, dims_last):
        if votes:
            counts = Counter(votes)
            inferred, _ = max(counts.items(), key=lambda kv: (kv[1], _score(kv[0])))
            if inferred > 0:
                return inferred
    return fallback


def _fallback_w(
    model: Any,
    *,
    hops: int,
    seed: int,
) -> CSRMatrix:
    from .graph import _make_blocky_mock

    try:
        params = list(getattr(model, "parameters", lambda **_: [])(recurse=True))
    except Exception:
        params = []

    d = max(16, min(4096, 256 if not params else sum(int(p.shape[-1]) if getattr(p, "ndim", 0) > 0 else 1 for p in params)))
    W = _make_blocky_mock(d=d, blocks=4, noise=0.01, seed=seed + len(params))

    import hashlib

    W.meta.setdefault("pytorch", {})
    W.meta["pytorch"].update(
        {
            "used_ops": [],
            "skipped_ops_count": 0,
            "hops": hops,
            "notes": "v0.9 last-dim heuristic (fallback)",
            "shapes": [],
            "trace_hash": hashlib.sha256(f"fallback|params={len(params)}|D={d}".encode("utf-8")).hexdigest(),
        }
    )
    return W


def build_w_from_pytorch(
    model: Any,
    example_inputs: Any,
    *,
    hops: int = 1,
    byte_weight: bool = True,
    reuse_decay: float = 0.7,
    max_len: int | None = None,
    seed: int = 0,
    attention_aware: bool = True,
    sectioning: bool = True,
    section_default: int = 64,
) -> CSRMatrix:
    """Profiler+FX v0 heuristic (banded W along last feature dim).

    If torch is not available, returns a deterministic synthetic W from params.
    """
    if not TORCH_AVAILABLE:
        return _fallback_w(model, hops=hops, seed=seed)

    import torch  # type: ignore
    from torch import fx  # type: ignore
    from torch.profiler import profile, ProfilerActivity  # type: ignore

    torch.manual_seed(seed)
    model_eval = model.eval()
    ex = example_inputs if isinstance(example_inputs, tuple) else (example_inputs,)

    # 1) Short profile (CPU/CUDA tolerant)
    activities = [ProfilerActivity.CPU]
    try:
        activities.append(ProfilerActivity.CUDA)
    except Exception:
        pass
    try:
        with profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
            with torch.no_grad():
                _ = model_eval(*(_clip_inputs(ex, max_len) if max_len is not None else ex))
        events = [e for e in prof.events() if getattr(e, 'key', None)]
    except Exception:
        events = []

    # 2) FX graph
    try:
        gm = fx.symbolic_trace(model_eval)
    except Exception:
        return _fallback_w(model, hops=hops, seed=seed)

    try:
        from torch.fx.passes.shape_prop import ShapeProp  # type: ignore

        ShapeProp(gm).propagate(*ex)
    except Exception:
        pass

    # 3) Infer feature dim D
    fallback_dim = _infer_feature_dim_from_tensor(ex[0])
    D = _infer_feature_dim_from_fx(gm, fallback=fallback_dim)
    feature_dim_source = "fx"
    feature_dim_overrides: List[str] = []
    if D == fallback_dim and fallback_dim > 0:
        feature_dim_source = "tensor"
    if D <= 0:
        D = 256
        feature_dim_source = "default"

    seq_len = _infer_seq_len_from_tensor(ex[0])
    tensor_penultimate = 0
    try:
        if hasattr(ex[0], "dim") and hasattr(ex[0], "shape") and ex[0].dim() >= 2:
            tensor_penultimate = int(ex[0].shape[-2])
    except Exception:
        tensor_penultimate = 0

    config = getattr(model, "config", None)
    def _config_int(attr: str) -> int:
        if config is None:
            return 0
        try:
            val = getattr(config, attr, None)
        except Exception:
            return 0
        if isinstance(val, (int, float)):
            ival = int(val)
            return ival if ival > 0 else 0
        return 0

    cfg_hs = _config_int("hidden_size")
    cfg_d_model = _config_int("d_model") or _config_int("model_dim")
    model_type = getattr(config, "model_type", None) if config is not None else None
    if model_type == "mamba":
        if cfg_hs > 0:
            cfg_primary = cfg_hs
            cfg_primary_source = "hf_config.hidden_size"
        elif cfg_d_model > 0:
            cfg_primary = cfg_d_model
            cfg_primary_source = "hf_config.d_model"
        else:
            cfg_primary = 0
            cfg_primary_source = None
    elif cfg_hs > 0:
        cfg_primary = cfg_hs
        cfg_primary_source = "hf_config.hidden_size"
    elif cfg_d_model > 0:
        cfg_primary = cfg_d_model
        cfg_primary_source = "hf_config.d_model"
    else:
        cfg_primary = 0
        cfg_primary_source = None
    cfg_inter = _config_int("intermediate_size")

    if seq_len > 0 and cfg_hs > 0 and D == seq_len and cfg_hs != D:
        D = cfg_hs
        feature_dim_source = "hf_config.hidden_size(seq_len_disambiguation)"
        feature_dim_overrides.append("seq_len_disambiguation")

    D_after_config, override_source = _maybe_override_feature_dim_from_config(model, D)
    if override_source is not None and D_after_config != D:
        feature_dim_source = override_source
        feature_dim_overrides.append(override_source)
    D = D_after_config

    if cfg_primary > 0 and D != cfg_primary:
        D = cfg_primary
        feature_dim_source = "config.primary_feature_dim"
        if cfg_primary_source:
            feature_dim_overrides.append(cfg_primary_source)
        feature_dim_overrides.append("config_primary")

    channel_candidate = tensor_penultimate if tensor_penultimate else seq_len
    if channel_candidate > 0 and D > channel_candidate:
        # Channel-first traces expose hidden width as the penultimate dim.
        channel_has_config_backing = (cfg_primary == channel_candidate) or (
            cfg_inter > 0 and channel_candidate > 0 and cfg_inter % channel_candidate == 0
        )
        ratio = D / float(channel_candidate)
        if ratio >= 1.2 and (channel_has_config_backing or cfg_primary == 0):
            D = channel_candidate
            feature_dim_source = "tensor.channel_first"
            feature_dim_overrides.append("channel_first")

    if D <= 0:
        D = 256
        feature_dim_source = "default"
        feature_dim_overrides.append("default_fallback")

    # 4) Op-based weight accumulation (coarse, but op-aware)
    total_weight = 0.0
    used_ops = set()
    skipped_ops = set()
    WEIGHTS = {
        "linear": 1.0,
        "matmul": 1.0,
        "mm": 1.0,
        "addmm": 1.0,
        "bmm": 0.8,
        # attention cores (cover fused/aten/3rd-party variants)
        "scaled_dot_product": 1.2,
        "scaled_dot_product_attention": 1.2,
        "aten::scaled_dot_product_attention": 1.2,
        "flash_attn": 1.2,
        "flashattention": 1.2,
        "mem_efficient_attention": 1.2,
        "add": 0.25,
        "transpose": 0.1,
        "permute": 0.1,
        "reshape": 0.1,
        "view": 0.05,
    }
    dtype = _infer_dtype_from_tensor(ex[0])
    bpf = _bytes_per_feature(dtype) if byte_weight else 1

    for e in events:
        key = str(getattr(e, 'key', ''))
        key_l = key.lower()
        matched = False
        for k, w in WEIGHTS.items():
            if k in key_l:
                total_weight += w * bpf
                used_ops.add(k)
                matched = True
                break
        if not matched and key:
            skipped_ops.add(key)

    if total_weight <= 0.0:
        total_weight = float(bpf)

    # 5) Build banded CSR favoring near neighbors up to 'hops'
    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    use_sectioning = bool(attention_aware or sectioning)
    sec_indptr: List[int] | None = None
    sec_indices: List[int] | None = None
    sec_data: List[float] | None = None
    att_meta: Dict[str, Any] | None = None
    if use_sectioning:
        # Prefer HuggingFace config metadata when available, then fall back to FX shapes
        head_dim = None
        num_heads = None
        try:
            cfg = getattr(model, "config", None)
            if cfg is not None:
                nh = int(getattr(cfg, "num_attention_heads", 0) or 0)
                hs = int(getattr(cfg, "hidden_size", 0) or 0)
                if nh > 0 and hs > 0 and hs % nh == 0:
                    head_dim = hs // nh
                    num_heads = nh
                kvh = int(getattr(cfg, "num_key_value_heads", 0) or 0)
                if head_dim is None and kvh > 0 and hs > 0 and hs % kvh == 0:
                    head_dim = hs // kvh
                    num_heads = kvh
        except Exception:
            pass
        if head_dim is None:
            for n in gm.graph.nodes:
                tm = n.meta.get("tensor_meta")
                shp = getattr(tm, "shape", None)
                if shp and hasattr(shp, "__len__") and len(shp) >= 2:
                    last = int(shp[-1])
                    for hd in (64, 80, 96, 112, 128, 160, 192):
                        if last % hd == 0 and 1 <= (last // hd) <= 256:
                            head_dim = int(hd)
                            num_heads = int(last // hd)
                            break
                if head_dim is not None:
                    break
        if head_dim is None:
            head_dim = int(section_default)
        section_size = head_dim
        from .graph import make_band_of_blocks
        Wsec = make_band_of_blocks(D, section_size=section_size, hops=hops, reuse_decay=reuse_decay)
        sec_indptr, sec_indices, sec_data = Wsec.indptr, Wsec.indices, Wsec.data
        att_meta = {"enabled": True, "num_heads": num_heads, "head_dim": head_dim, "section_size": section_size}
    else:
        for i in range(D):
            for d in range(1, min(hops + 1, D - i)):
                w = (total_weight / max(1, D)) * (reuse_decay ** d)
                rows.append(i)
                cols.append(i + d)
                vals.append(w)
                rows.append(i + d)
                cols.append(i)
                vals.append(w)

    if not vals:
        rows, cols, vals = [0], [0], [1e-6]

    # Convert COO->CSRMatrix or use sectioned
    if use_sectioning and sec_indptr is not None:
        indptr, indices, data = sec_indptr, sec_indices or [], sec_data or []
    else:
        indptr = [0]
        by_row: Dict[int, List[Tuple[int, float]]] = {}
        for r, c, v in zip(rows, cols, vals):
            if r == c:
                continue
            by_row.setdefault(r, []).append((c, float(max(0.0, v))))
        indices: List[int] = []
        data: List[float] = []
        for r in range(D):
            row = sorted(by_row.get(r, []))
            for c, v in row:
                indices.append(c)
                data.append(v)
            indptr.append(len(indices))
    # Collect op meta and compute trace_hash (names+shapes)
    used_meta: List[Dict[str, Any]] = []
    for n in gm.graph.nodes:
        if n.op not in ("call_function", "call_module"):
            continue
        tm = n.meta.get("tensor_meta")
        shp = getattr(tm, "shape", None)
        dtype = getattr(tm, "dtype", None)
        used_meta.append({
            "name": str(n.target),
            "shape": tuple(shp) if shp is not None else None,
            "dtype": str(dtype).split(".")[-1] if dtype is not None else None,
            "feature_dim": (len(shp) - 1) if hasattr(shp, "__len__") and len(shp) >= 2 else None,
            "last_dim_size": int(shp[-1]) if hasattr(shp, "__len__") and len(shp) >= 1 else None,
        })

    import hashlib, json as _json
    trace_hash = hashlib.sha256(_json.dumps([(m["name"], m["shape"]) for m in used_meta]).encode("utf-8")).hexdigest()

    inter_size = None
    try:
        config = getattr(model, "config", None)
        if config is not None:
            maybe_inter = getattr(config, "intermediate_size", None)
            if isinstance(maybe_inter, int) and maybe_inter > 0:
                inter_size = maybe_inter
    except Exception:
        inter_size = None

    meta = {
        "shape": D,
        "format": "csr",
        "nnz": len(data),
        "source": "pytorch",
        "seed": seed,
        "pytorch": {
            "used_ops": sorted(list(used_ops)) if used_ops else [m["name"] for m in used_meta],
            "skipped_ops_count": len(skipped_ops),
            "hops": hops,
            "notes": "v0.9 last-dim heuristic",
            "shapes": used_meta,
            "trace_hash": trace_hash,
            "feature_dim": D,
            "feature_dim_source": feature_dim_source,
            "feature_dim_overrides": feature_dim_overrides,
            "feature_dim_raw": fallback_dim if fallback_dim > 0 else None,
            "sequence_dim": seq_len if seq_len > 0 else None,
            "penultimate_dim": tensor_penultimate if tensor_penultimate > 0 else None,
            "nnz": len(data),
        },
    }
    if inter_size and D > 0:
        intermediate_meta: Dict[str, Any] = {"size": inter_size}
        if inter_size % D == 0:
            intermediate_meta["expansion_factor"] = inter_size // D
        meta["pytorch"]["intermediate"] = intermediate_meta
    if att_meta is not None:
        meta["pytorch"]["attention"] = att_meta
    return CSRMatrix(indptr=indptr, indices=indices, data=data, shape=(D, D), meta=meta)


def _clip_inputs(ex: tuple, max_len: int | None):
    try:
        import torch  # type: ignore
    except Exception:
        torch = None
    def clip(x):
        if torch is not None and hasattr(x, 'dim') and hasattr(x, 'shape') and x.dim() >= 2 and max_len is not None:
            sl = list(x.shape)
            sl[-2] = min(sl[-2], max_len)
            return x[..., :sl[-2], :]
        return x
    return tuple(clip(t) for t in ex)


def _infer_feature_dim_from_tensor(t: Any) -> int:
    try:
        if hasattr(t, 'dim') and hasattr(t, 'shape') and t.dim() >= 2:
            return int(t.shape[-1])
    except Exception:
        pass
    return 0


def _infer_dtype_from_tensor(t: Any):
    try:
        return getattr(t, 'dtype', None)
    except Exception:
        return None


def _bytes_per_feature(dtype: Any) -> int:
    # Try to derive from dtype; default 2 bytes
    try:
        itemsize = getattr(dtype, 'itemsize', None)
        if isinstance(itemsize, int) and itemsize > 0:
            return itemsize
    except Exception:
        pass
    return 2


__all__ = ["build_w_from_pytorch"]
