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

from .graph import CSRMatrix


def _maybe_override_feature_dim_from_config(model: Any, current_dim: int) -> tuple[int, str | None]:
    """Use HuggingFace config metadata to refine feature dimension when available."""

    config = getattr(model, "config", None)
    if config is None:
        return current_dim, None

    try:
        model_type = str(getattr(config, "model_type", ""))
    except Exception:
        model_type = ""

    hidden_size = getattr(config, "hidden_size", None)
    if isinstance(hidden_size, int) and hidden_size > 0:
        if model_type.lower() == "mamba" and hidden_size != current_dim:
            return hidden_size, "hf_config.hidden_size"
        if current_dim <= 0:
            return hidden_size, "hf_config.hidden_size"

    return current_dim, None


def _infer_feature_dim_from_fx(gm: Any, fallback: int = 0) -> int:
    """Infer feature dimension from FX trace metadata.

    Falls back to the provided ``fallback`` when inference fails. Chooses the
    most common positive last-dimension size observed across FX nodes.
    """

    if gm is None:
        return fallback

    dims: List[int] = []
    graph = getattr(gm, "graph", None)
    nodes = getattr(graph, "nodes", []) if graph is not None else []
    for node in nodes:
        meta = getattr(node, "meta", {})
        tensor_meta = meta.get("tensor_meta") if isinstance(meta, dict) else getattr(meta, "tensor_meta", None)
        shape = getattr(tensor_meta, "shape", None)
        if shape is None or not hasattr(shape, "__len__") or len(shape) < 2:
            continue
        try:
            last = int(shape[-1])
        except Exception:
            continue
        if last > 0:
            dims.append(last)

    if not dims:
        return fallback

    counts = Counter(dims)
    inferred, _ = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
    return inferred if inferred > 0 else fallback


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
    if D == fallback_dim and fallback_dim > 0:
        feature_dim_source = "tensor"
    if D <= 0:
        D = 256
        feature_dim_source = "default"

    D, override_source = _maybe_override_feature_dim_from_config(model, D)
    if override_source is not None:
        feature_dim_source = override_source
    if D <= 0:
        D = 256
        feature_dim_source = "default"

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
        "scaled_dot_product": 1.2,  # attention core
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
        matched = False
        for k, w in WEIGHTS.items():
            if k in key:
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
        # Heuristic head_dim inference from FX shapes
        head_dim = None
        num_heads = None
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
        })

    import hashlib, json as _json
    trace_hash = hashlib.sha256(_json.dumps([(m["name"], m["shape"]) for m in used_meta]).encode("utf-8")).hexdigest()

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
        },
    }
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
        if hasattr(t, 'numel'):
            return int(t.numel())
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
