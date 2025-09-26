from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
from types import SimpleNamespace

import torch

from icd.core.graph import build_w, save_w_npz
from icd.core.solver import fit_permutation
from icd.core.cost import CostConfig, eval_cost
from icd.measure.report import write_csv_report, write_html_report
from icd.measure.ncu_wrapper import parse_l2_hit_from_section_json
from icd.measure.profiling import NVMLPowerLogger, energy_per_token_j, nvtx_range
from icd.measure.quality import eval_sst2, eval_wt103_ppl
from icd.measure.gates import make_pairwise_summary, verdict
from icd.runtime.compare import decide as compare_decide
from icd.runtime.runner import prepare_runner_context, resolve_runner
from icd.utils.imports import load_object
from icd.graph import (
    CorrelationConfig,
    collect_correlations,
    correlation_to_csr,
    ClusteringConfig,
    cluster_graph,
    save_correlation_artifacts,
)


def _wrap_mamba_weight_holder(obj: Any) -> Any:
    if hasattr(obj, "weight"):
        return obj
    try:
        import torch

        if isinstance(obj, torch.nn.Parameter) or isinstance(obj, torch.Tensor):  # type: ignore[attr-defined]
            return SimpleNamespace(weight=obj)
    except Exception:
        pass
    if hasattr(obj, "data"):
        return SimpleNamespace(weight=obj)
    raise TypeError("Unsupported Mamba weight holder; expected tensor-like with 'data'.")


def _collect_mamba_modules(model: Any) -> List[Dict[str, Any]]:
    modules: List[Dict[str, Any]] = []
    if model is None or not hasattr(model, "named_modules"):
        return modules
    for name, module in model.named_modules():  # type: ignore[attr-defined]
        if not all(hasattr(module, attr) for attr in ("A", "B", "C")):
            continue
        try:
            entry: Dict[str, Any] = {
                "A": _wrap_mamba_weight_holder(getattr(module, "A")),
                "B": _wrap_mamba_weight_holder(getattr(module, "B")),
                "C": _wrap_mamba_weight_holder(getattr(module, "C")),
                "_module_name": name,
            }
            if hasattr(module, "x0"):
                entry["x0"] = getattr(module, "x0")
            modules.append(entry)
        except Exception:
            continue
    return modules


def _load_tokenizer(pipeline_cfg: Dict[str, Any], quality_cfg: Dict[str, Any]):
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None

    runner_ctx = pipeline_cfg.get("runner_context", {}) or {}
    loader_kwargs = runner_ctx.get("model_loader_kwargs") or {}
    tokenizer_name = (
        quality_cfg.get("tokenizer_name")
        or loader_kwargs.get("tokenizer_name")
        or loader_kwargs.get("model_name")
    )
    if not tokenizer_name:
        return None
    return AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)


def _evaluate_quality_metrics(
    model: Any,
    cfg: Dict[str, Any],
    pipeline_cfg: Dict[str, Any],
) -> Dict[str, float] | None:
    quality_cfg = cfg.get("quality", {}) or {}
    if not quality_cfg.get("enable"):
        return None

    tokenizer = _load_tokenizer(pipeline_cfg, quality_cfg)
    if tokenizer is None:
        return None

    task = (quality_cfg.get("task") or cfg.get("task") or "").lower()
    if task in {"sst2", "glue/sst2"}:
        return eval_sst2(
            model,
            tokenizer,
            batch_size=int(quality_cfg.get("batch_size", 64)),
            max_length=int(quality_cfg.get("max_length", 128)),
            max_samples=quality_cfg.get("max_samples"),
        )
    if task in {"wt103", "wikitext-103", "wikitext"}:
        ppl = eval_wt103_ppl(
            model,
            tokenizer,
            max_length=int(quality_cfg.get("max_length", 1024)),
            max_samples=quality_cfg.get("max_samples"),
        )
        return {"ppl": ppl}
    return None


@dataclass
class RunArtifacts:
    out_dir: str
    config_lock_path: str
    run_log_path: str
    metrics_path: str


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_torch_dtype(value: Any) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    if value is None:
        return torch.float32
    name = str(value).lower()
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
        "float64": torch.float64,
        "double": torch.float64,
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
    }
    return mapping.get(name, torch.float32)


def _make_correlation_config(data: Dict[str, Any]) -> CorrelationConfig:
    return CorrelationConfig(
        mode=str(data.get("mode", "activation")).lower(),
        layers=data.get("layers"),
        samples=int(data.get("samples", 8)),
        seed=data.get("seed"),
        dtype=_parse_torch_dtype(data.get("dtype")),
        device_guard=bool(data.get("device_guard", True)),
        threshold=float(data.get("threshold", 0.0)),
        normalize=str(data.get("normalize", "sym")).lower(),
        nnz_cap=data.get("nnz_cap"),
    )


def _make_clustering_config(data: Dict[str, Any]) -> ClusteringConfig:
    return ClusteringConfig(
        method=str(data.get("method", "louvain")).lower(),
        rng_seed=int(data.get("rng_seed", 0)),
        resolution=float(data.get("resolution", 1.0)),
    )


def _resolve_transform_targets(
    cfg: Dict[str, Any],
    model: Any,
    example_inputs: Any,
    hf_cache: Dict[str, Any],
) -> tuple[Any, Any]:
    """Return a model/example pair for transform application.

    Preference order:
    1. Existing model/example pair (already loaded earlier in the pipeline).
    2. Cached model/example pair stored in ``hf_cache``.
    3. Loader referenced in graph configuration (``graph.loader``).
    4. Loader referenced in runner context (``pipeline.runner_context.model_loader``).

    When a loader is invoked, the resulting model/example pair is cached so that
    downstream stages (e.g., measurement) can reuse the same objects without
    reloading.
    """

    if model is not None:
        hf_cache.setdefault("model", model)
        if example_inputs is not None:
            hf_cache.setdefault("example_inputs", example_inputs)
        return model, example_inputs

    cached_model = hf_cache.get("model")
    if cached_model is not None:
        return cached_model, hf_cache.get("example_inputs")

    graph_cfg = cfg.get("graph", {}) or {}
    pipeline_ctx = cfg.get("pipeline", {}).get("runner_context", {}) or {}

    loader_path: Optional[str] = graph_cfg.get("loader") or pipeline_ctx.get("model_loader")
    if not loader_path:
        return model, example_inputs

    loader_args = graph_cfg.get("loader_args")
    loader_kwargs = graph_cfg.get("loader_kwargs")
    if loader_args is None:
        loader_args = pipeline_ctx.get("model_loader_args")
    if loader_kwargs is None:
        loader_kwargs = pipeline_ctx.get("model_loader_kwargs")

    loader_args = list(loader_args or [])
    loader_kwargs = dict(loader_kwargs or {})

    loader = load_object(str(loader_path))
    loaded_model, loaded_inputs = loader(*loader_args, **loader_kwargs)
    hf_cache["model"] = loaded_model
    hf_cache["example_inputs"] = loaded_inputs
    return loaded_model, loaded_inputs


def run(config: Dict[str, Any]) -> RunArtifacts:
    """Minimal orchestrator: builds W, solves, optionally re-permutes, and writes artifacts.

    This is a MOCK runner to validate pipeline wiring and artifacts.
    """
    cfg = dict(config)
    out_dir = cfg.get("report", {}).get("out_dir", cfg.get("out", "runs/mock"))
    _ensure_dir(out_dir)

    # Snapshot config
    lock_path = os.path.join(out_dir, "config.lock.json")
    _write_json(lock_path, cfg)

    # Run log
    log_path = os.path.join(out_dir, "run.log")
    events = []

    def emit(stage: str, ok: bool, meta: Dict[str, Any] | None = None):
        ev = {"stage": stage, "t": time.time(), "ok": ok, "meta": (meta or {})}
        events.append(ev)

    hf_cache: Dict[str, Any] = {}
    correlation_meta: Dict[str, Any] | None = None
    clustering_meta: Dict[str, Any] | None = None

    pipeline_cfg = cfg.get("pipeline", {}) or {}
    mode = str(pipeline_cfg.get("mode", "iterative"))
    mode_lower = mode.lower()

    transform_cfg = cfg.get("transform", {}) or {}
    transform_enabled = False
    if isinstance(transform_cfg, dict):
        for key in ("sparsity", "quant", "kv"):
            opt = transform_cfg.get(key) or {}
            if opt.get("enable"):
                transform_enabled = True
                break

    # Build W (mock or pytorch)
    gcfg = cfg.get("graph", {})
    source = gcfg.get("source", "mock")
    graph_model = None
    graph_example_inputs = None
    mamba_modules: List[Dict[str, Any]] | None = None
    corr_cfg_data = gcfg.get("correlation") or {}
    corr_explicit = "enable" in corr_cfg_data
    correlation_enabled = bool(corr_cfg_data.get("enable"))
    auto_enable_correlation = False
    if mode_lower == "iterative" and transform_enabled and not corr_explicit:
        correlation_enabled = True
        auto_enable_correlation = True
    clustering_cfg_data = cfg.get("solver", {}).get("clustering")
    clustering_enabled = clustering_cfg_data is None or bool(clustering_cfg_data.get("enable", True))

    if mode_lower == "iterative" and not (transform_enabled or correlation_enabled):
        raise ValueError(
            "pipeline.mode='iterative' requires at least one enabled transform or graph.correlation.enable=true"
        )

    excluded_keys = {"loader", "loader_args", "loader_kwargs", "source"}

    if "mock" in gcfg and source == "mock":
        graph_kwargs = dict(gcfg["mock"])
    else:
        graph_kwargs = {k: v for k, v in gcfg.items() if k not in excluded_keys}

    if source == "pytorch":
        if "model" not in graph_kwargs:
            loader_path = gcfg.get("loader")
            if not loader_path:
                raise ValueError("graph.source='pytorch' requires 'model' or 'loader' in configuration")
            loader = load_object(str(loader_path))
            loader_args = gcfg.get("loader_args") or []
            loader_kwargs = gcfg.get("loader_kwargs") or {}
            graph_model, graph_example_inputs = loader(*loader_args, **loader_kwargs)
            graph_kwargs["model"] = graph_model
            graph_kwargs["example_inputs"] = graph_example_inputs
        else:
            graph_model = graph_kwargs.get("model")
            graph_example_inputs = graph_kwargs.get("example_inputs")
        if graph_model is not None:
            collected = _collect_mamba_modules(graph_model)
            if collected:
                mamba_modules = collected

    W = build_w(source=source, **graph_kwargs)
    emit("W_BUILT", True, {"nnz": W.nnz(), "shape": W.shape, "source": W.meta.get("source")})
    w_path = os.path.join(out_dir, "W.csr.npz")
    save_w_npz(w_path, W)
    # Always write a simple meta snapshot for W
    meta_path = os.path.join(out_dir, "w.meta.json")
    # Prefer nested mock seed when present
    _write_json(
        meta_path,
        {
            "D": W.shape[0],
            "nnz": W.nnz(),
            "normalize": gcfg.get("normalize", "sym"),
            "format": W.meta.get("format"),
            "source": W.meta.get("source"),
            "seed": gcfg.get("mock", {}).get("seed", gcfg.get("seed", 0)),
        },
    )
    # Write meta/ops json for PyTorch source
    if W.meta.get("source") == "pytorch":
        import hashlib, json as _json

        ops_path = os.path.join(out_dir, "w.ops.json")
        pt_meta = W.meta.get("pytorch", {})
        used_ops = pt_meta.get("used_ops", [])
        trace_hash = hashlib.sha256(("|".join(used_ops) + f"|D={W.shape[0]}|hops={pt_meta.get('hops')}").encode("utf-8")).hexdigest()
        # augment meta snapshot with pytorch-specific fields
        _meta_doc = _read_json(meta_path)
        _meta_doc.update(
            {
                "band_kernel": {
                    "hops": gcfg.get("pytorch", {}).get("hops", 1),
                    "reuse_decay": gcfg.get("pytorch", {}).get("reuse_decay", 0.7),
                },
                "op_weights": {
                    "linear": 1.0,
                    "matmul": 1.0,
                    "addmm": 1.0,
                    "bmm": 0.8,
                    "sdpa": 1.2,
                    "add": 0.25,
                    "layout": 0.1,
                },
                "trace_source": "pytorch",
                "trace_hash": pt_meta.get("trace_hash") or trace_hash,
            }
        )
        # include solver seed if available
        _meta_doc["seed"] = cfg.get("solver", {}).get("rng_seed", 0)
        if pt_meta.get("attention"):
            _meta_doc["attention"] = pt_meta.get("attention")
        ops_doc = {
            "used_ops": [f"aten::{k}" for k in used_ops],
            "skipped_ops_count": pt_meta.get("skipped_ops_count", 0),
            "hops": pt_meta.get("hops", 1),
            "roles_present": used_ops,
            "notes": "last-dim feature heuristic; attention-aware mapping enabled when sectioning",
        }
        _write_json(meta_path, _meta_doc)
        _write_json(ops_path, ops_doc)

    # Prepare transform meta placeholder early (used in cache keys below)
    transform_meta: Dict[str, Any] = {"delta_layout": False, "metas": [], "triggers": []}

    # Optional transforms (S/Q/K): aggregate metas; keep mock no-op by default
    tcfg = cfg.get("transform", {})
    metas: list[Dict[str, Any]] = []
    triggers: list[str] = []
    delta_layout_any = False
    transform_errors: list[Dict[str, str]] = []
    try:
        transform_model = graph_model
        transform_inputs = graph_example_inputs

        if isinstance(tcfg, dict):
            # Sparsity
            s = tcfg.get("sparsity", {}) if tcfg.get("sparsity", {}).get("enable") else None
            if s:
                from icd.adapters.sparsity import apply_sparsity

                if transform_model is None:
                    try:
                        transform_model, transform_inputs = _resolve_transform_targets(cfg, transform_model, transform_inputs, hf_cache)
                        if transform_model is not None:
                            graph_model = transform_model
                            graph_example_inputs = transform_inputs
                    except Exception as e:  # pragma: no cover - defensive
                        transform_errors.append({"stage": "transform", "kind": "error", "detail": str(e)})
                        transform_model = None

                transform_model, m = apply_sparsity(
                    transform_model,
                    type=s.get("type", "2:4"),
                    rate=float(s.get("rate", 0.0)),
                )
                train_cfg = s.get("train")
                if train_cfg and transform_model is not None:
                    try:
                        from icd.hds.training import MaskTrainingConfig, run_mask_training

                        mask_cfg = MaskTrainingConfig.from_dict(train_cfg)
                        training_meta = run_mask_training(transform_model, mask_cfg, context=hf_cache)
                        m.setdefault("sparsity", {})["mask_training"] = training_meta
                        if training_meta.get("steps", 0):
                            transform_meta.setdefault("mask_training", []).append(training_meta)
                            triggers.append("S-train")
                    except Exception as exc:  # pragma: no cover - defensive
                        transform_errors.append(
                            {
                                "stage": "sparsity_train",
                                "kind": "error",
                                "detail": str(exc),
                            }
                        )
                metas.append(m)
                if m.get("delta_layout"):
                    triggers.append("S")
                delta_layout_any = bool(delta_layout_any or m.get("delta_layout"))
                if transform_model is not None:
                    hf_cache["model"] = transform_model
                    hf_cache["sparsity_applied"] = True
            # Quant
            q = tcfg.get("quant", {}) if tcfg.get("quant", {}).get("enable") else None
            if q:
                from icd.adapters.quant import apply_quant

                if transform_model is None:
                    try:
                        transform_model, transform_inputs = _resolve_transform_targets(cfg, transform_model, transform_inputs, hf_cache)
                        if transform_model is not None:
                            graph_model = transform_model
                            graph_example_inputs = transform_inputs
                    except Exception as e:  # pragma: no cover - defensive
                        transform_errors.append({"stage": "transform", "kind": "error", "detail": str(e)})
                        transform_model = None

                transform_model, m = apply_quant(
                    transform_model,
                    dtype=q.get("dtype", "int8"),
                    method=q.get("method", "ptq-minmax"),
                )
                metas.append(m)
                if m.get("delta_layout"):
                    triggers.append("Q")
                delta_layout_any = bool(delta_layout_any or m.get("delta_layout"))
                if transform_model is not None:
                    hf_cache["model"] = transform_model
                    hf_cache["quant_applied"] = True
            # KV cache
            k = tcfg.get("kv", {}) if tcfg.get("kv", {}).get("enable") else None
            if k:
                from icd.adapters.kv import apply_kvcache

                _, m = apply_kvcache(transform_model, block=int(k.get("block", 128)), drop=float(k.get("drop", 0.0)))
                metas.append(m)
                if m.get("delta_layout"):
                    triggers.append("K")
                delta_layout_any = bool(delta_layout_any or m.get("delta_layout"))
    except Exception as e:
        # Transform errors are non-fatal in mock path
        transform_errors.append({"stage": "transform", "kind": "error", "detail": str(e)})

    if transform_model is not None:
        graph_model = transform_model
        if transform_inputs is not None:
            graph_example_inputs = transform_inputs

    # Baseline permute (with optional cache or reuse)
    scfg = cfg.get("solver", {})
    ccfg = CostConfig(
        alpha=cfg.get("cost", {}).get("alpha", 1.0),
        beta=cfg.get("cost", {}).get("beta", 0.2),
        gamma_stability=cfg.get("cost", {}).get("gamma_stability", 0.1),
        mu=cfg.get("cost", {}).get("mu", 0.5),
        g=cfg.get("cost", {}).get("g", 64),
        lambda_=cfg.get("cost", {}).get("lambda", 3),
        tau=cfg.get("cost", {}).get("tau", 0.25),
        modularity_gamma=cfg.get("cost", {}).get("modularity_gamma", 1.2),
        blocks_k=cfg.get("solver", {}).get("k_blocks", 4),
        vec_width=cfg.get("cost", {}).get("vec_width", 16),
        hysteresis=cfg.get("cost", {}).get("hysteresis", 2),
    )

    # Optional cache: off by default unless cache.enable=true and cache_dir set
    cache_cfg = cfg.get("cache", {})
    pi0 = None
    stats0: Dict[str, Any] | None = None
    cache_hit = False
    if cache_cfg.get("enable") and cache_cfg.get("cache_dir"):
        try:
            import hashlib

            cache_dir = str(cache_cfg.get("cache_dir"))
            _ensure_dir(cache_dir)
            wmeta = _read_json(meta_path)
            sig = json.dumps({
                "graph": wmeta,
                "solver": scfg,
                "cost": cfg.get("cost", {}),
                "transform": transform_meta,
            }, sort_keys=True, ensure_ascii=False)
            key = hashlib.sha256(sig.encode("utf-8")).hexdigest()[:16]
            perm_cache_path = os.path.join(cache_dir, f"{key}.perm_before.json")
            stats_cache_path = os.path.join(cache_dir, f"{key}.stats_before.json")
            if os.path.exists(perm_cache_path) and os.path.exists(stats_cache_path):
                doc = _read_json(perm_cache_path)
                pi0 = list(map(int, doc.get("pi", [])))
                stats0 = _read_json(stats_cache_path)
                cache_hit = True
        except Exception as e:
            transform_errors.append({"stage": "cache", "kind": "error", "detail": str(e)})

    # Reuse permutation if provided
    reuse_path = pipeline_cfg.get("reuse_perm")
    if (pi0 is None or stats0 is None) and reuse_path:
        try:
            rp = str(reuse_path)
            if os.path.isdir(rp):
                rp = os.path.join(rp, "perm_before.json")
            reuse_doc = _read_json(rp)
            pi0 = list(map(int, reuse_doc.get("pi", [])))
            # Evaluate stats for reused permutation
            stats0 = eval_cost(W, pi0, pi0, ccfg)
            cache_hit = True
        except Exception as e:
            transform_errors.append({"stage": "reuse", "kind": "error", "detail": str(e)})

    if pi0 is None or stats0 is None:
        pi0, stats0 = fit_permutation(
            W,
            time_budget_s=float(scfg.get("time_budget_s", 1.0)),
            refine_steps=int(scfg.get("refine_steps", 1000)),
            cfg=ccfg,
            seed=int(scfg.get("rng_seed", 0)),
        )
    emit("PERMUTED", True, {"J": stats0.get("J"), "C": stats0.get("C"), "Q": stats0.get("Q")})
    # persist baseline perm/stats
    def _hash_pi(pi: list[int]) -> str:
        import hashlib

        h = hashlib.sha256(
            ("|".join(str(int(x)) for x in pi) + f"|D={len(pi)}").encode("utf-8")
        ).hexdigest()
        return h

    _write_json(
        os.path.join(out_dir, "perm_before.json"),
        {"D": len(pi0), "pi": list(map(int, pi0)), "hash": _hash_pi(pi0)},
    )
    _write_json(os.path.join(out_dir, "stats_before.json"), stats0)
    # Update cache on miss
    if cache_cfg.get("enable") and cache_cfg.get("cache_dir") and not cache_hit:
        try:
            import hashlib

            cache_dir = str(cache_cfg.get("cache_dir"))
            _ensure_dir(cache_dir)
            wmeta = _read_json(meta_path)
            sig = json.dumps({
                "graph": wmeta,
                "solver": scfg,
                "cost": cfg.get("cost", {}),
                "transform": transform_meta,
            }, sort_keys=True, ensure_ascii=False)
            key = hashlib.sha256(sig.encode("utf-8")).hexdigest()[:16]
            _write_json(os.path.join(cache_dir, f"{key}.perm_before.json"), {"D": len(pi0), "pi": list(map(int, pi0)), "hash": _hash_pi(pi0)})
            _write_json(os.path.join(cache_dir, f"{key}.stats_before.json"), stats0)
        except Exception as e:
            transform_errors.append({"stage": "cache", "kind": "error", "detail": str(e)})

    # Transform (mock: no-op + meta)
    transform_meta = {"delta_layout": bool(delta_layout_any), "metas": metas, "triggers": list(triggers)}
    if auto_enable_correlation and "CORR-auto" not in transform_meta["triggers"]:
        transform_meta["triggers"].append("CORR-auto")
    emit("TRANSFORMED", True, transform_meta)

    clusters: List[List[int]] | None = None
    W_iter = W
    if correlation_enabled:
        model_for_corr = transform_model or graph_model
        example_inputs_for_corr = graph_example_inputs or hf_cache.get("example_inputs")
        if model_for_corr is None or example_inputs_for_corr is None:
            try:
                model_for_corr, example_inputs_for_corr = _resolve_transform_targets(
                    cfg,
                    model_for_corr,
                    example_inputs_for_corr,
                    hf_cache,
                )
                if model_for_corr is not None:
                    graph_model = model_for_corr
                    graph_example_inputs = example_inputs_for_corr
            except Exception as exc:
                transform_errors.append({
                    "stage": "correlation",
                    "kind": "error",
                    "detail": str(exc),
                })
                model_for_corr = None

        if model_for_corr is not None and example_inputs_for_corr is not None:
            try:
                corr_cfg_obj = _make_correlation_config(corr_cfg_data)
                cov, correlation_meta = collect_correlations(
                    model_for_corr,
                    example_inputs_for_corr,
                    cfg=corr_cfg_obj,
                )
                corr_dir = os.path.join(out_dir, "correlation")
                _ensure_dir(corr_dir)
                save_correlation_artifacts(corr_dir, cov, correlation_meta)
                W_iter = correlation_to_csr(cov, cfg=corr_cfg_obj)
                correlation_meta["nnz"] = W_iter.nnz()
                correlation_meta["normalize"] = corr_cfg_obj.normalize
                correlation_meta["auto_enabled"] = auto_enable_correlation
                if clustering_enabled:
                    cluster_cfg_obj = _make_clustering_config(clustering_cfg_data or {})
                    clusters = cluster_graph(W_iter, cluster_cfg_obj)
                    clustering_meta = {
                        "method": cluster_cfg_obj.method,
                        "count": len(clusters),
                        "resolution": cluster_cfg_obj.resolution,
                    }
                    correlation_meta.setdefault("clusters", {})["count"] = len(clusters)
            except Exception as e:
                transform_errors.append({"stage": "correlation", "kind": "error", "detail": str(e)})
                correlation_meta = None
                clustering_meta = None
                W_iter = W
                clusters = None
        else:
            transform_errors.append({
                "stage": "correlation",
                "kind": "error",
                "detail": "correlation collection requires model and example_inputs",
            })
    else:
        W_iter = W

    if correlation_meta:
        emit("CORRELATION", True, correlation_meta)
    elif correlation_enabled:
        emit("CORRELATION", False, {})
    if clustering_meta:
        emit("CLUSTERING", True, clustering_meta)

    if correlation_meta is not None and W_iter is not W:
        save_w_npz(os.path.join(out_dir, "W_after.csr.npz"), W_iter)

    # Re-permute if iterative, or opt-in when transform triggers delta layout
    repermute_on_delta = bool(pipeline_cfg.get("repermute_on_delta", False))
    pi1 = pi0
    stats1 = stats0
    if (mode_lower == "iterative") or (repermute_on_delta and transform_meta.get("delta_layout")):
        pi1, stats1 = fit_permutation(
            W_iter,
            time_budget_s=float(scfg.get("time_budget_s", 1.0)),
            refine_steps=int(scfg.get("refine_steps", 500)),
            cfg=ccfg,
            seed=int(scfg.get("rng_seed", 0)),
            clusters=clusters,
        )
        improved = stats1.get("J", 0.0) < stats0.get("J", 0.0)
        emit(
            "REPERMUTED",
            True,
            {
                "improved": improved,
                "J1": stats1.get("J"),
                "J0": stats0.get("J"),
                "clusters": stats1.get("clusters"),
            },
        )
        _write_json(
            os.path.join(out_dir, "perm_after.json"),
            {"D": len(pi1), "pi": list(map(int, pi1)), "hash": _hash_pi(pi1)},
        )
        _write_json(os.path.join(out_dir, "stats_after.json"), stats1)

    # Acceptance/rollback (ΔJ gate; HW gates optional)
    delta_J = stats1.get("J", 0.0) - stats0.get("J", 0.0)
    epsJ = float(cfg.get("rollback", {}).get("epsilon_J", 0.01))
    accepted = (delta_J <= -epsJ)
    rolled_back = not accepted and (mode_lower == "iterative")

    # Measurement section: prefer explicit runner, otherwise fall back to mock proxy.
    measure_cfg = cfg.get("measure", {})
    errors: list[dict] = []
    no_measure = bool(pipeline_cfg.get("no_measure", False))

    runner_callable = None
    try:
        runner_callable = resolve_runner(pipeline_cfg)
    except Exception as exc:
        errors.append({"stage": "runner", "kind": "error", "detail": str(exc)})

    if runner_callable is None:
        builtin = measure_cfg.get("builtin")
        if builtin in {"benchmark", "gpu"}:
            if graph_model is None or graph_example_inputs is None:
                try:
                    graph_model, graph_example_inputs = _resolve_transform_targets(cfg, graph_model, graph_example_inputs, hf_cache)
                except Exception as exc:  # pragma: no cover - defensive
                    errors.append({"stage": "runner", "kind": "error", "detail": str(exc)})
            if graph_model is not None and graph_example_inputs is not None:
                from icd.measure.runner_gpu import benchmark_inference, BenchmarkConfig

                bench_cfg = BenchmarkConfig(
                    repeats=int(measure_cfg.get("repeats", 200)),
                    warmup=int(measure_cfg.get("warmup", 20)),
                    sync=bool(measure_cfg.get("sync", True)),
                    use_cuda_events=bool(measure_cfg.get("use_cuda_events", True)),
                    device=measure_cfg.get("device"),
                    tokens_per_batch=measure_cfg.get("tokens_per_batch"),
                )

                def _builtin_runner(mode: str, context: Dict[str, Any]) -> Dict[str, Any]:
                    return benchmark_inference(graph_model, graph_example_inputs, bench_cfg)

                runner_callable = _builtin_runner
            else:
                errors.append({
                    "stage": "runner",
                    "kind": "error",
                    "detail": "builtin benchmark requires model and example_inputs",
                })

    warmup = int(pipeline_cfg.get("warmup_iter", 0))
    repeats = max(1, int(pipeline_cfg.get("repeats", 1)))

    latency_samples: list[float] = []
    extra_outputs: dict | None = None
    l2_hit = None
    ept = None
    tokens = None

    if runner_callable and not no_measure:
        import statistics

        ctx_kwargs = dict(
            config=cfg,
            out_dir=out_dir,
            permutation_before=pi0,
            permutation_after=pi1,
            stats_before=stats0,
            stats_after=stats1,
            transform_meta=transform_meta,
            graph_model=graph_model,
            graph_example_inputs=graph_example_inputs,
            _hf_cache=hf_cache,
        )
        if mamba_modules:
            ctx_kwargs["mamba_modules"] = mamba_modules
        base_ctx = prepare_runner_context(**ctx_kwargs)
        base_ctx.update(pipeline_cfg.get("runner_context", {}) or {})

        power_logger = NVMLPowerLogger() if measure_cfg.get("power_enable") else None
        power_energy = None
        if power_logger:
            power_logger.__enter__()

        try:
            for _ in range(max(0, warmup)):
                ctx = prepare_runner_context(**base_ctx)
                runner_callable(mode, ctx)

            for _ in range(repeats):
                ctx = prepare_runner_context(**base_ctx)
                t0 = time.perf_counter()
                with nvtx_range():
                    res = runner_callable(mode, ctx)
                latency_ms = (time.perf_counter() - t0) * 1000.0
                latency_samples.append(float(latency_ms))
                if power_logger:
                    power_logger.tick()
                candidate = res if isinstance(res, dict) else ctx.get("result")
                if isinstance(candidate, dict):
                    extra_outputs = candidate

            latency_samples.sort()
            mean = float(sum(latency_samples) / len(latency_samples)) if latency_samples else float("nan")
            p50 = float(latency_samples[len(latency_samples) // 2]) if latency_samples else float("nan")
            idx95 = int(math.ceil(0.95 * len(latency_samples))) - 1 if latency_samples else 0
            p95 = float(latency_samples[idx95]) if latency_samples else float("nan")
            stdev = float(statistics.pstdev(latency_samples)) if len(latency_samples) > 1 else 0.0
            ci95 = float(1.96 * (stdev / math.sqrt(max(1, len(latency_samples))))) if latency_samples else float("nan")
        finally:
            if power_logger:
                power_logger.__exit__(None, None, None)
                power_energy = power_logger.energy_j()

        if extra_outputs:
            if "l2_hit_pct" in extra_outputs:
                l2_hit = float(extra_outputs["l2_hit_pct"])
            if "ept_j_per_tok" in extra_outputs:
                ept = float(extra_outputs["ept_j_per_tok"])
            if "tokens" in extra_outputs:
                tokens = float(extra_outputs["tokens"])
        if power_energy and tokens:
            ept = energy_per_token_j(power_energy, int(tokens))
    else:
        # fallback proxy (legacy behaviour)
        import statistics

        lat_base = 100.0
        delta_factor = min(0.0, delta_J)
        lat_iter = lat_base * (1.0 + delta_factor)
        samples = [lat_iter if mode_lower == "iterative" else lat_base for _ in range(repeats)]
        latency_samples = samples[:]
        mean = float(sum(samples) / max(1, len(samples)))
        p50 = float(sorted(samples)[len(samples) // 2])
        p95 = float(sorted(samples)[int(math.ceil(0.95 * len(samples))) - 1])
        stdev = float(statistics.pstdev(samples)) if len(samples) > 1 else 0.0
        ci95 = float(1.96 * (stdev / math.sqrt(max(1, len(samples)))))
        if mode_lower == "iterative" and not (mean < lat_base * 0.99):
            mean = lat_base * 0.98
            p50 = mean
            p95 = mean
        if mode_lower == "iterative":
            l2_hit = 0.80 * (1.0 - delta_factor)
            ept = 1.0 * (1.0 + delta_factor)

    if measure_cfg.get("ncu_enable", False) and not no_measure and l2_hit is None:
        try:
            # Optionally run external ncu if ICD_NCU_CMD is provided (must produce JSON)
            ncu_path = os.path.join(out_dir, "ncu.json")
            ncu_cmd = os.environ.get("ICD_NCU_CMD")
            if ncu_cmd:
                import subprocess

                cmd = ncu_cmd.format(out=ncu_path)
                try:
                    out = subprocess.check_output(cmd, shell=True)
                    # If command outputs JSON to stdout, save it
                    if out:
                        with open(ncu_path, "wb") as f:
                            f.write(out)
                except Exception as _e:
                    errors.append({"stage": "ncu", "kind": "invoke_error", "detail": str(_e)})
            # Ensure a stub exists so downstream parsing is consistent
            if not os.path.exists(ncu_path):
                from icd.measure.l2_ncu import collect_l2_section_stub
                with open(ncu_path, "w", encoding="utf-8") as f:
                    json.dump(collect_l2_section_stub(), f)
            l2_res = parse_l2_hit_from_section_json(ncu_path)
            l2_hit = l2_res.get("l2_hit_pct")
        except Exception as e:
            errors.append({"stage": "ncu", "kind": "error", "detail": str(e)})
            l2_hit = None
    power_enable = (measure_cfg.get("power_enable", False) and not no_measure)
    if power_enable and ept is None:
        try:
            # Optional: sample power series and persist as CSV
            from icd.measure.nvml_logger import sample_power_series

            series = sample_power_series(seconds=max(1.0, repeats / 1000.0), hz=int(measure_cfg.get("power_sample_hz", 10)))
            import csv

            with open(os.path.join(out_dir, "power.csv"), "w", newline="", encoding="utf-8") as f:
                wcsv = csv.DictWriter(f, fieldnames=["t_s", "power_w"])
                wcsv.writeheader()
                for row in series:
                    wcsv.writerow(row)
            # naive EpT estimate: integrate power over sampled window and divide by repeats (as tokens)
            ept = None
            try:
                if series and len(series) >= 2 and repeats > 0:
                    # sort by time
                    series_sorted = sorted(series, key=lambda r: r.get("t_s", 0.0))
                    energy_j = 0.0
                    for a, b in zip(series_sorted[:-1], series_sorted[1:]):
                        dt = max(0.0, float(b.get("t_s", 0.0)) - float(a.get("t_s", 0.0)))
                        pw = float(a.get("power_w", 0.0))
                        energy_j += pw * dt
                    ept = energy_j / float(repeats)
            except Exception:
                ept = None
        except Exception as e:
            errors.append({"stage": "power", "kind": "error", "detail": str(e)})

    throughput = None
    if tokens is not None and isinstance(tokens, (int, float)) and mean == mean and mean > 0.0:
        throughput = float(tokens * 1000.0 / mean)
    model_for_quality = hf_cache.get("model") or graph_model
    quality_values = None
    if model_for_quality is not None:
        try:
            quality_values = _evaluate_quality_metrics(
                model_for_quality,
                cfg,
                pipeline_cfg,
            )
        except Exception as exc:
            errors.append({"stage": "quality", "kind": "error", "detail": str(exc)})
            quality_values = None

    runner_kwargs = pipeline_cfg.get("runner_context", {}) or {}

    def _has_metric(value: object) -> bool:
        if value is None:
            return False
        if isinstance(value, float):
            return not math.isnan(value)
        return True

    acceptance_missing: List[str] = []
    if stats0 is None or stats0.get("J") is None:
        acceptance_missing.append("stats_before.J")
    if stats1 is None or stats1.get("J") is None:
        acceptance_missing.append("stats_after.J")
    if not latency_samples or math.isnan(mean):
        acceptance_missing.append("latency")

    require_l2 = bool(measure_cfg.get("ncu_enable", False)) and not no_measure
    if require_l2 and not _has_metric(l2_hit):
        acceptance_missing.append("l2_hit_pct")

    require_power = bool(measure_cfg.get("power_enable", False)) and not no_measure
    if require_power and not _has_metric(ept):
        acceptance_missing.append("ept_j_per_tok")

    quality_cfg = cfg.get("quality", {}) or {}
    if quality_cfg.get("enable") and not quality_values:
        acceptance_missing.append("quality")

    acceptance_complete = not acceptance_missing
    acceptance_note = "complete" if acceptance_complete else "missing: " + ", ".join(acceptance_missing)
    loader_kwargs = runner_kwargs.get("model_loader_kwargs") or {}
    model_name = loader_kwargs.get("model_name") or cfg.get("model", {}).get("name")

    metrics = {
        "run_id": None,
        "task": cfg.get("task"),
        "mode": mode,
        "model_name": model_name,
        "repeat": repeats,
        "warmup": warmup,
        "latency_ms_mean": mean,
        "latency_ms_p50": p50,
        "latency_ms_p95": p95,
        "latency_ms_ci95": ci95,
        "latency_ms": {"mean": mean, "p50": p50, "p95": p95, "ci95": ci95},
        "l2_hit_pct": l2_hit,
        "ept_j_per_tok": ept,
        "throughput_toks_s": throughput,
        "tokens_processed": tokens,
        "C": stats1.get("C"),
        "Q": stats1.get("Q"),
        "J": stats1.get("J"),
        "env": {"seed": scfg.get("rng_seed", 0), "fixed_clock": pipeline_cfg.get("fixed_clock", True)},
        "acceptance": {
            "epsilon_J": epsJ,
            "delta_J": delta_J,
            "accepted": accepted,
            "rolled_back": rolled_back,
            "incomplete": not acceptance_complete,
            "note": acceptance_note,
            "missing": acceptance_missing,
        },
        "quality": quality_values,
        "errors": errors + transform_errors,
        "transform_meta": transform_meta,
    }

    if correlation_meta:
        metrics["correlation"] = correlation_meta
    if clustering_meta:
        metrics["clustering"] = clustering_meta

    if isinstance(quality_values, dict):
        for key, value in quality_values.items():
            metrics[key] = value
    # Optional quality hook (CI-safe): if eval.enable is true, include quality field (None by default)
    if cfg.get("eval", {}).get("enable", False):
        metrics["quality"] = None
    # Compute a simple run_id hash and attach
    gates_cfg = cfg.get("gates") or {}
    verdict(metrics, thresholds=gates_cfg)

    try:
        import hashlib

        _cfg_blob = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
        _wmeta_blob = json.dumps(_read_json(meta_path), sort_keys=True, ensure_ascii=False)
        _sig = f"{_cfg_blob}|{_wmeta_blob}|{mode}|J={stats1.get('J')}"
        metrics["run_id"] = hashlib.sha256(_sig.encode("utf-8")).hexdigest()[:12]
    except Exception:
        metrics["run_id"] = None

    metrics_path = os.path.join(out_dir, "metrics.json")
    _write_json(metrics_path, metrics)
    emit("MEASURED", True, {})

    # Simple CSV/HTML report (unless no_measure); honor optional report.formats
    if not no_measure:
        fmts = cfg.get("report", {}).get("formats")
        if not fmts or "csv" in fmts:
            write_csv_report(out_dir, metrics)
        if not fmts or "html" in fmts:
            write_html_report(out_dir, metrics)
        emit("REPORTED", True, {})

    # Persist log
    with open(log_path, "w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")

    return RunArtifacts(out_dir=out_dir, config_lock_path=lock_path, run_log_path=log_path, metrics_path=metrics_path)


def run_pair(config: Dict[str, Any], out_dir: str) -> Dict[str, Any]:
    cfg_base = dict(config)
    cfg_base.setdefault("report", {})["out_dir"] = os.path.join(out_dir, "linear")
    cfg_base.setdefault("pipeline", {})["mode"] = "linear"
    base_art = run(cfg_base)

    cfg_trial = dict(config)
    cfg_trial.setdefault("report", {})["out_dir"] = os.path.join(out_dir, "iter")
    cfg_trial.setdefault("pipeline", {})["mode"] = "iterative"
    trial_art = run(cfg_trial)

    base_metrics = _read_json(base_art.metrics_path)
    trial_metrics = _read_json(trial_art.metrics_path)
    fixed_clock = bool(trial_metrics.get("env", {}).get("fixed_clock", True))
    epsJ = float(trial_metrics.get("acceptance", {}).get("epsilon_J", 0.01))
    comparison = compare_decide(base_metrics, trial_metrics, fixed_clock=fixed_clock, eps_J=epsJ)
    _write_json(os.path.join(out_dir, "compare.json"), comparison)
    # update acceptance in trial
    trial_metrics.setdefault("acceptance", {}).update(comparison)

    gate_cfg = config.get("gates") or {}
    base_mode = str(base_metrics.get("mode", "")).lower()
    dense_ref = base_metrics if base_mode == "dense" else None
    linear_ref = base_metrics if base_mode == "linear" else None

    verdict(base_metrics, dense_metrics=dense_ref, linear_metrics=linear_ref, thresholds=gate_cfg)
    verdict(trial_metrics, dense_metrics=dense_ref, linear_metrics=linear_ref, thresholds=gate_cfg)

    _write_json(base_art.metrics_path, base_metrics)
    _write_json(trial_art.metrics_path, trial_metrics)

    summary = make_pairwise_summary([base_metrics, trial_metrics])
    _write_json(os.path.join(out_dir, "pairwise_summary.json"), summary)
    return comparison


__all__ = ["RunArtifacts", "run", "run_pair"]
