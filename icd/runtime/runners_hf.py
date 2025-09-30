"""Runners that execute HuggingFace models for real measurements."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple, List

from collections.abc import Mapping
from types import SimpleNamespace

import torch

from icd.adapters.quant import (
    QuantConfig,
    apply_quant_from_config,
    repack_linear_after_permutation,
)
from icd.adapters.sparsity import SparsityConfig, apply_sparsity_from_config
from icd.runtime.apply_pi import apply_pi_to_bert, apply_pi_to_mamba, apply_pi_to_mamba_hf, perm_signature_from_iterable


def _wrap_mamba_weight_holder(obj: Any) -> Any:
    if hasattr(obj, "weight"):
        return obj
    try:
        if isinstance(obj, torch.nn.Parameter) or torch.is_tensor(obj):
            return SimpleNamespace(weight=obj)
    except Exception:
        pass
    if hasattr(obj, "data"):
        return SimpleNamespace(weight=getattr(obj, "data"))
    raise TypeError("Unsupported Mamba weight holder; expected tensor-like with 'data'.")


def _collect_mamba_modules_from_model(model: Any) -> List[Dict[str, Any]]:
    modules: List[Dict[str, Any]] = []
    if model is None or not hasattr(model, "named_modules"):
        return modules
    for name, module in model.named_modules():  # type: ignore[attr-defined]
        # Check for original mamba-ssm structure (A, B, C)
        if all(hasattr(module, attr) for attr in ("A", "B", "C")):
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
        # Check for HuggingFace Mamba structure (MambaMixer with in_proj, x_proj, out_proj)
        elif all(hasattr(module, attr) for attr in ("in_proj", "x_proj", "out_proj")):
            try:
                entry: Dict[str, Any] = {
                    "A": _wrap_mamba_weight_holder(getattr(module, "in_proj")),
                    "B": _wrap_mamba_weight_holder(getattr(module, "x_proj")),
                    "C": _wrap_mamba_weight_holder(getattr(module, "out_proj")),
                    "_module_name": name,
                    "_hf_mamba": True,  # Flag for HF Mamba
                    "_module_ref": module,  # Reference to actual module for A_log, D, conv1d
                }
                modules.append(entry)
            except Exception:
                continue
    return modules
from icd.utils.imports import load_object


def _ensure_cache(context: Dict[str, Any]) -> Dict[str, Any]:
    cache = context.setdefault("_hf_cache", {})
    return cache


def _maybe_from_context(context: Dict[str, Any], context_key: str, cache_key: str):
    value = context.get(context_key)
    if value is not None:
        cache = _ensure_cache(context)
        cache[cache_key] = value


def _resolve_model_and_inputs(context: Dict[str, Any]) -> Tuple[Any, Tuple[Any, ...]]:
    cache = _ensure_cache(context)
    model = cache.get("model")
    example_inputs = cache.get("example_inputs")

    if model is None or example_inputs is None:
        _maybe_from_context(context, "graph_model", "model")
        _maybe_from_context(context, "graph_example_inputs", "example_inputs")
        model = cache.get("model", model)
        example_inputs = cache.get("example_inputs", example_inputs)

    if model is None or example_inputs is None:
        loader_path = context.get("model_loader")
        if not loader_path:
            raise ValueError("runner context must provide 'model_loader' when model is not preloaded")
        loader = load_object(str(loader_path))
        loader_kwargs = context.get("model_loader_kwargs") or {}
        loader_args = context.get("model_loader_args") or []
        model, example_inputs = loader(*loader_args, **loader_kwargs)
        cache["model"] = model
        cache["example_inputs"] = example_inputs
    return model, example_inputs


def _iter_quantized_children(root: torch.nn.Module) -> Iterable[tuple[torch.nn.Module, str, torch.nn.Module]]:
    for module in root.modules():
        for name, child in list(module.named_children()):
            yield module, name, child


def _repack_quantized_modules(model: torch.nn.Module) -> None:
    for parent, name, child in _iter_quantized_children(model):
        new_child = repack_linear_after_permutation(child)
        if new_child is not child:
            setattr(parent, name, new_child)


def _tensor_from_pi(model: Any, pi_seq: Iterable[int]) -> torch.LongTensor:
    try:
        device = next(model.parameters()).device  # type: ignore[attr-defined]
    except StopIteration:
        device = torch.device("cpu")
    return torch.as_tensor(list(pi_seq), device=device, dtype=torch.long)


def _apply_pi_sequence(
    model: Any,
    context: Dict[str, Any],
    pi_seq: Iterable[int] | None,
    quant_cfg: QuantConfig,
) -> None:
    if pi_seq is None:
        return
    signature = perm_signature_from_iterable(pi_seq)
    cache = _ensure_cache(context)
    applied = cache.setdefault("pi_signatures", set())
    if signature in applied:
        return

    pi_tensor = _tensor_from_pi(model, pi_seq)
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if model_type == "bert" or hasattr(model, "bert"):
        apply_pi_to_bert(model, pi_tensor)
    elif model_type == "mamba":
        modules = context.get("mamba_modules")
        if modules is None:
            modules = _collect_mamba_modules_from_model(model)
            if modules:
                context["mamba_modules"] = modules
        if modules is None:
            raise RuntimeError("runner context missing 'mamba_modules' for Mamba permutation application")
        if isinstance(modules, Mapping):
            if modules.get("_hf_mamba"):
                apply_pi_to_mamba_hf(modules, pi_tensor)
            else:
                apply_pi_to_mamba(modules, pi_tensor)
        else:
            any_applied = False
            for entry in modules:
                if entry.get("_hf_mamba"):
                    apply_pi_to_mamba_hf(entry, pi_tensor)
                else:
                    apply_pi_to_mamba(entry, pi_tensor)
                any_applied = True
            if not any_applied:
                raise RuntimeError("no applicable Mamba modules found for permutation application")
    else:
        apply_pi_to_bert(model, pi_tensor)

    applied.add(signature)
    cache["last_pi_signature"] = signature

    if quant_cfg.type != "none":
        _repack_quantized_modules(model)


def _prepare_model_for_mode(mode: str, context: Dict[str, Any], model: Any) -> Any:
    cache = _ensure_cache(context)
    prepared_key = f"prepared::{mode}"
    current_model = cache.get("model", model)
    if cache.get(prepared_key):
        return current_model

    config = context.get("config", {}) or {}
    sparsity_cfg = SparsityConfig.from_dict(config.get("sparsity"))
    quant_cfg = QuantConfig.from_dict(config.get("quant"))

    def apply_sparsity_once() -> None:
        if cache.get("sparsity_applied"):
            return
        apply_sparsity_from_config(current_model, sparsity_cfg)
        cache["sparsity_applied"] = True

    def apply_quant_once() -> None:
        if cache.get("quant_applied"):
            return
        nonlocal current_model
        new_model = apply_quant_from_config(current_model, quant_cfg)
        if new_model is not current_model:
            current_model = new_model
            cache["model"] = current_model
            context["graph_model"] = current_model
        cache["quant_applied"] = True

    pi_before = context.get("permutation_before")
    pi_after = context.get("permutation_after")

    def pi_value(pi) -> Iterable[int] | None:
        if pi is None:
            return None
        if isinstance(pi, torch.Tensor):
            return pi.tolist()
        return list(pi)

    pi0 = pi_value(pi_before)
    pi1 = pi_value(pi_after)

    mode_lower = (mode or "").lower()

    if mode_lower in {"dense", "distill"}:
        pass
    elif mode_lower == "perm-only":
        _apply_pi_sequence(current_model, context, pi1 or pi0, quant_cfg)
    elif mode_lower == "sparsity-only":
        apply_sparsity_once()
        apply_quant_once()
    elif mode_lower == "linear":
        if quant_cfg.order == "permute-then-quant":
            _apply_pi_sequence(current_model, context, pi0 or pi1, quant_cfg)
            apply_sparsity_once()
            apply_quant_once()
            if pi1 is not None:
                _apply_pi_sequence(current_model, context, pi1, quant_cfg)
        else:
            apply_sparsity_once()
            apply_quant_once()
            _apply_pi_sequence(current_model, context, pi1 or pi0, quant_cfg)
    elif mode_lower == "iterative":
        _apply_pi_sequence(current_model, context, pi0, quant_cfg)
        apply_sparsity_once()
        apply_quant_once()
        _apply_pi_sequence(current_model, context, pi1 or pi0, quant_cfg)
    else:
        # fallback: behave like dense with optional transforms
        apply_sparsity_once()
        apply_quant_once()

    cache[prepared_key] = True
    cache.setdefault("model", current_model)
    return cache.get("model") or current_model


def hf_sequence_classifier_runner(mode: str, context: Dict[str, Any]) -> Dict[str, Any]:
    model, example_inputs = _resolve_model_and_inputs(context)
    model = _prepare_model_for_mode(mode, context, model)

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("torch is required for HuggingFace runners") from exc

    args = example_inputs if isinstance(example_inputs, (tuple, list)) else (example_inputs,)

    with torch.no_grad():
        model(*args)

    tokens = None
    if args:
        first = args[0]
        if hasattr(first, "shape") and len(first.shape) >= 2:
            tokens = int(first.shape[0] * first.shape[1])

    result: Dict[str, Any] = {}
    if tokens is not None:
        result["tokens"] = tokens
    return result


def hf_causal_lm_runner(mode: str, context: Dict[str, Any]) -> Dict[str, Any]:
    model, example_inputs = _resolve_model_and_inputs(context)
    model = _prepare_model_for_mode(mode, context, model)

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("torch is required for HuggingFace runners") from exc

    if isinstance(example_inputs, dict):
        args = ()
        kwargs = example_inputs
    else:
        args = example_inputs if isinstance(example_inputs, (tuple, list)) else (example_inputs,)
        kwargs = {}

    with torch.no_grad():
        model(*args, **kwargs)

    tokens = None
    if args:
        first = args[0]
        if hasattr(first, "shape") and len(first.shape) >= 2:
            tokens = int(first.shape[0] * first.shape[1])

    result: Dict[str, Any] = {}
    if tokens is not None:
        result["tokens"] = tokens
    return result


__all__ = ["hf_sequence_classifier_runner", "hf_causal_lm_runner"]
