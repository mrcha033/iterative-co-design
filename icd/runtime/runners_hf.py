"""Runners that execute HuggingFace models for real measurements."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from icd.utils.imports import load_object


def _ensure_cache(context: Dict[str, Any]) -> Dict[str, Any]:
    return context.setdefault("_hf_cache", {})


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


def hf_sequence_classifier_runner(mode: str, context: Dict[str, Any]) -> Dict[str, Any]:
    del mode  # unused for now; hook for future mode-specific logic
    model, example_inputs = _resolve_model_and_inputs(context)

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
    del mode
    model, example_inputs = _resolve_model_and_inputs(context)

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
