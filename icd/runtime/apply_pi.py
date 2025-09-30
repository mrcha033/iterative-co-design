from __future__ import annotations

"""Permutation application utilities for HuggingFace runners."""

from dataclasses import dataclass, field
import hashlib
import importlib
import importlib.util
import os
from importlib import metadata
from typing import Iterable, Mapping, Any

import torch
from torch import nn

try:  # pragma: no cover - optional dependency during type checking
    from torch.fx.proxy import Proxy
except ImportError:  # pragma: no cover
    Proxy = tuple()  # type: ignore[assignment]

__all__ = [
    "inv_perm",
    "reindex_vec",
    "reindex_rows",
    "reindex_cols",
    "PWP_inv",
    "apply_pi_to_bert",
    "apply_pi_to_mamba",
    "perm_signature",
    "perm_signature_from_iterable",
    "StableHLOCapability",
    "detect_stablehlo_capability",
    "require_stablehlo_capability",
]


def inv_perm(pi: torch.LongTensor) -> torch.LongTensor:
    """Return the inverse permutation of ``pi``."""

    if pi.ndim != 1:
        raise ValueError("pi must be 1-D")
    inv = torch.empty_like(pi)
    inv[pi] = torch.arange(pi.numel(), device=pi.device, dtype=pi.dtype)
    return inv


def _validate_perm(pi: torch.LongTensor, length: int) -> torch.LongTensor:
    if pi.ndim != 1:
        raise ValueError("permutation must be 1-D")
    if pi.numel() != length:
        raise ValueError(f"permutation length {pi.numel()} does not match expected {length}")
    sorted_pi = torch.sort(pi.cpu()).values
    if not torch.equal(sorted_pi, torch.arange(length, dtype=sorted_pi.dtype)):
        raise ValueError("permutation must contain each index exactly once in [0, length)")
    return pi


def reindex_vec(v: torch.Tensor, pi: torch.LongTensor) -> torch.Tensor:
    return v.index_select(0, pi).contiguous()


def reindex_rows(W: torch.Tensor, pi: torch.LongTensor) -> torch.Tensor:
    return W.index_select(0, pi).contiguous()


def reindex_cols(W: torch.Tensor, perm: torch.LongTensor) -> torch.Tensor:
    return W.index_select(1, perm).contiguous()


def PWP_inv(W: torch.Tensor, pi: torch.LongTensor, pinv: torch.LongTensor) -> torch.Tensor:
    """Return ``P · W · P^{-1}`` for hidden ``→`` hidden mappings."""

    if W.ndim != 2:
        raise ValueError("W must be a 2-D tensor")
    if W.shape[0] != W.shape[1]:
        raise ValueError("W must be square for hidden→hidden mappings")
    _validate_perm(pi, W.shape[0])
    _validate_perm(pinv, W.shape[0])
    expected_pi = inv_perm(pinv)
    if not torch.equal(expected_pi.to(device=pi.device, dtype=pi.dtype), pi):
        raise ValueError("pi and pinv must be inverse permutations")
    return reindex_rows(reindex_cols(W, pinv), pi)


def PTWP(W: torch.Tensor, pi: torch.LongTensor) -> torch.Tensor:
    """Return ``Pᵀ · W · P`` without explicit matrices."""

    return reindex_rows(reindex_cols(W, pi), pi)


def Pinv_W_P(W: torch.Tensor, pi: torch.LongTensor, pinv: torch.LongTensor) -> torch.Tensor:
    """Return ``P · W · P^{-1}`` using the provided forward and inverse permutations."""

    expected_pi = inv_perm(pinv)
    if not torch.equal(expected_pi, pi):
        raise ValueError("pi and pinv must be inverse permutations")
    return reindex_rows(reindex_cols(W, pi), expected_pi)


try:  # pragma: no cover - defensive convenience for tests
    import builtins as _builtins

    for _name in ("reindex_rows", "reindex_cols"):
        if not hasattr(_builtins, _name):
            setattr(_builtins, _name, globals()[_name])
except Exception:
    pass


def make_head_block_sigma(pi: torch.LongTensor, num_heads: int, head_dim: int) -> tuple[torch.LongTensor, torch.LongTensor]:
    """Permutation that restores head-contiguous layout after applying hidden permutation."""

    H = pi.numel()
    device = pi.device
    dtype = pi.dtype

    if num_heads <= 0 or head_dim <= 0 or num_heads * head_dim != H:
        sigma = torch.arange(H, device=device, dtype=dtype)
        return sigma, sigma

    head_idx = pi // head_dim
    within = pi % head_dim
    order: list[torch.LongTensor] = []
    for h in range(int(num_heads)):
        pos = (head_idx == h).nonzero(as_tuple=False).flatten()
        if pos.numel() == 0:
            continue
        pos_sorted = pos[torch.argsort(within[pos])]
        order.append(pos_sorted)
    sigma = torch.cat(order) if order else torch.arange(H, device=device, dtype=dtype)
    sinv = torch.argsort(sigma)
    return sigma, sinv


def perm_signature_from_iterable(seq: Iterable[int]) -> str:
    values = [int(x) for x in seq]
    pack = ",".join(str(v) for v in values).encode("utf-8")
    digest = hashlib.sha256(pack).hexdigest()
    return f"{len(values)}:{digest}"


def perm_signature(pi: torch.LongTensor) -> str:
    return perm_signature_from_iterable(pi.detach().cpu().tolist())


@dataclass(frozen=True)
class StableHLOCapability:
    """Represents the runtime readiness of StableHLO integrations."""

    available: bool
    reason: str | None = None
    details: Mapping[str, object] = field(default_factory=dict)

    def __bool__(self) -> bool:  # pragma: no cover - provided for ergonomics
        return self.available


_DISABLE_ENV = "ICD_DISABLE_STABLEHLO"


def _is_env_disabled(env: Mapping[str, str]) -> str | None:
    raw = env.get(_DISABLE_ENV)
    if raw is None:
        return None
    normalized = raw.strip().lower()
    if normalized in {"0", "", "false", "no", "off"}:
        return None
    return f"StableHLO disabled via {_DISABLE_ENV}={raw}"


def detect_stablehlo_capability(env: Mapping[str, str] | None = None) -> StableHLOCapability:
    """Probe whether StableHLO dependencies are importable.

    Args:
        env: Optional environment mapping. Defaults to :data:`os.environ`.

    Returns:
        A :class:`StableHLOCapability` describing availability and failure reason.
    """

    env_map = os.environ if env is None else env
    disabled_reason = _is_env_disabled(env_map)
    if disabled_reason is not None:
        return StableHLOCapability(False, disabled_reason, {"stablehlo": False})

    spec = importlib.util.find_spec("stablehlo")
    if spec is None:
        return StableHLOCapability(
            False,
            "Python package 'stablehlo' is not importable. Install stablehlo>=0.14.",
            {"stablehlo": False},
        )

    try:
        module = importlib.import_module("stablehlo")
    except Exception as exc:  # pragma: no cover - defensive guardrail
        return StableHLOCapability(
            False,
            f"Importing 'stablehlo' failed: {exc}",
            {"stablehlo": True, "exception": repr(exc)},
        )

    details: dict[str, object] = {"stablehlo": True, "module": getattr(module, "__name__", "stablehlo")}
    try:
        version = metadata.version("stablehlo")
    except metadata.PackageNotFoundError:  # pragma: no cover - metadata edge case
        version = None
    except Exception:  # pragma: no cover - fallback for non-standard metadata
        version = None
    if version is not None:
        details["version"] = version

    return StableHLOCapability(True, None, details)


def require_stablehlo_capability(env: Mapping[str, str] | None = None) -> StableHLOCapability:
    """Ensure StableHLO dependencies are available, raising on failure."""

    capability = detect_stablehlo_capability(env=env)
    if not capability.available:
        reason = capability.reason or "StableHLO capability detection failed"
        raise RuntimeError(reason)
    return capability


@dataclass
class _BertModules:
    base: nn.Module
    embeddings: nn.Module | None
    classifier_modules: Iterable[nn.Module]


def _resolve_bert_modules(model: nn.Module) -> _BertModules:
    base = getattr(model, "bert", getattr(model, "base_model", model))
    embeddings = getattr(base, "embeddings", None)
    classifier_modules: list[nn.Module] = []

    for name in ("classifier", "qa_outputs", "cls"):
        mod = getattr(model, name, None)
        if mod is not None:
            classifier_modules.append(mod)

    return _BertModules(base=base, embeddings=embeddings, classifier_modules=classifier_modules)


def _collect_embedding_weight_ptrs(embeddings: nn.Module | None) -> set[int]:
    ptrs: set[int] = set()
    if embeddings is None:
        return ptrs

    for attr in ("word_embeddings", "position_embeddings", "token_type_embeddings"):
        emb = getattr(embeddings, attr, None)
        if emb is not None and hasattr(emb, "weight"):
            ptrs.add(emb.weight.data_ptr())  # type: ignore[attr-defined]

    if hasattr(embeddings, "LayerNorm"):
        ln = embeddings.LayerNorm
        if hasattr(ln, "weight"):
            ptrs.add(ln.weight.data_ptr())

    return ptrs


def _apply_pi_to_embedding(embeddings: nn.Module, pi: torch.LongTensor, pinv: torch.LongTensor) -> None:
    for attr in ("word_embeddings", "position_embeddings", "token_type_embeddings"):
        if hasattr(embeddings, attr):
            emb = getattr(embeddings, attr)
            if hasattr(emb, "weight"):
                emb.weight.data.copy_(reindex_cols(emb.weight.data, pi))
            if getattr(emb, "bias", None) is not None:
                emb.bias.data.copy_(reindex_vec(emb.bias.data, pi))

    ln = getattr(embeddings, "LayerNorm", None)
    if ln is not None:
        ln.weight.data.copy_(reindex_vec(ln.weight.data, pi))
        ln.bias.data.copy_(reindex_vec(ln.bias.data, pi))


def _apply_pi_to_bert_encoder(base: nn.Module, pi: torch.LongTensor, pinv: torch.LongTensor) -> None:
    encoder = getattr(base, "encoder", None)
    if encoder is None:
        return

    layers = getattr(encoder, "layer", [])
    for layer in layers:
        attention = layer.attention
        self_attn = attention.self

        head_dim = getattr(self_attn, "attention_head_size", None)
        num_heads = getattr(self_attn, "num_attention_heads", None)
        sigma: torch.LongTensor | None = None
        if isinstance(num_heads, int) and isinstance(head_dim, int):
            sigma, _ = make_head_block_sigma(pi, num_heads, head_dim)

        for proj_name in ("query", "key", "value"):
            proj = getattr(self_attn, proj_name)
            weight = Pinv_W_P(proj.weight.data, pi, pinv)
            if sigma is not None and sigma.numel() == weight.shape[0]:
                weight = reindex_rows(weight, sigma)
                proj.bias.data.copy_(reindex_vec(reindex_vec(proj.bias.data, pi), sigma))
            else:
                proj.bias.data.copy_(reindex_vec(proj.bias.data, pi))
            proj.weight.data.copy_(weight)

        out_proj = attention.output.dense
        weight = Pinv_W_P(out_proj.weight.data, pi, pinv)
        if sigma is not None and sigma.numel() == weight.shape[1]:
            weight = reindex_cols(weight, sigma)
        out_proj.weight.data.copy_(weight)
        out_proj.bias.data.copy_(reindex_vec(out_proj.bias.data, pi))

        attn_ln = attention.output.LayerNorm
        attn_ln.weight.data.copy_(reindex_vec(attn_ln.weight.data, pi))
        attn_ln.bias.data.copy_(reindex_vec(attn_ln.bias.data, pi))

        ffn_intermediate = layer.intermediate.dense
        ffn_output = layer.output.dense
        ffn_intermediate.weight.data.copy_(reindex_cols(ffn_intermediate.weight.data, pi))
        if ffn_intermediate.bias is not None:
            ffn_intermediate.bias.data = ffn_intermediate.bias.data.contiguous()
        ffn_output.weight.data.copy_(reindex_rows(ffn_output.weight.data, pi))
        ffn_output.bias.data.copy_(reindex_vec(ffn_output.bias.data, pi))

        output_ln = layer.output.LayerNorm
        output_ln.weight.data.copy_(reindex_vec(output_ln.weight.data, pi))
        output_ln.bias.data.copy_(reindex_vec(output_ln.bias.data, pi))

    pooler = getattr(base, "pooler", None)
    if pooler is not None and hasattr(pooler, "dense"):
        dense = pooler.dense
        dense.weight.data.copy_(Pinv_W_P(dense.weight.data, pi, pinv))
        if dense.bias is not None:
            dense.bias.data.copy_(reindex_vec(dense.bias.data, pi))


def _apply_pi_to_classifier(
    module: nn.Module,
    pi: torch.LongTensor,
    pinv: torch.LongTensor,
    skip_weight_ptrs: set[int] | None = None,
) -> None:
    if isinstance(module, nn.Linear):
        module.weight.data.copy_(reindex_cols(module.weight.data, pi))
        if module.bias is not None:
            module.bias.data = module.bias.data.contiguous()
        return

    if hasattr(module, "dense") and isinstance(module.dense, nn.Linear):
        dense = module.dense
        dense.weight.data.copy_(Pinv_W_P(dense.weight.data, pi, pinv))
        if dense.bias is not None:
            dense.bias.data.copy_(reindex_vec(dense.bias.data, pi))

    if hasattr(module, "decoder") and isinstance(module.decoder, nn.Linear):
        decoder = module.decoder
        if skip_weight_ptrs is None or decoder.weight.data_ptr() not in skip_weight_ptrs:
            decoder.weight.data.copy_(reindex_cols(decoder.weight.data, pi))
        if decoder.bias is not None:
            decoder.bias.data = decoder.bias.data.contiguous()

    if hasattr(module, "LayerNorm") and isinstance(module.LayerNorm, nn.LayerNorm):
        ln = module.LayerNorm
        ln.weight.data.copy_(reindex_vec(ln.weight.data, pi))
        ln.bias.data.copy_(reindex_vec(ln.bias.data, pi))

    if hasattr(module, "predictions"):
        _apply_pi_to_classifier(module.predictions, pi, pinv, skip_weight_ptrs)

    if hasattr(module, "transform"):
        _apply_pi_to_classifier(module.transform, pi, pinv, skip_weight_ptrs)


def apply_pi_to_bert(model: nn.Module, pi: torch.LongTensor) -> None:
    """In-place reindexing for all hidden-dimension tensors in BERT."""

    config = getattr(model, "config", None)
    if config is None or not hasattr(config, "hidden_size"):
        raise ValueError("model does not expose hidden_size in config")
    device = next(model.parameters()).device
    pi = _validate_perm(pi.to(device=device, dtype=torch.long), config.hidden_size)
    pinv = inv_perm(pi)

    modules = _resolve_bert_modules(model)
    skip_weight_ptrs = _collect_embedding_weight_ptrs(modules.embeddings)

    if modules.embeddings is not None:
        _apply_pi_to_embedding(modules.embeddings, pi, pinv)

    _apply_pi_to_bert_encoder(modules.base, pi, pinv)

    for classifier in modules.classifier_modules:
        _apply_pi_to_classifier(
            classifier,
            pi,
            pinv,
            skip_weight_ptrs if skip_weight_ptrs else None,
        )

    if hasattr(modules.base, "set_attn_implementation"):
        # Leave implementation unchanged to avoid altering outputs.
        pass

    try:
        from torch.fx._symbolic_trace import is_fx_tracing
    except ImportError:  # pragma: no cover - FX optional
        is_fx_tracing = None  # type: ignore[assignment]

    if is_fx_tracing is not None:
        for layer in getattr(getattr(modules.base, "encoder", None), "layer", []):
            for submod in layer.modules():
                if getattr(submod, "_pi_fx_guarded", False) or not hasattr(submod, "forward"):
                    continue
                submod.forward = _FXForwardGuard(submod.forward)  # type: ignore[assignment]
                submod._pi_fx_guarded = True

    signature = perm_signature(pi)
    model.config.pi_applied = True
    model.config.pi = pi.detach().cpu().tolist()
    model.config.pi_signature = signature


def apply_pi_to_mamba(module_dict: Mapping[str, Any], pi: torch.LongTensor) -> None:
    required = {"A", "B", "C"}
    missing = required.difference(module_dict.keys())
    if missing:
        raise ValueError(f"missing required Mamba modules: {sorted(missing)}")

    A = module_dict["A"]
    weight = getattr(A, "weight", None)
    if weight is None:
        raise ValueError("module 'A' must expose a weight parameter")

    hidden_dim = getattr(weight, "shape", [None])[0]
    if hidden_dim is None:
        raise ValueError("unable to infer hidden dimension from module 'A'")

    device = weight.device
    pi = _validate_perm(pi.to(device=device, dtype=torch.long), int(hidden_dim))
    pinv = inv_perm(pi)

    runtime_weight = Pinv_W_P(A.weight.data, pi, pinv)
    A.weight.data.copy_(PWP_inv(A.weight.data, pi, pinv))
    if getattr(A, "bias", None) is not None:
        A.bias.data.copy_(reindex_vec(A.bias.data, pi))

    if isinstance(A, torch.nn.Module):
        def _mamba_linear_hook(module: torch.nn.Module, inputs, output):
            (x,) = inputs
            return torch.nn.functional.linear(x, runtime_weight, module.bias)

        for hook in getattr(A, "_pi_runtime_hooks", () ):
            hook.remove()
        hook = A.register_forward_hook(_mamba_linear_hook)
        A._pi_runtime_hooks = (hook,)

    B = module_dict["B"]
    B.weight.data.copy_(reindex_rows(B.weight.data, pi))
    if getattr(B, "bias", None) is not None:
        B.bias.data.copy_(reindex_vec(B.bias.data, pi))

    C = module_dict["C"]
    C.weight.data.copy_(reindex_cols(C.weight.data, pi))
    if getattr(C, "bias", None) is not None:
        C.bias.data = C.bias.data.contiguous()

    x0 = module_dict.get("x0")
    if isinstance(x0, torch.nn.Parameter):
        x0.data.copy_(reindex_vec(x0.data, pi))
    elif isinstance(x0, torch.Tensor):
        module_dict["x0"] = reindex_vec(x0, pi)


def _contains_proxy(obj) -> bool:
    if isinstance(obj, Proxy):
        return True
    if isinstance(obj, (tuple, list, set)):
        return any(_contains_proxy(v) for v in obj)
    if isinstance(obj, Mapping):
        return any(_contains_proxy(v) for v in obj.values())
    return False


class _FXForwardGuard:
    __slots__ = ("_orig_forward",)

    def __init__(self, orig_forward):
        self._orig_forward = orig_forward

    def __call__(self, *args, **kwargs):
        try:
            from torch.fx._symbolic_trace import is_fx_tracing
        except ImportError:  # pragma: no cover
            is_fx_tracing = None  # type: ignore[assignment]

        if (is_fx_tracing is not None and is_fx_tracing()) or _contains_proxy(args) or _contains_proxy(kwargs):
            raise RuntimeError("FX tracing disabled after permutation application")
        return self._orig_forward(*args, **kwargs)
