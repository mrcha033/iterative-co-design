"""Phase-1 sparsity utilities (unstructured pruning)."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch.nn import Linear
from torch.nn.utils import prune

try:  # pragma: no cover - HDS components are optional during bootstrap
    from icd.hds.layers import NMLinear  # type: ignore
    from icd.hds.topk import TopKMaskerConfig  # type: ignore
except Exception:  # pragma: no cover - fallback when HDS isnt available
    NMLinear = None  # type: ignore[assignment]
    TopKMaskerConfig = None  # type: ignore[assignment]

__all__ = ["SparsityConfig", "iter_prunable_linears", "apply_unstructured"]


@dataclass
class SparsityConfig:
    type: str = "none"
    amount: float = 0.5
    scope: str = "global"  # or "per_layer"
    apply_to: Sequence[str] | None = None
    exclude: Sequence[str] | None = None

    @classmethod
    def from_dict(cls, data: dict | None) -> "SparsityConfig":
        if not data:
            return cls()
        return cls(
            type=str(data.get("type", "none")).lower(),
            amount=float(data.get("amount", 0.5)),
            scope=str(data.get("scope", "global")).lower(),
            apply_to=data.get("apply_to"),
            exclude=data.get("exclude"),
        )


_DEFAULT_APPLY_TO = ("bert", "encoder", "attention", "dense", "linear", "classifier")
_DEFAULT_EXCLUDE = ("embedding", "layernorm", "lm_head", "pooler")


def iter_prunable_linears(
    model: torch.nn.Module,
    *,
    apply_to: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
) -> Iterable[tuple[str, Linear]]:
    targets = tuple(s.lower() for s in (apply_to or _DEFAULT_APPLY_TO))
    blockers = tuple(s.lower() for s in (exclude or _DEFAULT_EXCLUDE))
    for name, module in model.named_modules():
        if not isinstance(module, Linear):
            continue
        lname = name.lower()
        if targets and not any(token in lname for token in targets):
            continue
        if blockers and any(token in lname for token in blockers):
            continue
        yield name, module


def _global_threshold(weights: list[torch.Tensor], amount: float) -> float:
    if not weights or amount <= 0.0:
        return float("inf")
    flat = torch.cat([w.reshape(-1).abs() for w in weights])
    k = int(flat.numel() * amount)
    if k <= 0:
        return -1.0
    if k >= flat.numel():
        return flat.max().item()
    values, _ = torch.topk(flat, k, largest=False)
    return values.max().item()


def apply_unstructured(
    model: torch.nn.Module,
    *,
    amount: float = 0.5,
    scope: str = "global",
    apply_to: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
) -> None:
    if amount <= 0.0:
        return
    scope = scope.lower()
    modules = list(iter_prunable_linears(model, apply_to=apply_to, exclude=exclude))
    if not modules:
        warnings.warn("no prunable linear layers found for sparsity", RuntimeWarning)
        return

    if scope == "global":
        threshold = _global_threshold([lin.weight.detach() for _, lin in modules], amount)
        for _, lin in modules:
            mask = (lin.weight.detach().abs() > threshold)
            prune.custom_from_mask(lin, name="weight", mask=mask)
            prune.remove(lin, "weight")
    else:
        for _, lin in modules:
            prune.l1_unstructured(lin, name="weight", amount=amount)
            prune.remove(lin, "weight")

    for _, lin in modules:
        lin.weight.data = lin.weight.data.contiguous()


def apply_sparsity_from_config(model: torch.nn.Module, cfg: SparsityConfig) -> None:
    if cfg.type == "none":
        return
    if cfg.type not in {"unstructured"}:
        raise ValueError(f"unsupported sparsity type '{cfg.type}' in Phase-1")
    apply_unstructured(
        model,
        amount=cfg.amount,
        scope=cfg.scope,
        apply_to=cfg.apply_to,
        exclude=cfg.exclude,
    )

# -----------------------
# Public sparsity wrapper
# -----------------------
from typing import Iterable, Optional, Sequence, Tuple
from torch import nn


def _default_prunable_modules(model: nn.Module) -> Sequence[nn.Module]:
    return [m for m in model.modules() if isinstance(m, nn.Linear)]


def _normalize_sparsity_kwargs(
    *,
    type: str | None,
    rate: Optional[float],
    amount: Optional[float],
    sparsity: Optional[float],
) -> tuple[str, float]:
    typ = (type or "unstructured").lower()
    effective = rate
    if effective is None:
        effective = amount if amount is not None else sparsity
    if effective is None:
        effective = 0.0
    return typ, float(effective)


def apply_sparsity(
    model: nn.Module | None,
    *,
    type: str | None = None,
    rate: Optional[float] = None,
    amount: Optional[float] = None,
    sparsity: Optional[float] = None,
    modules: Optional[Iterable[nn.Module]] = None,
    param_name: str = "weight",
    method: str = "l1_unstructured",
    global_unstructured: bool = True,
    remove_reparam: bool = True,
) -> tuple[nn.Module | None, dict[str, object]]:
    """Apply simple sparsity and return ``(model, meta)`` for adapters/tests."""

    typ, effective = _normalize_sparsity_kwargs(type=type, rate=rate, amount=amount, sparsity=sparsity)
    meta: dict[str, object] = {
        "delta_layout": bool(effective > 0.0 and typ not in {"none", "identity"}),
        "sparsity": {
            "type": typ,
            "rate": effective,
            "method": method,
        },
    }

    if model is None or effective <= 0.0 or typ in {"none", "identity"}:
        return model, meta

    target_modules = list(modules) if modules is not None else list(_default_prunable_modules(model))

    if typ in {"2:4", "structured", "nm", "hardware"}:
        converted = _apply_structured_nm(model, target_modules)
        meta["sparsity"]["method"] = "nm_linear"
        meta["sparsity"]["group"] = {"active": 2, "group": 4}
        meta["sparsity"]["converted"] = converted
        meta["delta_layout"] = bool(converted > 0)
        if converted == 0:
            warnings.warn("structured sparsity requested but no linear layers were converted", RuntimeWarning)
        return model, meta

    params_to_prune: list[Tuple[nn.Module, str]] = [(m, param_name) for m in target_modules if hasattr(m, param_name)]
    if not params_to_prune:
        warnings.warn("no prunable modules found; sparsity skipped", RuntimeWarning)
        return model, meta

    prune_method = method.lower()
    if prune_method == "random_unstructured":
        pruner_cls = prune.RandomUnstructured
    else:
        pruner_cls = prune.L1Unstructured

    if global_unstructured and len(params_to_prune) > 1 and pruner_cls is prune.L1Unstructured:
        prune.global_unstructured(params_to_prune, pruning_method=pruner_cls, amount=float(effective))
    else:
        for module, pname in params_to_prune:
            if pruner_cls is prune.RandomUnstructured:
                prune.random_unstructured(module, name=pname, amount=float(effective))
            else:
                prune.l1_unstructured(module, name=pname, amount=float(effective))

    if remove_reparam:
        for module, pname in params_to_prune:
            try:
                prune.remove(module, pname)
            except Exception:
                pass

    return model, meta


if "apply_sparsity" not in __all__:
    __all__.append("apply_sparsity")


def _apply_structured_nm(model: nn.Module, target_modules: Iterable[nn.Module]) -> int:
    if NMLinear is None or TopKMaskerConfig is None:
        warnings.warn("NMLinear not available; skipping structured sparsity", RuntimeWarning)
        return 0

    converted = 0
    seen: set[int] = set()
    for module in target_modules:
        if not isinstance(module, nn.Linear):
            continue
        if id(module) in seen:
            continue
        parent, name = _find_parent_module(model, module)
        if parent is None or name is None:
            continue
        nm = NMLinear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            n_active=2,
            m_group=4,
            masker_config=TopKMaskerConfig(active=2, group_size=4),
        )
        nm.load_from_linear(module)
        nm.to(module.weight.device)
        setattr(parent, name, nm)
        converted += 1
        seen.add(id(module))
    return converted


def _find_parent_module(root: nn.Module, child: nn.Module) -> tuple[nn.Module | None, str | None]:
    for parent in root.modules():
        for name, module in parent.named_children():
            if module is child:
                return parent, name
    return None, None
