"""Training utilities for Hardware-native Differentiable Sparsity (HDS).

These helpers provide a minimal, configurable training loop that focuses on
mask parameters in :class:`~icd.hds.layers.NMLinear` modules.  The loop is
intended to be orchestration-friendly: it exposes the effective mask update
step, supports deterministic temperature schedules for Gumbel-TopK sampling,
and stages optimizer state so higher layers can restore optimizers when using
straight-through estimators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, MutableMapping, Optional, Sequence
import copy

import torch

from icd.utils.imports import load_object

try:  # pragma: no cover - optional import guarding bootstrap
    from .layers import NMLinear  # type: ignore
    from .topk import TopKMasker
except Exception:  # pragma: no cover
    NMLinear = None  # type: ignore[assignment]
    TopKMasker = None  # type: ignore[assignment]

__all__ = ["MaskTrainingConfig", "run_mask_training", "iter_masked_modules"]


@dataclass
class MaskTrainingConfig:
    """Configuration for the sparse mask training loop."""

    steps: int = 0
    sample: bool = True
    temperature_init: float = 1.0
    temperature_final: float = 0.1
    anneal_steps: Optional[int] = None
    optimizer: str = "adam"
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: Optional[float] = None
    seed: Optional[int] = None
    reset_step: bool = True
    stash_optimizer: bool = True
    state_key: str = "hds.optimizer"
    stepper: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Mapping[str, object] | None) -> "MaskTrainingConfig":
        if not data:
            return cls()
        return cls(
            steps=int(data.get("steps", 0)),
            sample=bool(data.get("sample", True)),
            temperature_init=float(data.get("temperature_init", 1.0)),
            temperature_final=float(data.get("temperature_final", 0.1)),
            anneal_steps=(
                int(data["anneal_steps"]) if data.get("anneal_steps") is not None else None
            ),
            optimizer=str(data.get("optimizer", "adam")).lower(),
            lr=float(data.get("lr", data.get("learning_rate", 1e-3))),
            weight_decay=float(data.get("weight_decay", 0.0)),
            grad_clip=(
                float(data["grad_clip"]) if data.get("grad_clip") is not None else None
            ),
            seed=(int(data["seed"]) if data.get("seed") is not None else None),
            reset_step=bool(data.get("reset_step", True)),
            stash_optimizer=bool(data.get("stash_optimizer", True)),
            state_key=str(data.get("state_key", "hds.optimizer")),
            stepper=(str(data["stepper"]) if data.get("stepper") else None),
        )

    def anneal_span(self) -> int:
        if self.anneal_steps is not None:
            return max(0, int(self.anneal_steps))
        return max(0, int(self.steps))

    def temperature_for_step(self, step: int) -> float:
        if self.anneal_span() <= 0:
            return float(self.temperature_final)
        denom = max(1, self.anneal_span() - 1)
        frac = min(1.0, max(0.0, float(step) / float(denom)))
        start = float(self.temperature_init)
        end = float(self.temperature_final)
        return start + frac * (end - start)

    def resolve_stepper(self) -> Callable[..., torch.Tensor] | None:
        if not self.stepper:
            return None
        stepper = load_object(self.stepper)
        if not callable(stepper):  # pragma: no cover - defensive
            raise TypeError(f"configured stepper '{self.stepper}' is not callable")
        return stepper  # type: ignore[return-value]


def iter_masked_modules(model: torch.nn.Module) -> Iterable[NMLinear]:
    if NMLinear is None:
        return []
    return (m for m in model.modules() if isinstance(m, NMLinear))


def _configure_masker(masker: TopKMasker, cfg: MaskTrainingConfig) -> None:
    if cfg.anneal_steps is not None:
        masker.config.anneal_steps = int(cfg.anneal_steps)
    else:
        masker.config.anneal_steps = max(cfg.steps, 0)
    masker.config.temperature_init = float(cfg.temperature_init)
    masker.config.temperature_final = float(cfg.temperature_final)
    if cfg.seed is not None:
        masker.config.seed = int(cfg.seed)


def _make_optimizer(cfg: MaskTrainingConfig, params: Iterable[torch.nn.Parameter]) -> torch.optim.Optimizer:
    params = list(params)
    opt = cfg.optimizer
    if opt == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if opt == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if opt not in {"adam", "adamw", "sgd"}:
        raise ValueError(f"unsupported optimizer '{opt}' for mask training")
    return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)


def _default_stepper(
    *,
    masks: Sequence[torch.Tensor],
    **_: object,
) -> torch.Tensor:
    loss = torch.zeros((), device=masks[0].device if masks else "cpu")
    for mask in masks:
        loss = loss + mask.pow(2).mean()
    return loss


def run_mask_training(
    model: torch.nn.Module,
    cfg: MaskTrainingConfig,
    *,
    context: MutableMapping[str, object] | None = None,
) -> dict[str, object]:
    """Run a mask-only training loop for :class:`NMLinear` modules.

    Parameters
    ----------
    model:
        Module containing one or more :class:`NMLinear` layers.
    cfg:
        Training configuration specifying steps, temperature schedule and
        optimizer settings.
    context:
        Optional mutable mapping used to stash optimizer state (before/after)
        so that orchestrators can restore the optimizer around STE updates.
    """

    if NMLinear is None or TopKMasker is None or cfg.steps <= 0:
        return {
            "steps": 0,
            "temperature": {
                "init": cfg.temperature_init,
                "final": cfg.temperature_final,
                "anneal_steps": cfg.anneal_span(),
                "values": [],
            },
            "optimizer": {
                "type": cfg.optimizer,
                "lr": cfg.lr,
                "weight_decay": cfg.weight_decay,
                "state_stashed": False,
                "state_key": cfg.state_key,
            },
        }

    modules = list(iter_masked_modules(model))
    if not modules:
        return {
            "steps": 0,
            "temperature": {
                "init": cfg.temperature_init,
                "final": cfg.temperature_final,
                "anneal_steps": cfg.anneal_span(),
                "values": [],
            },
            "optimizer": {
                "type": cfg.optimizer,
                "lr": cfg.lr,
                "weight_decay": cfg.weight_decay,
                "state_stashed": False,
                "state_key": cfg.state_key,
            },
        }

    params: list[torch.nn.Parameter] = []
    original_modes: list[bool] = []
    for module in modules:
        masker = module.masker
        _configure_masker(masker, cfg)
        original_modes.append(masker.training)
        masker.train()
        params.append(masker.logits)
        if cfg.reset_step:
            masker.reset_step()

    optimizer = _make_optimizer(cfg, params)
    if cfg.stash_optimizer:
        state_before = copy.deepcopy(optimizer.state_dict())
        if context is not None:
            context[f"{cfg.state_key}::before"] = state_before
    else:
        state_before = None

    stepper = cfg.resolve_stepper() or _default_stepper
    temps: list[float] = []
    losses: list[float] = []

    for step in range(int(cfg.steps)):
        temperature = cfg.temperature_for_step(step)
        temps.append(float(temperature))
        optimizer.zero_grad(set_to_none=True)
        masks: list[torch.Tensor] = []
        for module in modules:
            mask = module.masker(step=None, sample=cfg.sample, temperature=temperature)
            masks.append(mask)
        loss = stepper(model=model, step=step, temperature=temperature, masks=masks, context=context)
        if not isinstance(loss, torch.Tensor):
            raise TypeError("mask training stepper must return a torch.Tensor loss")
        loss.backward()
        if cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))

    if cfg.stash_optimizer:
        state_after = copy.deepcopy(optimizer.state_dict())
        if context is not None:
            context[f"{cfg.state_key}::after"] = state_after
    else:
        state_after = None

    for masker, was_training in zip((m.masker for m in modules), original_modes):
        if not was_training:
            masker.eval()

    return {
        "steps": int(cfg.steps),
        "temperature": {
            "init": cfg.temperature_init,
            "final": cfg.temperature_final,
            "anneal_steps": cfg.anneal_span(),
            "values": temps,
        },
        "optimizer": {
            "type": cfg.optimizer,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "state_stashed": cfg.stash_optimizer,
            "state_key": cfg.state_key,
            "restorable": state_before is not None,
        },
        "loss": {
            "history": losses,
            "final": losses[-1] if losses else None,
        },
    }

