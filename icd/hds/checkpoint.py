"""Utilities for persisting and restoring NMLinear mask state."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping

import torch

from .layers import NMLinear
MASK_STATE_VERSION = 1

__all__ = [
    "MASK_STATE_VERSION",
    "MaskState",
    "serialize_mask_state",
    "deserialize_mask_state",
    "save_mask_state",
    "load_mask_state",
]


@dataclass
class MaskState:
    """Serializable mask snapshot for a single :class:`NMLinear` layer."""

    path: str
    logits: torch.Tensor
    step: int
    config: Mapping[str, object]

    def to_payload(self) -> Dict[str, object]:
        return {
            "path": self.path,
            "logits": self.logits.detach().cpu().tolist(),
            "step": int(self.step),
            "config": dict(self.config),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "MaskState":
        logits = torch.tensor(payload.get("logits", []), dtype=torch.float32)
        return cls(
            path=str(payload.get("path", "")),
            logits=logits,
            step=int(payload.get("step", 0)),
            config=dict(payload.get("config", {})),
        )


def _state_for_module(path: str, module: NMLinear) -> MaskState:
    cfg = module.masker.config
    config_payload = {
        "group_size": cfg.group_size,
        "active": cfg.active,
        "temperature_init": cfg.temperature_init,
        "temperature_final": cfg.temperature_final,
        "anneal_steps": cfg.anneal_steps,
        "seed": cfg.seed,
    }
    return MaskState(
        path=path,
        logits=module.masker.logits.detach().cpu(),
        step=int(module.masker._step.item()),
        config=config_payload,
    )


def serialize_mask_state(model: torch.nn.Module) -> Dict[str, object]:
    """Capture mask state for all NMLinear modules in ``model``."""

    payloads: List[Dict[str, object]] = []
    for name, module in model.named_modules():
        if isinstance(module, NMLinear):
            payloads.append(_state_for_module(name or "<root>", module).to_payload())
    return {
        "version": MASK_STATE_VERSION,
        "modules": payloads,
    }


def deserialize_mask_state(model: torch.nn.Module, state: Mapping[str, object]) -> None:
    """Restore mask state captured by :func:`serialize_mask_state`."""

    modules_by_name: Dict[str, NMLinear] = {
        name: module for name, module in model.named_modules() if isinstance(module, NMLinear)
    }
    for entry in state.get("modules", []):
        mask_state = MaskState.from_payload(entry)
        module = modules_by_name.get(mask_state.path)
        if module is None:
            continue
        logits = mask_state.logits.to(module.masker.logits.device)
        if logits.shape != module.masker.logits.shape:
            raise ValueError(
                f"mask logits shape mismatch for '{mask_state.path}': "
                f"expected {tuple(module.masker.logits.shape)}, got {tuple(logits.shape)}"
            )
        module.masker.logits.data.copy_(logits)
        module.masker._step.fill_(int(mask_state.step))


def save_mask_state(path: str | Path, model: torch.nn.Module) -> None:
    """Write serialized mask state to ``path`` as JSON."""

    import json

    payload = serialize_mask_state(model)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def load_mask_state(path: str | Path, model: torch.nn.Module) -> Dict[str, object]:
    """Load mask state from ``path`` and apply to ``model``."""

    import json

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    deserialize_mask_state(model, payload)
    return payload

