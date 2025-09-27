"""Export helpers for converting NMLinear weights to serialized payloads."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping

import torch

from icd.hds.layers import NMLinear

__all__ = ["SparseLayerExport", "collect_sparse_layers", "export_sparse_model"]


@dataclass
class SparseLayerExport:
    name: str
    weight: torch.Tensor
    bias: torch.Tensor | None
    mask: torch.Tensor
    metadata: Mapping[str, object]

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "name": self.name,
            "weight": self.weight.detach().cpu(),
            "mask": self.mask.detach().cpu(),
            "metadata": dict(self.metadata),
        }
        if self.bias is not None:
            payload["bias"] = self.bias.detach().cpu()
        return payload


def collect_sparse_layers(model: torch.nn.Module) -> List[SparseLayerExport]:
    exports: List[SparseLayerExport] = []
    for name, module in model.named_modules():
        if isinstance(module, NMLinear):
            mask = module.masker.last_mask().view(module.out_features, module.in_features)
            weight = module.linear.weight.detach()
            bias = module.linear.bias.detach() if module.linear.bias is not None else None
            exports.append(
                SparseLayerExport(
                    name=name,
                    weight=weight * mask,
                    bias=bias,
                    mask=mask,
                    metadata={
                        "group_size": module.masker.config.group_size,
                        "active": module.masker.config.active,
                    },
                )
            )
    return exports


def export_sparse_model(model: torch.nn.Module, output: str | Path, *, metadata: Mapping[str, object] | None = None) -> Dict[str, object]:
    exports = collect_sparse_layers(model)
    payload = {
        "layers": [layer.to_dict() for layer in exports],
        "meta": dict(metadata or {}),
        "version": 1,
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return payload

