import types

import torch
import torch.nn as nn
from torch import fx

from icd.core.graph_pytorch import (
    _infer_feature_dim_from_fx,
    _infer_feature_dim_from_tensor,
)


class TinyLM(nn.Module):
    def __init__(self, d: int = 64, vocab: int = 50000):
        super().__init__()
        self.l1 = nn.Linear(d, 4 * d)
        self.l2 = nn.Linear(4 * d, d)
        self.lm_head = nn.Linear(d, vocab)
        self.config = types.SimpleNamespace(
            model_type="mamba2", hidden_size=d, intermediate_size=4 * d
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.l2(torch.relu(self.l1(x)))
        logits = self.lm_head(h)
        return logits


def _shape_prop(gm: fx.GraphModule, *inputs: torch.Tensor) -> None:
    from torch.fx.passes.shape_prop import ShapeProp

    ShapeProp(gm).propagate(*inputs)


def test_fx_ignores_vocab_pollution() -> None:
    m = TinyLM(d=96, vocab=50000).eval()
    gm = fx.symbolic_trace(m)
    ex = torch.randn(2, 4096, 96)
    _shape_prop(gm, ex)
    D = _infer_feature_dim_from_fx(gm, fallback=ex.shape[-1])
    assert D == 96


def test_tensor_fallback_2d_required() -> None:
    x = torch.randn(8)
    assert _infer_feature_dim_from_tensor(x) == 0


def test_fx_prefers_fallback_on_tie() -> None:
    m = TinyLM(d=128, vocab=4096).eval()
    gm = fx.symbolic_trace(m)
    ex = torch.randn(1, 4096, 128)
    _shape_prop(gm, ex)
    D = _infer_feature_dim_from_fx(gm, fallback=128)
    assert D == 128
