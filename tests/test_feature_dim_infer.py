import types

import torch
import torch.nn as nn
from torch import fx

from icd.core.graph_pytorch import (
    _infer_feature_dim_from_fx,
    _maybe_override_feature_dim_from_config,
    build_w_from_pytorch,
)


class ToyModel(nn.Module):
    def __init__(self, hidden: int = 64, intermediate: int = 128):
        super().__init__()
        self.l1 = nn.Linear(hidden, intermediate)
        self.l2 = nn.Linear(intermediate, hidden)
        self.conv = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.config = types.SimpleNamespace(
            model_type="mamba2",
            hidden_size=hidden,
            intermediate_size=intermediate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.l1(x)
        y = torch.relu(y)
        y = self.l2(y)
        z = x.transpose(1, 2)
        z = self.conv(z)
        z = z.transpose(1, 2)
        return y + z


class TinyLM(nn.Module):
    def __init__(self, d: int = 96, vocab: int = 50000, heads: int = 12):
        super().__init__()
        self.l1 = nn.Linear(d, 4 * d)
        self.l2 = nn.Linear(4 * d, d)
        self.lm_head = nn.Linear(d, vocab)
        self.config = types.SimpleNamespace(
            model_type="mamba2",
            hidden_size=d,
            intermediate_size=4 * d,
            num_attention_heads=heads,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.l2(torch.relu(self.l1(x)))
        return self.lm_head(h)


def _shape_prop(gm: fx.GraphModule, *example_inputs: torch.Tensor) -> None:
    from torch.fx.passes.shape_prop import ShapeProp

    ShapeProp(gm).propagate(*example_inputs)


@torch.no_grad()
def test_fx_prefers_linear_over_conv_lastdim():
    hidden = 64
    model = ToyModel(hidden=hidden, intermediate=hidden * 2).eval()
    example = torch.randn(2, 4096, hidden)
    gm = fx.symbolic_trace(model)
    _shape_prop(gm, example)
    inferred = _infer_feature_dim_from_fx(gm, fallback=0)
    assert inferred == hidden


def test_config_override_prefers_hidden_size_for_mamba_family():
    current = 4096
    hidden = 96
    model = ToyModel(hidden=hidden, intermediate=hidden * 2)
    model.config = types.SimpleNamespace(model_type="mamba2", hidden_size=hidden)
    override, source = _maybe_override_feature_dim_from_config(model, current)
    assert override == hidden
    assert source == "hf_config.hidden_size"


def test_config_override_uses_d_model_when_hidden_missing():
    current = 0
    d_model = 128
    model = ToyModel(hidden=d_model, intermediate=d_model * 2)
    model.config = types.SimpleNamespace(model_type="ssm", d_model=d_model)
    override, source = _maybe_override_feature_dim_from_config(model, current)
    assert override == d_model
    assert source == "hf_config.d_model"


@torch.no_grad()
def test_build_w_uses_config_for_attention_meta() -> None:
    model = TinyLM(d=96, vocab=50000, heads=12).eval()
    example = torch.randn(1, 32, 96)
    W = build_w_from_pytorch(model, example, seed=0)

    meta = W.meta["pytorch"]
    assert meta["feature_dim"] == 96
    assert meta["feature_dim_source"] in {"hf_config.hidden_size", "fx", "tensor"}

    inter = meta.get("intermediate", {})
    assert inter["size"] == 384
    assert inter["expansion_factor"] == 4

    att = meta.get("attention", {})
    assert att["enabled"] is True
    assert att["head_dim"] == 8
    assert att["num_heads"] == 12
