import types

import torch
import torch.nn as nn

from icd.core.graph_pytorch import build_w_from_pytorch


class TinyMamba(nn.Module):
    def __init__(self, d: int = 2560, inter: int = 5120, heads: int = 20):
        super().__init__()
        self.l1 = nn.Linear(d, inter)
        self.l2 = nn.Linear(inter, d)
        self.config = types.SimpleNamespace(
            hidden_size=d,
            intermediate_size=inter,
            num_attention_heads=heads,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l2(torch.relu(self.l1(x)))


def test_hidden_size_wins_over_seq_len():
    m = TinyMamba().eval()
    x = torch.randn(1, 4096, 2560)

    W = build_w_from_pytorch(m, x, seed=0)
    meta = W.meta["pytorch"]

    assert W.shape == (2560, 2560)
    assert meta["feature_dim"] == 2560
    assert meta["feature_dim_source"] in {
        "hf_config.hidden_size",
        "hf_config.hidden_size(seq_len_disambiguation)",
    }
    assert meta["intermediate"]["expansion_factor"] == 2
