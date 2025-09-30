import types

import torch
import torch.nn as nn

from icd.core.graph_pytorch import build_w_from_pytorch


class TinyLM(nn.Module):
    def __init__(self, d=96, vocab=50000, heads=12):
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

    def forward(self, x):
        h = self.l2(torch.relu(self.l1(x)))
        return self.lm_head(h)


def test_feature_dim_detect_and_meta():
    torch.manual_seed(0)
    m = TinyLM().eval()
    x = torch.randn(2, 4096, 96)
    W = build_w_from_pytorch(m, x, seed=0)
    assert W.shape == (96, 96)
    meta = W.meta["pytorch"]
    assert meta["feature_dim"] == 96
    assert meta["feature_dim_source"] in {"hf_config.hidden_size", "fx", "tensor", "default"}
    assert meta["intermediate"]["expansion_factor"] == 4
    att = meta.get("attention", {})
    assert att.get("head_dim") == 8
    assert att.get("section_size") == 8


def test_profile_keyword_matching_is_case_insensitive():
    m = TinyLM().eval()
    x = torch.randn(1, 32, 96)
    W = build_w_from_pytorch(m, x, seed=1, attention_aware=False, sectioning=False)
    assert len(W.data) == W.meta["pytorch"]["nnz"] > 0
