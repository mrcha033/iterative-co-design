from __future__ import annotations

import pytest
import torch
from types import SimpleNamespace

from icd.runtime.apply_pi import (
    PWP_inv,
    apply_pi_to_bert,
    apply_pi_to_mamba,
    inv_perm,
    perm_signature,
    reindex_vec,
)
from icd.adapters.quant import QuantConfig
from icd.runtime.runners_hf import _apply_pi_sequence


def test_inv_perm_roundtrip():
    pi = torch.tensor([2, 0, 1, 3, 4], dtype=torch.long)
    pinv = inv_perm(pi)
    assert torch.equal(pi[pinv], torch.arange(pi.numel()))
    assert torch.equal(pinv[pi], torch.arange(pi.numel()))


def test_pwp_inv_equivalence():
    torch.manual_seed(0)
    H = 16
    pi = torch.randperm(H)
    pinv = inv_perm(pi)

    W = torch.randn(H, H)
    b = torch.randn(H)
    x = torch.randn(3, H)

    y = x @ W.t() + b

    Wp = PWP_inv(W, pi, pinv)
    bp = reindex_vec(b, pi)
    xp = x.index_select(1, pinv)
    yp = xp @ Wp.t() + bp
    y_back = yp.index_select(1, pinv)
    assert torch.allclose(y, y_back, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("seq_len", [8])
def test_bert_pi_equivalence_and_no_runtime_permute(seq_len):
    pytest.importorskip("transformers")
    from transformers import BertConfig, BertForSequenceClassification

    torch.manual_seed(42)
    cfg = BertConfig(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_hidden_layers=2,
        vocab_size=128,
    )
    model = BertForSequenceClassification(cfg)
    model.eval()

    input_ids = torch.randint(0, cfg.vocab_size, (2, seq_len))
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        logits_before = model(input_ids=input_ids, attention_mask=attention_mask).logits

    pi = torch.randperm(cfg.hidden_size)
    apply_pi_to_bert(model, pi)

    with torch.no_grad():
        logits_after = model(input_ids=input_ids, attention_mask=attention_mask).logits

    assert torch.allclose(logits_before, logits_after, atol=1e-5, rtol=1e-5)

    try:
        import torch.fx as fx

        layer = model.bert.encoder.layer[0]
        layer.eval()
        gm = fx.symbolic_trace(
            layer,
            concrete_args={
                "attention_mask": None,
                "encoder_hidden_states": None,
                "encoder_attention_mask": None,
                "past_key_value": None,
                "output_attentions": False,
            },
        )
    except Exception as exc:  # pragma: no cover - tracing is optional
        pytest.skip(f"symbolic tracing unavailable: {exc}")

    banned_ops = {
        "index_select",
        "permute",
        "gather",
        "transpose",
        "narrow",
        "slice",
        "select",
        "view",
        "reshape",
        "cat",
        "stack",
    }
    offending = [node for node in gm.graph.nodes if any(op in str(node.target) for op in banned_ops)]
    assert not offending, f"unexpected runtime permutation nodes: {[str(node.target) for node in offending]}"


@pytest.mark.parametrize("with_pooler", [True, False])
def test_bert_pooler_equivalence(with_pooler):
    pytest.importorskip("transformers")
    from transformers import BertConfig, BertModel

    torch.manual_seed(7)
    cfg = BertConfig(
        hidden_size=24,
        intermediate_size=48,
        num_attention_heads=4,
        num_hidden_layers=1,
        vocab_size=64,
        add_pooling_layer=with_pooler,
    )
    model = BertModel(cfg)
    model.eval()

    inputs = torch.randint(0, cfg.vocab_size, (3, 12))
    attention_mask = torch.ones_like(inputs)

    with torch.no_grad():
        pooled0 = model(input_ids=inputs, attention_mask=attention_mask).pooler_output if with_pooler else model(input_ids=inputs, attention_mask=attention_mask).last_hidden_state

    pi = torch.randperm(cfg.hidden_size)
    apply_pi_to_bert(model, pi)

    with torch.no_grad():
        pooled1 = model(input_ids=inputs, attention_mask=attention_mask).pooler_output if with_pooler else model(input_ids=inputs, attention_mask=attention_mask).last_hidden_state

    pinv = inv_perm(pi)
    pooled1_back = pooled1.index_select(-1, pinv)
    assert torch.allclose(pooled0, pooled1_back, atol=1e-5, rtol=1e-5)


def test_tied_lm_head_equivalence():
    pytest.importorskip("transformers")
    from transformers import BertConfig, BertForMaskedLM

    torch.manual_seed(11)
    cfg = BertConfig(
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=4,
        num_hidden_layers=1,
        vocab_size=40,
    )
    model = BertForMaskedLM(cfg)
    model.eval()

    inputs = torch.randint(0, cfg.vocab_size, (4, 10))
    with torch.no_grad():
        logits0 = model(input_ids=inputs).logits

    pi = torch.randperm(cfg.hidden_size)
    apply_pi_to_bert(model, pi)

    with torch.no_grad():
        logits1 = model(input_ids=inputs).logits

    assert torch.allclose(logits0, logits1, atol=1e-5, rtol=1e-5)
    # Ensure embeddings and decoder share signatures and align
    sig = perm_signature(pi)
    assert getattr(model.config, "pi_signature") == sig
    assert torch.allclose(
        model.bert.embeddings.word_embeddings.weight,
        model.cls.predictions.decoder.weight,
    )


def test_apply_pi_to_mamba_mock_layer():
    torch.manual_seed(13)

    class MiniMamba(torch.nn.Module):
        def __init__(self, dim: int, input_dim: int, output_dim: int):
            super().__init__()
            self.A = torch.nn.Linear(dim, dim, bias=False)
            self.B = torch.nn.Linear(input_dim, dim, bias=False)
            self.C = torch.nn.Linear(dim, output_dim, bias=False)
            self.x0 = torch.nn.Parameter(torch.randn(dim))

        def forward(self, u: torch.Tensor) -> torch.Tensor:
            x = self.x0
            outputs = []
            for t in range(u.shape[0]):
                x = self.A(x) + self.B(u[t])
                y = self.C(x)
                outputs.append(y)
            return torch.stack(outputs), x

    layer = MiniMamba(dim=8, input_dim=3, output_dim=5)
    seq = torch.randn(6, 3)
    with torch.no_grad():
        y0, x_final0 = layer(seq)

    pi = torch.randperm(8)
    apply_pi_to_mamba({"A": layer.A, "B": layer.B, "C": layer.C, "x0": layer.x0}, pi)

    with torch.no_grad():
        y1, x_final1 = layer(seq)

    assert torch.allclose(y0, y1, atol=1e-5, rtol=1e-5)
    pinv = inv_perm(pi)
    assert torch.allclose(x_final0, x_final1.index_select(0, pinv), atol=1e-5, rtol=1e-5)


def test_apply_pi_sequence_auto_collects_mamba_modules():
    class TinyMamba(torch.nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.A = torch.nn.Linear(dim, dim, bias=True)
            self.B = torch.nn.Linear(dim, dim, bias=True)
            self.C = torch.nn.Linear(dim, dim, bias=True)
            self.x0 = torch.nn.Parameter(torch.arange(dim, dtype=torch.float32))
            self.config = SimpleNamespace(model_type="mamba")

    dim = 4
    model = TinyMamba(dim)
    pi = torch.tensor([2, 0, 3, 1], dtype=torch.long)
    pinv = inv_perm(pi)

    orig_A = model.A.weight.detach().clone()
    orig_B = model.B.weight.detach().clone()
    orig_C = model.C.weight.detach().clone()
    orig_x0 = model.x0.detach().clone()

    ctx = {"_hf_cache": {}, "permutation_after": pi.tolist(), "config": {}}
    quant_cfg = QuantConfig()

    _apply_pi_sequence(model, ctx, pi.tolist(), quant_cfg)

    modules = ctx.get("mamba_modules")
    assert isinstance(modules, list) and modules, "expected auto-collected Mamba modules"

    assert torch.allclose(model.A.weight, PWP_inv(orig_A, pi, pinv))
    assert torch.allclose(model.B.weight, reindex_rows(orig_B, pi))
    assert torch.allclose(model.C.weight, reindex_cols(orig_C, pi))
    assert torch.allclose(model.x0, reindex_vec(orig_x0, pi))
