from __future__ import annotations

import pytest
import torch
from transformers import BertConfig, BertForSequenceClassification

from icd.adapters.quant import (
    _BNB_AVAILABLE,
    QuantConfig,
    apply_bnb_int8,
    apply_quant_from_config,
    repack_linear_after_permutation,
)
from icd.adapters.sparsity import SparsityConfig, apply_unstructured


def _mini_bert() -> BertForSequenceClassification:
    cfg = BertConfig(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_hidden_layers=2,
        vocab_size=128,
    )
    return BertForSequenceClassification(cfg)


def _count_nonzero_weights(model: torch.nn.Module) -> int:
    return sum(int((param != 0).sum().item()) for name, param in model.named_parameters() if name.endswith("weight"))


def test_unstructured_reduces_density():
    model = _mini_bert()
    before_nz = _count_nonzero_weights(model)
    apply_unstructured(model, amount=0.5, scope="per_layer")
    after_nz = _count_nonzero_weights(model)
    assert after_nz < before_nz


def test_sparsity_config_helper():
    cfg = SparsityConfig.from_dict({"type": "unstructured", "amount": 0.3, "scope": "global"})
    model = _mini_bert()
    apply_unstructured(model, amount=cfg.amount, scope=cfg.scope)


@pytest.mark.cuda
def test_bnb_int8_quant_runs_forward():
    if not torch.cuda.is_available() or not _BNB_AVAILABLE:
        pytest.skip("CUDA not available")
    model = _mini_bert().to("cuda")
    quantized = apply_bnb_int8(model)
    inputs = torch.randint(0, 128, (2, 16), device="cuda")
    with torch.no_grad():
        quantized(input_ids=inputs)


def test_repack_linear_after_permutation_handles_non_quantized():
    lin = torch.nn.Linear(8, 4)
    repacked = repack_linear_after_permutation(lin)
    assert repacked is lin


def test_quant_config_routes_none():
    model = _mini_bert()
    cfg = QuantConfig.from_dict(None)
    same = apply_quant_from_config(model, cfg)
    assert same is model
