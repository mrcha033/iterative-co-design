from __future__ import annotations

import pytest
import torch
import torch.backends.quantized
from torch.nn.utils import prune
from types import SimpleNamespace
from transformers import BertConfig, BertForSequenceClassification

import icd.adapters.quant as quant_mod
import icd.adapters.sparsity as sparsity_mod
from icd.adapters.quant import (
    _BNB_AVAILABLE,
    QuantConfig,
    apply_bnb_int8,
    apply_post_training_quantization,
    apply_quant,
    apply_quant_from_config,
    repack_linear_after_permutation,
)
from icd.adapters.sparsity import (
    SparsityConfig,
    apply_unstructured,
    apply_sparsity,
    apply_sparsity_from_config,
    iter_prunable_linears,
    _global_threshold,
    _normalize_sparsity_kwargs,
)
from icd.hds.layers import NMLinear


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


def test_structured_sparsity_converts_linears():
    model = _mini_bert()
    converted_before = sum(isinstance(m, torch.nn.Linear) for m in model.modules())
    nm_before = sum(isinstance(m, NMLinear) for m in model.modules())

    result, meta = apply_sparsity(model, type="2:4", rate=0.5)
    assert result is model
    nm_after = sum(isinstance(m, NMLinear) for m in model.modules())
    assert nm_after >= nm_before
    assert meta["delta_layout"] is True
    assert meta["sparsity"]["type"] == "2:4"
    assert meta["sparsity"]["converted"] == nm_after
    summary = meta["sparsity"].get("summary")
    assert summary is not None
    assert summary.get("converted") == meta["sparsity"]["converted"]
    assert summary.get("skipped") == []
    lin_after = sum(isinstance(m, torch.nn.Linear) for m in model.modules())
    assert lin_after + nm_after >= converted_before


def test_structured_sparsity_skips_incompatible_layer():
    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bad = torch.nn.Linear(3, 2)

    model = Tiny()
    result, meta = apply_sparsity(model, type="2:4", rate=1.0, modules=[model.bad])
    assert result is model
    assert meta["delta_layout"] is False
    assert meta["sparsity"]["converted"] == 0
    summary = meta["sparsity"].get("summary")
    assert summary is not None
    assert summary.get("converted") == 0
    skipped = summary.get("skipped")
    assert skipped and skipped[0]["reason"] == "incompatible_input_dims"
    warnings_meta = meta.get("warnings", [])
    assert warnings_meta and warnings_meta[0]["kind"] == "sparsity_skip"


def test_apply_sparsity_no_effect_returns_meta():
    model, meta = apply_sparsity(None, type="none", rate=0.0)
    assert model is None
    assert meta["delta_layout"] is False


def test_apply_sparsity_warns_when_no_modules(monkeypatch):
    class Empty(torch.nn.Module):
        pass

    with pytest.warns(RuntimeWarning, match="no prunable modules"):
        result, meta = apply_sparsity(Empty(), rate=0.5, modules=[torch.nn.ReLU()])
    assert result is not None
    assert meta["delta_layout"] is True


def test_apply_sparsity_random_unstructured_respects_method():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    result, meta = apply_sparsity(
        model,
        rate=0.25,
        method="random_unstructured",
        global_unstructured=False,
    )
    assert result is model
    assert meta["sparsity"]["method"] == "random_unstructured"


def test_apply_sparsity_structured_without_nmlinear(monkeypatch):
    monkeypatch.setattr("icd.adapters.sparsity.NMLinear", None)
    monkeypatch.setattr("icd.adapters.sparsity.TopKMaskerConfig", None)

    model = torch.nn.Sequential(torch.nn.Linear(8, 4))
    with pytest.warns(RuntimeWarning, match="NMLinear not available"):
        result, meta = apply_sparsity(model, type="2:4", rate=1.0)
    assert result is model
    assert meta["sparsity"]["converted"] == 0


def test_apply_sparsity_structured_parent_lookup(monkeypatch):
    model = torch.nn.Sequential(torch.nn.Linear(8, 4))
    external = torch.nn.Linear(8, 4)
    with pytest.warns(RuntimeWarning):
        result, meta = apply_sparsity(model, type="2:4", rate=1.0, modules=[external])
    assert result is model
    summary = meta["sparsity"].get("summary")
    assert summary and summary.get("skipped")


def test_apply_sparsity_structured_skips_non_linear_and_duplicates():
    model = torch.nn.Sequential(torch.nn.Linear(8, 4))
    linear = list(model.modules())[1]
    modules = [torch.nn.ReLU(), linear, linear]
    result, meta = apply_sparsity(model, type="2:4", rate=1.0, modules=modules)
    assert result is model
    summary = meta["sparsity"].get("summary")
    assert summary["total_targets"] >= 1


def test_iter_prunable_linears_respects_filters():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    items = list(iter_prunable_linears(model, apply_to=("encoder",), exclude=("linear",)))
    assert items == []


def test_global_threshold_empty_returns_inf():
    assert _global_threshold([], amount=0.5) == float("inf")


def test_global_threshold_full_amount():
    tensor = torch.ones(4)
    assert _global_threshold([tensor], amount=1.0) == pytest.approx(1.0)


def test_global_threshold_small_amount():
    tensor = torch.arange(1, 5, dtype=torch.float32)
    assert _global_threshold([tensor], amount=0.01) == pytest.approx(-1.0)


def test_apply_unstructured_per_layer(monkeypatch):
    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    prune_calls = {"count": 0}

    def fake_l1(module, name, amount):
        prune_calls["count"] += 1

    monkeypatch.setattr(prune, "l1_unstructured", fake_l1)
    monkeypatch.setattr(prune, "remove", lambda module, name: None)

    apply_unstructured(model, amount=0.25, scope="per_layer", apply_to=("0",), exclude=())
    assert prune_calls["count"] == 1


def test_apply_unstructured_no_amount():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    apply_unstructured(model, amount=0.0)


def test_apply_unstructured_warns_when_empty(monkeypatch):
    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    with pytest.warns(RuntimeWarning):
        apply_unstructured(model, amount=0.5, apply_to=("missing",))


def test_apply_sparsity_global_unstructured(monkeypatch):
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
    modules = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
    called = {}

    def fake_global(params, pruning_method, amount):
        called["params"] = params

    monkeypatch.setattr(prune, "global_unstructured", fake_global)
    monkeypatch.setattr(prune, "remove", lambda module, name: None)

    result, meta = apply_sparsity(
        model,
        type="unstructured",
        rate=0.25,
        modules=modules,
        global_unstructured=True,
    )
    assert result is model
    assert called["params"]
    assert meta["sparsity"]["method"] == "l1_unstructured"


def test_apply_sparsity_local_unstructured(monkeypatch):
    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    modules = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
    calls = {"count": 0}
    removals = {"count": 0}

    def fake_l1(module, name, amount):
        calls["count"] += 1

    def fake_remove(module, name):
        removals["count"] += 1
        if removals["count"] == 1:
            raise RuntimeError("fail")

    monkeypatch.setattr(prune, "l1_unstructured", fake_l1)
    monkeypatch.setattr(prune, "remove", fake_remove)

    result, meta = apply_sparsity(
        model,
        type="unstructured",
        rate=0.5,
        modules=modules,
        global_unstructured=False,
    )
    assert result is model
    assert calls["count"] == len(modules)


def test_apply_sparsity_from_config_errors():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    cfg = SparsityConfig(type="none")
    apply_sparsity_from_config(model, cfg)
    with pytest.raises(ValueError):
        apply_sparsity_from_config(model, SparsityConfig(type="structured"))


def test_apply_sparsity_from_config_invokes_unstructured(monkeypatch):
    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    called = {}

    def fake_apply(model, **kwargs):
        called["kwargs"] = kwargs

    monkeypatch.setattr(sparsity_mod, "apply_unstructured", fake_apply)
    cfg = SparsityConfig(type="unstructured", amount=0.2, scope="per_layer")
    apply_sparsity_from_config(model, cfg)
    assert called["kwargs"]["amount"] == 0.2


def test_normalize_sparsity_kwargs_defaults():
    typ, effective = _normalize_sparsity_kwargs(type=None, rate=None, amount=None, sparsity=None)
    assert typ == "unstructured" and effective == 0.0

def test_apply_quant_dynamic_and_fp16_paths():
    prev_engine = torch.backends.quantized.engine
    torch.backends.quantized.engine = "qnnpack"
    try:
        model = torch.nn.Sequential(torch.nn.Linear(4, 4))

        quantized, meta = apply_quant(model, method="dynamic", dtype="qint8")
        quantized_linears = [
            m for m in quantized.modules() if m.__class__.__module__.startswith("torch.ao.nn.quantized")
        ]
        assert quantized_linears, "expected quantized linear modules to be created"
        assert meta["quant"]["method"] == "dynamic"

        fp_model = torch.nn.Sequential(torch.nn.Linear(4, 4))
        fp_quant, fp_meta = apply_quant(fp_model, method="fp16")
        assert fp_quant[0].weight.dtype == torch.float16
        assert fp_meta["delta_layout"] is True
    finally:
        torch.backends.quantized.engine = prev_engine


def test_apply_quant_handles_unavailable_backends(monkeypatch):
    model = torch.nn.Sequential(torch.nn.Linear(4, 4))

    monkeypatch.setattr(quant_mod, "_quantize_dynamic", None)
    with pytest.warns(UserWarning, match="quantization unavailable"):
        unchanged, meta = apply_quant(model, method="dynamic")
    assert unchanged is model
    assert meta["quant"]["method"] == "dynamic"


def test_apply_quant_bnb_falls_back_to_dynamic(monkeypatch):
    prev_engine = torch.backends.quantized.engine
    torch.backends.quantized.engine = "qnnpack"
    calls: dict[str, int] = {"count": 0}

    def fake_quantize(model, modules, dtype=None, inplace=True):
        calls["count"] += 1
        return model

    monkeypatch.setattr(quant_mod, "_quantize_dynamic", fake_quantize)
    monkeypatch.setattr(quant_mod, "_BNB_AVAILABLE", False)
    try:
        with pytest.warns(UserWarning, match="bitsandbytes not available"):
            quantized, meta = apply_quant(torch.nn.Sequential(torch.nn.Linear(4, 4)), method="bnb")
        assert calls["count"] == 1
        assert meta["quant"]["method"] == "dynamic"
        assert quantized is not None
    finally:
        torch.backends.quantized.engine = prev_engine


def test_apply_bnb_int8_uses_stubbed_linear(monkeypatch):
    monkeypatch.setattr(quant_mod, "_BNB_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="bitsandbytes is required"):
        apply_bnb_int8(torch.nn.Linear(2, 2))

    class FakeState:
        def __init__(self) -> None:
            self.dtype = None

        def set_compute_type(self, dtype: torch.dtype) -> None:
            self.dtype = dtype

    class FakeBnbLinear(torch.nn.Module):
        def __init__(self, in_features=4, out_features=4, bias=True) -> None:
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.weight = torch.nn.Parameter(torch.ones(out_features, in_features))
            self.bias = torch.nn.Parameter(torch.zeros(out_features)) if bias else None
            self.state = FakeState()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    class FakeQuantModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer = FakeBnbLinear()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    def fake_replace(model, modules_to_not_convert=None, quantization_config=None):
        return FakeQuantModel()

    import transformers

    monkeypatch.setattr(quant_mod, "_BNB_AVAILABLE", True)
    monkeypatch.setattr(quant_mod, "_ensure_bnb", lambda: None)
    monkeypatch.setattr(quant_mod, "_bnb_linear_type", lambda: FakeBnbLinear)
    monkeypatch.setattr(transformers, "replace_with_bnb_linear", fake_replace, raising=False)

    result = apply_bnb_int8(torch.nn.Linear(4, 4), compute_dtype=torch.float32)
    assert isinstance(result, FakeQuantModel)
    layer = result.layer
    assert isinstance(layer, FakeBnbLinear)
    assert layer.state.dtype == torch.float32


def test_repack_linear_after_permutation_recreates_quant_module(monkeypatch):
    class FakeState:
        def __init__(self) -> None:
            self.dtype = None

        def set_compute_type(self, dtype: torch.dtype) -> None:
            self.dtype = dtype

    class FakeBnbLinear(torch.nn.Module):
        def __init__(self, in_features=4, out_features=4, bias=True) -> None:
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.weight = torch.nn.Parameter(torch.ones(out_features, in_features))
            self.bias = torch.nn.Parameter(torch.zeros(out_features)) if bias else None
            self.state = FakeState()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    fake_module = FakeBnbLinear()
    monkeypatch.setattr(quant_mod, "_BNB_AVAILABLE", True)
    monkeypatch.setattr(quant_mod, "_bnb_linear_type", lambda: FakeBnbLinear)

    repacked = repack_linear_after_permutation(fake_module)
    assert isinstance(repacked, FakeBnbLinear)
    assert repacked is not fake_module


def test_repack_linear_after_permutation_skips_non_bnb(monkeypatch):
    class Dummy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2, 2))
            self.bias = torch.nn.Parameter(torch.zeros(2))

    module = Dummy()
    monkeypatch.setattr(quant_mod, "_BNB_AVAILABLE", True)
    monkeypatch.setattr(quant_mod, "_bnb_linear_type", lambda: type("Fake", (), {}))

    same = repack_linear_after_permutation(module)
    assert same is module

def test_apply_quant_from_config_and_post_training(monkeypatch):
    prev_engine = torch.backends.quantized.engine
    torch.backends.quantized.engine = "qnnpack"
    try:
        model = torch.nn.Sequential(torch.nn.Linear(4, 4))
        cfg = QuantConfig(type="dynamic")
        result = apply_quant_from_config(model, cfg)
        assert isinstance(result, torch.nn.Module)

        quantized = apply_post_training_quantization(torch.nn.Sequential(torch.nn.Linear(4, 4)), dtype="int8")
        assert isinstance(quantized, torch.nn.Module)

        with pytest.warns(UserWarning, match="Unknown dtype"):
            same = apply_post_training_quantization(torch.nn.Sequential(torch.nn.Linear(4, 4)), dtype="unknown")
        assert isinstance(same, torch.nn.Module)
    finally:
        torch.backends.quantized.engine = prev_engine


def test_apply_quant_fp16_manual_fallback(monkeypatch):
    class FragileLinear(torch.nn.Linear):
        def to(self, *args, **kwargs):  # type: ignore[override]
            raise RuntimeError("cannot cast")

    class Container(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer = FragileLinear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.layer(x)

    model = Container()
    result, meta = apply_quant(model, method="fp16")
    assert result.layer.weight.dtype == torch.float16
    assert meta["quant"]["method"] == "fp16"


def test_apply_quant_bnb_with_stub(monkeypatch):
    prev_engine = torch.backends.quantized.engine
    torch.backends.quantized.engine = "qnnpack"
    try:
        monkeypatch.setattr(quant_mod, "_BNB_AVAILABLE", True)
        called = {}

        def fake_apply_bnb_int8(model):
            called["model"] = model
            return model

        monkeypatch.setattr(quant_mod, "apply_bnb_int8", fake_apply_bnb_int8)

        model = torch.nn.Sequential(torch.nn.Linear(4, 4))
        quantized, meta = apply_quant(model, method="bnb")
        assert called["model"] is model
        assert quantized is model
        assert meta["quant"]["method"] == "bnb"
    finally:
        torch.backends.quantized.engine = prev_engine


def test_apply_quant_unknown_method(monkeypatch):
    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    with pytest.warns(UserWarning, match="Unknown quantization method"):
        same, meta = apply_quant(model, method="mystery")
    assert same is model
    assert meta["quant"]["method"] == "mystery"


def test_apply_post_training_quantization_fp16():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    quantized = apply_post_training_quantization(model, dtype="fp16")
    assert quantized[0].weight.dtype == torch.float16


def test_quant_helper_normalizations():
    assert quant_mod._normalize_dtype(torch.qint8) is torch.qint8
    assert quant_mod._normalize_dtype("float16") is torch.float16
    assert quant_mod._normalize_dtype("other") is torch.qint8
    classes = quant_mod._default_quant_modules(torch.nn.Linear(4, 4))
    assert torch.nn.Linear in classes
    assert quant_mod._dtype_label(torch.float16) == "float16"
    assert quant_mod._dtype_label(torch.float32) == "float32"


def test_default_quant_modules_handles_missing_nn(monkeypatch):
    monkeypatch.setattr(quant_mod, "nn", None)
    assert quant_mod._default_quant_modules(None) == tuple()


def test_ensure_bnb_and_linear_type(monkeypatch):
    monkeypatch.setattr(quant_mod, "_BNB_AVAILABLE", False)
    with pytest.raises(RuntimeError):
        quant_mod._ensure_bnb()

    class FakeLinear:
        pass

    fake_bnb = SimpleNamespace(nn=SimpleNamespace(Linear8bitLt=FakeLinear))
    monkeypatch.setattr(quant_mod, "bnb", fake_bnb)
    monkeypatch.setattr(quant_mod, "_BNB_AVAILABLE", True)
    assert quant_mod._bnb_linear_type() is FakeLinear


def test_apply_quant_from_config_bnb4bit(monkeypatch):
    class FakeBitsAndBytesConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeQuant(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    import transformers

    monkeypatch.setattr(quant_mod, "_BNB_AVAILABLE", True)
    monkeypatch.setattr(quant_mod, "_ensure_bnb", lambda: None)
    monkeypatch.setattr(transformers, "BitsAndBytesConfig", FakeBitsAndBytesConfig, raising=False)

    called = {}

    def fake_replace(model, modules_to_not_convert=None, quantization_config=None):
        called["modules"] = modules_to_not_convert
        called["cfg"] = quantization_config
        return FakeQuant()

    monkeypatch.setattr(transformers, "replace_with_bnb_linear", fake_replace, raising=False)

    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    result = apply_quant_from_config(model, QuantConfig(type="bnb-4bit"))
    assert isinstance(result, FakeQuant)
    assert called["modules"] == ["lm_head", "classifier"]
    assert isinstance(called["cfg"], FakeBitsAndBytesConfig)


def test_apply_quant_from_config_bnb_int8(monkeypatch):
    monkeypatch.setattr(quant_mod, "_BNB_AVAILABLE", True)
    called = {}

    def fake_apply(model):
        called["model"] = model
        return model

    monkeypatch.setattr(quant_mod, "apply_bnb_int8", fake_apply)

    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    result = apply_quant_from_config(model, QuantConfig(type="bnb-int8"))
    assert result is model
    assert called["model"] is model


def test_apply_quant_from_config_unsupported():
    with pytest.raises(ValueError):
        apply_quant_from_config(torch.nn.Sequential(torch.nn.Linear(4, 4)), QuantConfig(type="unknown"))

def test_apply_post_training_quantization_int4(monkeypatch):
    recorded = {}

    def fake_apply_quant_from_config(model, cfg):
        recorded["cfg"] = cfg
        return model

    monkeypatch.setattr(quant_mod, "apply_quant_from_config", fake_apply_quant_from_config)

    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    quantized = apply_post_training_quantization(model, dtype="int4")
    assert quantized is model
    assert recorded["cfg"].type == "bnb-4bit"
