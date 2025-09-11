from icd.adapters.sparsity import apply_sparsity
from icd.adapters.quant import apply_quant
from icd.adapters.kv import apply_kvcache


def test_sparsity_meta_and_trigger():
    _, m = apply_sparsity(None, type="2:4", rate=0.5)
    assert m.get("delta_layout") is True
    assert m.get("sparsity", {}).get("type") == "2:4"


def test_quant_meta_and_trigger():
    _, m = apply_quant(None, dtype="int8", method="ptq-minmax")
    assert m.get("delta_layout") is True
    assert m.get("quant", {}).get("dtype") == "int8"


def test_kv_meta_and_trigger():
    _, m = apply_kvcache(None, block=128, drop=0.05)
    assert m.get("delta_layout") is True
    assert m.get("kv", {}).get("block") == 128

