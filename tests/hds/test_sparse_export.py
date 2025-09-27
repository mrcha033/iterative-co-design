from pathlib import Path

import torch

from icd.hds.layers import NMLinear
from icd.runtime.export_sparse import collect_sparse_layers, export_sparse_model


def _model_with_masks() -> torch.nn.Module:
    torch.manual_seed(0)
    layer = NMLinear(4, 2, bias=True)
    layer.masker.logits.data.fill_(10.0)
    model = torch.nn.Sequential(layer)
    with torch.no_grad():
        layer(torch.randn(1, 4))  # populate last_mask
    return model


def test_collect_sparse_layers_returns_metadata() -> None:
    model = _model_with_masks()
    layers = collect_sparse_layers(model)
    assert len(layers) == 1
    assert layers[0].metadata["group_size"] == 4


def test_export_sparse_model(tmp_path: Path) -> None:
    model = _model_with_masks()
    out_path = tmp_path / "sparse.pt"
    payload = export_sparse_model(model, out_path, metadata={"model": "toy"})
    assert out_path.exists()
    loaded = torch.load(out_path)
    assert loaded["version"] == 1
    assert loaded["meta"]["model"] == "toy"
    assert torch.equal(loaded["layers"][0]["mask"], payload["layers"][0]["mask"])

