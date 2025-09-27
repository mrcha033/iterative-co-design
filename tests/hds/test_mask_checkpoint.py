from pathlib import Path

import torch

from icd.hds.checkpoint import deserialize_mask_state, load_mask_state, save_mask_state, serialize_mask_state
from icd.hds.layers import NMLinear


def _toy_model() -> torch.nn.Module:
    torch.manual_seed(0)
    model = torch.nn.Sequential(NMLinear(8, 4, bias=False))
    return model


def test_serialize_and_deserialize_roundtrip(tmp_path: Path) -> None:
    model = _toy_model()
    module = next(m for m in model.modules() if isinstance(m, NMLinear))
    module.masker.logits.data.uniform_(-1.0, 1.0)

    state = serialize_mask_state(model)
    assert state["version"] == 1
    assert state["modules"]

    # Zero logits to ensure load repopulates them
    module.masker.logits.data.zero_()
    deserialize_mask_state(model, state)
    assert torch.allclose(module.masker.logits, torch.tensor(state["modules"][0]["logits"]))


def test_save_and_load_mask_state(tmp_path: Path) -> None:
    model = _toy_model()
    module = next(m for m in model.modules() if isinstance(m, NMLinear))
    module.masker.logits.data.fill_(0.25)

    out_path = tmp_path / "mask_state.json"
    save_mask_state(out_path, model)
    assert out_path.exists()

    module.masker.logits.data.zero_()
    payload = load_mask_state(out_path, model)
    assert payload["modules"][0]["path"] == "0"
    assert torch.allclose(module.masker.logits, torch.full_like(module.masker.logits, 0.25))

