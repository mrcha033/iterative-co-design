import pytest

torch = pytest.importorskip("torch")

from icd.hds.topk import TopKMasker, TopKMaskerConfig
from icd.hds.layers import NMLinear
from icd.hds.training import MaskTrainingConfig, run_mask_training


def mask_training_stepper(*, model, step, temperature, masks, context=None):
    loss = torch.zeros((), device=masks[0].device if masks else "cpu")
    for mask in masks:
        loss = loss + (mask - mask.mean()).pow(2).mean()
    if context is not None:
        context.setdefault("calls", []).append({"step": step, "temperature": temperature})
    return loss


def test_topk_masker_deterministic_eval():
    cfg = TopKMaskerConfig(group_size=4, active=2, temperature_init=1.0, temperature_final=0.5, seed=42)
    masker = TopKMasker(size=8, config=cfg)
    masker.eval()
    mask = masker(sample=False)
    mask = mask.view(2, 4)
    assert mask.sum().item() == pytest.approx(4.0)
    for row in mask:
        assert row.sum().item() == pytest.approx(2.0)
    mask2 = masker(sample=False)
    assert torch.equal(mask, mask2.view(2, 4))


def test_topk_masker_gradients():
    cfg = TopKMaskerConfig(group_size=4, active=2, temperature_init=1.0, temperature_final=0.5, seed=0)
    masker = TopKMasker(size=8, config=cfg)
    optimizer = torch.optim.SGD(masker.parameters(), lr=0.1)
    masker.train()
    for step in range(3):
        optimizer.zero_grad()
        mask = masker(step=step)
        loss = mask.pow(2).sum()
        loss.backward()
        optimizer.step()
    assert masker.logits.grad is not None
    assert torch.isfinite(masker.logits.grad).all()


def test_nmlinear_masks_weights():
    layer = NMLinear(4, 2, n_active=2, m_group=4)
    x = torch.ones(1, 4)
    output = layer(x, sample=False)
    assert output.shape == (1, 2)
    mask = layer.masker.last_mask().view(2, 4)
    assert torch.all(mask.sum(dim=1) == 2)
    masked_weight = layer.masked_weight()
    manual = layer.linear.weight * mask
    assert torch.allclose(masked_weight, manual)


def test_mask_training_runs_and_stashes_state():
    layer = NMLinear(4, 2, n_active=2, m_group=4)
    cfg = MaskTrainingConfig(
        steps=3,
        temperature_init=1.0,
        temperature_final=0.2,
        anneal_steps=3,
        optimizer="sgd",
        lr=0.05,
        state_key="test.opt",
    )
    context: dict[str, object] = {}
    meta = run_mask_training(layer, cfg, context=context)
    assert meta["steps"] == 3
    temps = meta["temperature"]["values"]
    assert len(temps) == 3
    assert temps[0] == pytest.approx(1.0)
    assert temps[-1] == pytest.approx(0.2, rel=1e-3)
    assert context["test.opt::before"]
    assert context["test.opt::after"]
    assert meta["optimizer"]["state_stashed"] is True


def test_mask_training_custom_stepper_records_context():
    layer = NMLinear(4, 2, n_active=2, m_group=4)
    cfg = MaskTrainingConfig(
        steps=2,
        optimizer="adam",
        lr=0.01,
        stepper="tests.unit.test_hds:mask_training_stepper",
        stash_optimizer=False,
        sample=False,
    )
    context: dict[str, object] = {}
    meta = run_mask_training(layer, cfg, context=context)
    assert meta["steps"] == 2
    calls = context.get("calls")
    assert isinstance(calls, list) and len(calls) == 2
    assert calls[0]["step"] == 0
    assert calls[1]["step"] == 1
