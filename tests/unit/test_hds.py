import pytest
import torch

from icd.hds.topk import TopKMasker, TopKMaskerConfig
from icd.hds.layers import NMLinear


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
