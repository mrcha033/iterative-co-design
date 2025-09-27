from pathlib import Path

import torch

from icd.graph.correlation import CorrelationConfig, collect_correlations


class _Toy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x):
        return self.linear(x)


def test_collect_correlations_respects_transfer_batch(tmp_path: Path) -> None:
    model = _Toy()
    cfg = CorrelationConfig(samples=4, layers=["linear"], transfer_batch_size=1, whiten=True)

    def iterator(idx: int):
        torch.manual_seed(idx)
        return torch.randn(2, 4)

    matrix, meta = collect_correlations(model, iterator, cfg=cfg)
    assert matrix.shape[0] == 4
    layer_meta = meta["layers"][0]
    assert layer_meta["storage_device"] == "cpu"

