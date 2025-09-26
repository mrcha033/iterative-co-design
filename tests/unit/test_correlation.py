import torch

from icd.graph.correlation import CorrelationConfig, collect_correlations, correlation_to_csr


class IdentityLinear(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(dim, dim, bias=False)
        torch.nn.init.eye_(self.linear.weight)

    def forward(self, x):
        return self.linear(x)


def test_activation_correlation_matches_manual():
    model = IdentityLinear(2)
    inputs = [
        (torch.tensor([[1.0, 2.0], [3.0, 4.0]]),),
        (torch.tensor([[0.0, 1.0], [1.0, 0.0]]),),
    ]
    cfg = CorrelationConfig(layers=["linear"], samples=len(inputs), dtype=torch.float64)
    matrix, meta = collect_correlations(model, inputs, cfg=cfg)

    all_samples = torch.cat([x for (x,) in inputs], dim=0)
    mean = all_samples.mean(dim=0)
    manual = (all_samples - mean).t().mm(all_samples - mean) / all_samples.shape[0]
    manual = manual.to(matrix.dtype)

    assert torch.allclose(matrix, manual, atol=1e-6)
    assert meta["samples"] == len(inputs)


def test_correlation_to_csr_threshold():
    matrix = torch.tensor(
        [
            [0.0, 0.5, 0.1],
            [0.5, 0.0, 0.2],
            [0.1, 0.2, 0.0],
        ]
    )
    cfg = CorrelationConfig(threshold=0.2, normalize="none", nnz_cap=10)
    csr = correlation_to_csr(matrix, cfg=cfg)
    assert csr.shape == (3, 3)
    assert csr.nnz() == 4  # pairs (0,1),(1,0),(1,2),(2,1)
