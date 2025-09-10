from icd.core.graph import build_w
from icd.core.solver import fit_permutation
from icd.core.cost import CostConfig


def test_solver_time_budget():
    W = build_w(source="mock", D=128, blocks=4, noise=0.02, seed=0)
    pi, stats = fit_permutation(W, time_budget_s=0.001, refine_steps=10, cfg=CostConfig(), seed=0)
    assert isinstance(pi, list) and len(pi) == W.shape[0]
    assert "J" in stats and "C" in stats and "Q" in stats

