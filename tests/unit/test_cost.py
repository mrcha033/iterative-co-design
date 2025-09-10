from icd.core.cost import CostConfig, eval_cost
from icd.core.graph import build_w
from icd.core.solver import fit_permutation


def test_blocky_monotonicity():
    W = build_w(source="mock", D=128, blocks=4, noise=0.02, seed=0)
    n = W.shape[0]
    pi_id = list(range(n))
    cfg = CostConfig()
    stats_id = eval_cost(W, pi_id, pi_id, cfg)
    pi, stats = fit_permutation(W, time_budget_s=0.5, refine_steps=200, cfg=cfg, seed=0)
    assert stats["C"] <= stats_id["C"] * 0.95
    assert stats["Q"] >= stats_id["Q"]

