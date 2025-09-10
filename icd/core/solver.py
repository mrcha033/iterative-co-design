from __future__ import annotations

import time
from typing import Dict, List, Tuple

from .graph import CSRMatrix
from .cost import CostConfig, eval_cost, invperm


def _spectral_init_like(W: CSRMatrix, seed: int = 0) -> List[int]:
    """Spectral-like initialization using Laplacian Fiedler vector when feasible.

    Fallbacks to degree-sort when eigensolve is not available or too costly.
    """
    n = W.shape[0]
    # Try a tiny eigen solve with numpy for small n
    try:
        if n <= 512:
            import numpy as np

            M = np.array(W.to_dense(), dtype=float)
            # symmetrize (defensive)
            M = (M + M.T) * 0.5
            d = M.sum(axis=1)
            L = np.diag(d) - M
            vals, vecs = np.linalg.eigh(L)
            # second smallest eigenvector (Fiedler)
            idx = np.argsort(vals)
            fied = vecs[:, idx[1]] if len(idx) > 1 else vecs[:, idx[0]]
            order = list(np.argsort(fied))
            return order
    except Exception:
        pass

    # Fallback: degree-based ordering with deterministic tie-breaker
    import random

    deg = [0.0] * n
    for i in range(n):
        for t in range(W.indptr[i], W.indptr[i + 1]):
            j = W.indices[t]
            w = W.data[t]
            deg[i] += w
            if j < n:
                deg[j] += w
    rng = random.Random(seed)
    tiny = [rng.random() * 1e-9 for _ in range(n)]
    order = list(range(n))
    order.sort(key=lambda i: (deg[i], tiny[i]))
    return order


def _local_refine_adjacent(W: CSRMatrix, pi: List[int], cfg: CostConfig, steps: int, time_budget_s: float) -> Tuple[List[int], bool]:
    start = time.perf_counter()
    improved = False
    best = pi[:]
    best_cost = eval_cost(W, best, best, cfg)["J"]
    n = len(pi)
    idx = 0
    while steps > 0 and (time.perf_counter() - start) < time_budget_s:
        i = idx % (n - 1)
        idx += 1
        steps -= 1
        cand = best[:]
        # swap adjacent positions
        cand[i], cand[i + 1] = cand[i + 1], cand[i]
        jcand = eval_cost(W, cand, best, cfg)["J"]
        if jcand < best_cost:
            best = cand
            best_cost = jcand
            improved = True
    return best, improved


def fit_permutation(
    W: CSRMatrix,
    time_budget_s: float = 5.0,
    refine_steps: int = 1000,
    cfg: CostConfig | None = None,
    seed: int = 0,
) -> Tuple[List[int], Dict[str, float]]:
    """Compute permutation using spectral-like init and local adjacent swaps.

    Returns (pi, stats) with stats including C, Q, J, improved flag.
    """
    cfg = cfg or CostConfig()
    n = W.shape[0]
    pi_id = list(range(n))
    stats_id = eval_cost(W, pi_id, pi_id, cfg)

    pi0 = _spectral_init_like(W, seed=seed)
    stats0 = eval_cost(W, pi0, pi0, cfg)

    # Start refine from the better of identity vs spectral init
    start_pi = pi0 if stats0["J"] <= stats_id["J"] else pi_id
    base_stats = stats0 if start_pi is pi0 else stats_id

    pi1, improved = _local_refine_adjacent(W, start_pi, cfg, steps=refine_steps, time_budget_s=time_budget_s)
    stats1 = eval_cost(W, pi1, start_pi, cfg)

    # Safety: never return worse than identity
    if stats1["J"] > stats_id["J"]:
        pi1, stats1 = pi_id, stats_id
        improved = False

    stats1["improved"] = bool(improved or (stats1["J"] < base_stats["J"]))
    return pi1, stats1


__all__ = ["fit_permutation"]
