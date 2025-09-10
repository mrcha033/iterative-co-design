from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .graph import CSRMatrix


@dataclass
class CostConfig:
    alpha: float = 1.0
    beta: float = 0.2
    gamma_stability: float = 0.1
    mu: float = 0.5
    g: int = 64
    lambda_: int = 3
    tau: float = 0.25
    modularity_gamma: float = 1.2
    blocks_k: int = 4
    vec_width: int = 16
    hysteresis: int = 2


def invperm(pi: List[int]) -> List[int]:
    pos = [0] * len(pi)
    for idx, v in enumerate(pi):
        pos[v] = idx
    return pos


def _phi(cfg: CostConfig, d_ij: int) -> float:
    if d_ij <= cfg.lambda_:
        return float(d_ij)
    return float(cfg.lambda_ + cfg.tau * (d_ij - cfg.lambda_))


def _sum_nnz(W: CSRMatrix) -> float:
    return float(sum(W.data))


def _phi_max(cfg: CostConfig, D: int, g: int) -> float:
    dmax = (max(0, D - 1)) // max(1, g)
    return _phi(cfg, dmax)


def cache_cost(W: CSRMatrix, pi: List[int], cfg: CostConfig) -> float:
    pos = invperm(pi)
    total = 0.0
    n = W.shape[0]
    for i in range(n):
        start, end = W.indptr[i], W.indptr[i + 1]
        for k in range(start, end):
            j = W.indices[k]
            if j <= i:
                continue
            dij = abs(pos[i] - pos[j]) // max(1, cfg.g)
            total += W.data[k] * _phi(cfg, dij)
    denom = _phi_max(cfg, n, cfg.g) * max(_sum_nnz(W), 1e-12)
    return total / denom if denom > 0 else 0.0


def modularity_contiguous_blocks(W: CSRMatrix, pi: List[int], cfg: CostConfig) -> float:
    # Approximate weighted modularity by slicing contiguous blocks in pi
    n = W.shape[0]
    k = max(1, cfg.blocks_k)
    block_len = max(1, n // k)
    block_id = [min(idx // block_len, k - 1) for idx in range(n)]
    pos = invperm(pi)
    # degree and total weight
    deg = [0.0] * n
    for i in range(n):
        for t in range(W.indptr[i], W.indptr[i + 1]):
            j = W.indices[t]
            w = W.data[t]
            deg[i] += w
            if j < n:
                deg[j] += w
    two_m = max(sum(deg), 1e-9)
    Q = 0.0
    for i in range(n):
        bi = block_id[pos[i]]
        for t in range(W.indptr[i], W.indptr[i + 1]):
            j = W.indices[t]
            if j <= i:
                continue
            bj = block_id[pos[j]]
            if bi != bj:
                continue
            w_ij = W.data[t]
            Q += w_ij - cfg.modularity_gamma * (deg[i] * deg[j] / max(two_m, 1e-9))
    return Q / max(two_m, 1e-9)


def align_penalty(W: CSRMatrix, pos: List[int], vec_width: int) -> float:
    n = W.shape[0]
    if n == 0:
        return 0.0
    total_w = max(_sum_nnz(W), 1e-12)
    bad = 0.0
    for i in range(n):
        for t in range(W.indptr[i], W.indptr[i + 1]):
            j = W.indices[t]
            if j <= i:
                continue
            bad += W.data[t] * (1.0 if (pos[i] % vec_width) != (pos[j] % vec_width) else 0.0)
    return bad / total_w


def stability_penalty(pi: List[int], pi_prev: List[int], h: int) -> float:
    if not pi_prev or len(pi_prev) != len(pi):
        return 0.0
    pos = invperm(pi)
    pos_prev = invperm(pi_prev)
    n = len(pi)
    moves = sum(1 for i in range(n) if abs(pos[i] - pos_prev[i]) > h)
    return moves / float(n)


def eval_cost(W: CSRMatrix, pi: List[int], pi_prev: List[int] | None, cfg: CostConfig) -> Dict[str, float]:
    Cn = cache_cost(W, pi, cfg)
    Qn = modularity_contiguous_blocks(W, pi, cfg)
    R_align = align_penalty(W, invperm(pi), cfg.vec_width)
    R_stab = stability_penalty(pi, pi_prev or [], cfg.hysteresis)
    J = cfg.alpha * Cn + cfg.beta * R_align + cfg.gamma_stability * R_stab - cfg.mu * Qn
    return {"C": Cn, "Q": Qn, "R_align": R_align, "R_stab": R_stab, "J": J}


__all__ = [
    "CostConfig",
    "invperm",
    "cache_cost",
    "modularity_contiguous_blocks",
    "align_penalty",
    "stability_penalty",
    "eval_cost",
]

