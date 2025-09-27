from __future__ import annotations

import math
import statistics
import time
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

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


def _degree_vector(W: CSRMatrix) -> List[float]:
    n = W.shape[0]
    deg = [0.0] * n
    for i in range(n):
        start, end = W.indptr[i], W.indptr[i + 1]
        for idx in range(start, end):
            j = W.indices[idx]
            w = float(W.data[idx])
            deg[i] += w
            if j < n:
                deg[j] += w
    return deg


def _local_refine_adjacent(
    W: CSRMatrix,
    pi: List[int],
    cfg: CostConfig,
    steps: int,
    time_budget_s: float,
) -> Tuple[List[int], bool]:
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


def _cluster_to_permutation(clusters: Sequence[Sequence[int]]) -> List[int]:
    permutation: List[int] = []
    for cluster in clusters:
        permutation.extend(int(node) for node in cluster)
    return permutation


def _modularity(W: CSRMatrix, clusters: Sequence[Sequence[int]]) -> float:
    try:
        import networkx as nx
    except ImportError:  # pragma: no cover
        return 0.0

    G = nx.Graph()
    n = W.shape[0]
    G.add_nodes_from(range(n))
    for i in range(n):
        start, end = W.indptr[i], W.indptr[i + 1]
        for idx in range(start, end):
            j = W.indices[idx]
            if j <= i:
                continue
            w = W.data[idx]
            if w <= 0:
                continue
            G.add_edge(i, j, weight=w)
    partition = [set(cluster) for cluster in clusters]
    try:
        return float(nx.algorithms.community.quality.modularity(G, partition, weight="weight"))
    except ZeroDivisionError:
        return 0.0


def _clusters_from_order(order: List[int], sizes: Sequence[int]) -> List[List[int]]:
    clusters: List[List[int]] = []
    idx = 0
    for size in sizes:
        clusters.append(order[idx : idx + size])
        idx += size
    return clusters


def _identity_stats(W: CSRMatrix, cfg: CostConfig) -> Tuple[List[int], Dict[str, float]]:
    n = W.shape[0]
    pi_id = list(range(n))
    stats = eval_cost(W, pi_id, pi_id, cfg)
    stats.setdefault("clusters", 0)
    stats.setdefault("improved", False)
    stats["method"] = "identity"
    return pi_id, stats


def _finalize_stats(
    pi: List[int],
    stats: Dict[str, float],
    *,
    method: str,
    reference: Dict[str, float],
    extra: Dict[str, float] | None = None,
) -> Tuple[List[int], Dict[str, float]]:
    result = dict(stats)
    ref_J = reference.get("J", math.inf)
    result["method"] = method
    result["improved"] = bool(result.get("J", math.inf) < ref_J)
    result.setdefault("clusters", 0)
    if extra:
        result.update(extra)
    return pi, result


def _solve_spectral_refine(
    W: CSRMatrix,
    *,
    time_budget_s: float,
    refine_steps: int,
    cfg: CostConfig,
    seed: int,
    clusters: Sequence[Sequence[int]] | None,
) -> Tuple[List[int], Dict[str, float]]:
    pi_id, stats_id = _identity_stats(W, cfg)

    stats_cluster: Dict[str, float] = {}
    cluster_sizes: List[int] | None = None
    if clusters:
        pi0 = _cluster_to_permutation(clusters)
        stats0 = eval_cost(W, pi0, pi0, cfg)
        stats_cluster["Q_cluster"] = _modularity(W, clusters)
        cluster_sizes = [len(c) for c in clusters]
    else:
        pi0 = _spectral_init_like(W, seed=seed)
        stats0 = eval_cost(W, pi0, pi0, cfg)

    start_pi = pi0 if stats0["J"] <= stats_id["J"] else pi_id
    base_stats = stats0 if start_pi is pi0 else stats_id

    pi1, improved = _local_refine_adjacent(
        W,
        start_pi,
        cfg,
        steps=refine_steps,
        time_budget_s=time_budget_s,
    )
    stats1 = eval_cost(W, pi1, start_pi, cfg)

    if stats1["J"] > stats_id["J"]:
        pi1, stats1 = pi_id, stats_id
        improved = False

    extra: Dict[str, float] = {"clusters": len(clusters) if clusters else 0}
    if clusters and cluster_sizes is not None:
        extra.update(stats_cluster)
        final_clusters = _clusters_from_order(pi1, cluster_sizes)
        extra["Q_final"] = _modularity(W, final_clusters)
    if improved:
        extra["improved"] = True

    return _finalize_stats(pi1, stats1, method="spectral_refine", reference=base_stats, extra=extra)


def _solve_louvain(
    W: CSRMatrix,
    *,
    time_budget_s: float,
    refine_steps: int,
    cfg: CostConfig,
    seed: int,
    clusters: Sequence[Sequence[int]] | None,
) -> Tuple[List[int], Dict[str, float]]:
    pi_id, stats_id = _identity_stats(W, cfg)
    try:
        import networkx as nx

        G = nx.Graph()
        n = W.shape[0]
        G.add_nodes_from(range(n))
        for i in range(n):
            start, end = W.indptr[i], W.indptr[i + 1]
            for idx in range(start, end):
                j = W.indices[idx]
                if j <= i:
                    continue
                w = float(W.data[idx])
                if w <= 0:
                    continue
                G.add_edge(i, j, weight=w)
        if G.number_of_edges() == 0:
            raise ValueError("graph has no edges")
        communities = nx.algorithms.community.louvain_communities(
            G,
            seed=seed,
            weight="weight",
            resolution=getattr(cfg, "modularity_gamma", 1.0),
        )
        if not communities:
            raise ValueError("empty partition")
        pi = _cluster_to_permutation(communities)
        stats = eval_cost(W, pi, pi, cfg)
        extra = {
            "clusters": len(communities),
            "Q_louvain": _modularity(W, communities),
        }
        return _finalize_stats(pi, stats, method="louvain", reference=stats_id, extra=extra)
    except Exception:
        # fall back to spectral refine if Louvain is unavailable
        return _solve_spectral_refine(
            W,
            time_budget_s=time_budget_s,
            refine_steps=refine_steps,
            cfg=cfg,
            seed=seed,
            clusters=clusters,
        )


def _solve_memory_aware(
    W: CSRMatrix,
    *,
    time_budget_s: float,
    refine_steps: int,
    cfg: CostConfig,
    seed: int,
    clusters: Sequence[Sequence[int]] | None,
) -> Tuple[List[int], Dict[str, float]]:
    _ = (time_budget_s, refine_steps, clusters)
    _, stats_id = _identity_stats(W, cfg)
    deg = _degree_vector(W)
    order = sorted(range(len(deg)), key=lambda i: (-deg[i], i))
    blocks = max(1, int(getattr(cfg, "blocks_k", 1)))
    assignments: List[List[int]] = [[] for _ in range(blocks)]
    loads = [0.0] * blocks
    for node in order:
        idx = min(range(blocks), key=lambda b: loads[b])
        assignments[idx].append(node)
        loads[idx] += deg[node]
    pi: List[int] = []
    for block in assignments:
        pi.extend(block)
    stats = eval_cost(W, pi, pi, cfg)
    extra = {
        "clusters": blocks,
        "memory_balance_std": float(statistics.pstdev(loads)) if blocks > 1 else 0.0,
    }
    return _finalize_stats(pi, stats, method="memory_aware", reference=stats_id, extra=extra)


def _solve_hardware_aware(
    W: CSRMatrix,
    *,
    time_budget_s: float,
    refine_steps: int,
    cfg: CostConfig,
    seed: int,
    clusters: Sequence[Sequence[int]] | None,
) -> Tuple[List[int], Dict[str, float]]:
    _ = (time_budget_s, refine_steps, seed, clusters)
    _, stats_id = _identity_stats(W, cfg)
    vec_width = max(1, int(getattr(cfg, "vec_width", 1)))
    deg = _degree_vector(W)
    lanes: List[List[int]] = [[] for _ in range(vec_width)]
    for node in range(W.shape[0]):
        lane = node % vec_width
        lanes[lane].append(node)
    for lane_nodes in lanes:
        lane_nodes.sort(key=lambda i: (-deg[i], i))
    pi: List[int] = []
    for lane_nodes in lanes:
        pi.extend(lane_nodes)
    stats = eval_cost(W, pi, pi, cfg)
    extra = {
        "lane_groups": vec_width,
    }
    return _finalize_stats(pi, stats, method="hardware_aware", reference=stats_id, extra=extra)


_SOLVER_REGISTRY: Dict[str, Callable[..., Tuple[List[int], Dict[str, float]]]] = {
    "spectral": _solve_spectral_refine,
    "spectral_refine": _solve_spectral_refine,
    "default": _solve_spectral_refine,
    "louvain": _solve_louvain,
    "memory": _solve_memory_aware,
    "memory_aware": _solve_memory_aware,
    "hardware": _solve_hardware_aware,
    "hardware_aware": _solve_hardware_aware,
}


def fit_permutation(
    W: CSRMatrix,
    time_budget_s: float = 5.0,
    refine_steps: int = 1000,
    cfg: CostConfig | None = None,
    seed: int = 0,
    clusters: Sequence[Sequence[int]] | None = None,
    method: str = "spectral_refine",
) -> Tuple[List[int], Dict[str, float]]:
    """Compute permutation using configurable heuristics and return stats."""

    cfg = cfg or CostConfig()
    solver_key = str(method or "spectral_refine").lower()
    solver = _SOLVER_REGISTRY.get(solver_key)
    if solver is None:
        raise ValueError(f"Unknown solver method: {method}")
    return solver(
        W,
        time_budget_s=time_budget_s,
        refine_steps=refine_steps,
        cfg=cfg,
        seed=seed,
        clusters=clusters,
    )


__all__ = ["fit_permutation"]
