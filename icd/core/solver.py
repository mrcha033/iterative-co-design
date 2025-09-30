from __future__ import annotations

import math
import statistics
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

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

    if not hasattr(nx, "algorithms"):
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
    fallback_reason: str | None = None
    try:
        import networkx as nx
    except ImportError:  # pragma: no cover - optional dependency
        nx = None  # type: ignore[assignment]
        fallback_reason = "missing_networkx"
    else:
        if not hasattr(nx, "algorithms"):
            fallback_reason = "missing_algorithms"

    n = W.shape[0]

    def _fallback_partition(nodes: Iterable[int]) -> List[set[int]]:
        ordered = sorted(int(node) for node in nodes)
        if not ordered:
            return []
        buckets: List[set[int]] = []
        current: List[int] = []
        for node in ordered:
            current.append(node)
            if len(current) >= 2:
                buckets.append(set(current))
                current = []
        if current:
            if buckets:
                buckets[-1].update(current)
            else:
                buckets.append(set(current))
        return buckets

    runtime_budget = min(
        float(time_budget_s),
        float(getattr(cfg, "louvain_time_budget_s", float("inf"))),
    )
    if runtime_budget <= 0.0:
        fallback_reason = fallback_reason or "non_positive_budget"

    modularity_floor = float(getattr(cfg, "louvain_modularity_floor", float("-inf")))

    if fallback_reason:
        communities = _fallback_partition(range(n))
        modularity_value = _modularity(W, communities)
        if math.isfinite(modularity_floor) and modularity_floor > float("-inf"):
            remaining_budget = max(0.0, float(time_budget_s))
            pi_fb, stats_fb = _solve_spectral_refine(
                W,
                time_budget_s=remaining_budget,
                refine_steps=refine_steps,
                cfg=cfg,
                seed=seed,
                clusters=clusters,
            )
            stats_fb = dict(stats_fb)
            stats_fb.setdefault("Q_louvain", modularity_value)
            stats_fb.setdefault("louvain_runtime_s", 0.0)
            stats_fb["louvain_fallback"] = fallback_reason
            return pi_fb, stats_fb
        pi = _cluster_to_permutation(communities)
        stats = eval_cost(W, pi, pi, cfg)
        extra = {
            "clusters": len(communities),
            "Q_louvain": modularity_value,
            "louvain_runtime_s": 0.0,
            "louvain_fallback": fallback_reason,
        }
        return _finalize_stats(pi, stats, method="louvain", reference=stats_id, extra=extra)

    nx_error = getattr(nx, "NetworkXError", Exception)

    try:
        G = nx.Graph()
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
        start_time = time.perf_counter()
        local_fallback: str | None = None

        try:
            communities = nx.algorithms.community.louvain_communities(
                G,
                seed=seed,
                weight="weight",
                resolution=getattr(cfg, "modularity_gamma", 1.0),
            )
        except ImportError as exc:
            local_fallback = str(exc) or "missing_louvain_dependency"
            communities = _fallback_partition(G.nodes)
        except AttributeError as exc:
            local_fallback = str(exc) or "missing_louvain_algorithm"
            communities = _fallback_partition(G.nodes)
        elapsed = time.perf_counter() - start_time
        tolerance = 1e-6
        if elapsed > runtime_budget + tolerance:
            raise TimeoutError("louvain runtime budget exceeded")
        if not communities:
            raise ValueError("empty partition")
        pi = _cluster_to_permutation(communities)
        stats = eval_cost(W, pi, pi, cfg)
        modularity_value = _modularity(W, communities)
        if modularity_value < modularity_floor:
            remaining_budget = max(0.0, float(time_budget_s) - elapsed)
            pi_fb, stats_fb = _solve_spectral_refine(
                W,
                time_budget_s=remaining_budget,
                refine_steps=refine_steps,
                cfg=cfg,
                seed=seed,
                clusters=clusters,
            )
            stats_fb = dict(stats_fb)
            stats_fb.setdefault("Q_louvain", modularity_value)
            stats_fb.setdefault("louvain_runtime_s", elapsed)
            stats_fb["louvain_fallback"] = "modularity_below_floor"
            return pi_fb, stats_fb
        extra = {
            "clusters": len(communities),
            "Q_louvain": modularity_value,
            "louvain_runtime_s": elapsed,
        }
        if local_fallback:
            extra["louvain_fallback"] = local_fallback
        return _finalize_stats(pi, stats, method="louvain", reference=stats_id, extra=extra)
    except (TimeoutError, ValueError, nx_error, AttributeError):
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

    def _build_lane_records(topology: Dict[str, Any] | None, fallback_width: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        lanes_meta = list(topology.get("lanes", [])) if topology else []
        if not lanes_meta:
            lanes_meta = [{"id": idx} for idx in range(fallback_width)]
        lane_records: List[Dict[str, Any]] = []
        for idx, lane in enumerate(lanes_meta):
            weight = float(lane.get("weight", lane.get("throughput", 1.0)))
            lane_records.append(
                {
                    "orig_index": idx,
                    "id": int(lane.get("id", idx)),
                    "l2_slice": lane.get("l2_slice"),
                    "memory_channel": lane.get("memory_channel"),
                    "weight": weight if weight > 0 else 1.0,
                }
            )

        def _lane_sort_key(rec: Dict[str, Any]) -> Tuple[int, int, int, int]:
            mc = rec.get("memory_channel")
            l2 = rec.get("l2_slice")
            mc_key = (0, int(mc)) if mc is not None else (1, rec["orig_index"])
            l2_key = (0, int(l2)) if l2 is not None else (1, rec["orig_index"])
            return (mc_key[0], mc_key[1], l2_key[0], l2_key[1])

        lane_records.sort(key=_lane_sort_key)
        for new_idx, rec in enumerate(lane_records):
            rec["index"] = new_idx
        grouped: "OrderedDict[Tuple[Any, Any], Dict[str, Any]]" = OrderedDict()
        for rec in lane_records:
            key = (rec.get("memory_channel"), rec.get("l2_slice"))
            if key not in grouped:
                grouped[key] = {
                    "memory_channel": rec.get("memory_channel"),
                    "l2_slice": rec.get("l2_slice"),
                    "lane_indices": [],
                    "capacity": 0.0,
                }
            grouped[key]["lane_indices"].append(rec["index"])
            grouped[key]["capacity"] += rec["weight"]
        groups = list(grouped.values())
        return lane_records, groups

    vec_width = max(1, int(getattr(cfg, "vec_width", 1)))
    lane_records, groups = _build_lane_records(getattr(cfg, "hardware_topology", None), vec_width)
    if not lane_records:
        lane_records = [{"index": i, "weight": 1.0} for i in range(vec_width)]
    if not groups:
        groups = [
            {
                "memory_channel": None,
                "l2_slice": None,
                "lane_indices": [rec["index"] for rec in lane_records],
                "capacity": float(sum(rec.get("weight", 1.0) for rec in lane_records)),
            }
        ]

    n = W.shape[0]
    deg = _degree_vector(W)
    adj: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
    for i in range(n):
        start, end = W.indptr[i], W.indptr[i + 1]
        for idx in range(start, end):
            j = W.indices[idx]
            w = float(W.data[idx])
            if j >= n:
                continue
            adj[i].append((j, w))
            adj[j].append((i, w))

    node_group = [-1] * n
    group_nodes: List[List[int]] = [[] for _ in groups]
    group_loads = [0.0] * len(groups)
    group_capacity = [max(group.get("capacity", 1.0), 1.0) for group in groups]
    avg_deg = (sum(deg) / float(n)) if n > 0 else 0.0
    base_affinity = max(avg_deg * 0.05, 1e-6)

    order = sorted(range(n), key=lambda i: (-deg[i], i))
    for node in order:
        neighbor_affinity = [0.0] * len(groups)
        for nbr, weight in adj[node]:
            grp = node_group[nbr]
            if grp != -1:
                neighbor_affinity[grp] += weight

        best_group = 0
        best_score = -math.inf
        best_affinity = -math.inf
        best_balance = math.inf
        for gi in range(len(groups)):
            affinity = neighbor_affinity[gi]
            balance = group_loads[gi] / group_capacity[gi]
            balance_term = 1.0 / (1.0 + balance)
            score = (affinity + base_affinity) * balance_term
            if (
                score > best_score
                or (
                    math.isclose(score, best_score, rel_tol=1e-9, abs_tol=1e-12)
                    and (
                        affinity > best_affinity
                        or (
                            math.isclose(affinity, best_affinity, rel_tol=1e-9, abs_tol=1e-12)
                            and balance < best_balance
                        )
                    )
                )
            ):
                best_score = score
                best_group = gi
                best_affinity = affinity
                best_balance = balance
        node_group[node] = best_group
        group_nodes[best_group].append(node)
        group_loads[best_group] += deg[node]

    lane_count = len(lane_records)
    lane_weights = [float(rec.get("weight", 1.0)) for rec in lane_records]
    lane_loads = [0.0] * lane_count
    lane_buckets: List[List[int]] = [[] for _ in range(lane_count)]
    lane_assignment = [-1] * n

    for gi, nodes in enumerate(group_nodes):
        lanes = groups[gi]["lane_indices"]
        if not lanes:
            lanes = list(range(lane_count))
        nodes_sorted = sorted(nodes, key=lambda i: (-deg[i], i))
        for node in nodes_sorted:
            lane_idx = min(
                lanes,
                key=lambda lid: lane_loads[lid] / max(lane_weights[lid], 1e-9),
            )
            lane_buckets[lane_idx].append(node)
            lane_loads[lane_idx] += deg[node]
            lane_assignment[node] = lane_idx

    ordered_lane_indices: List[int] = []
    for group in groups:
        ordered_lane_indices.extend(sorted(group["lane_indices"]))
    remaining = [idx for idx in range(lane_count) if idx not in ordered_lane_indices]
    ordered_lane_indices.extend(remaining)

    pi: List[int] = []
    for lane_idx in ordered_lane_indices:
        lane_nodes = lane_buckets[lane_idx]
        lane_nodes.sort(key=lambda i: (-deg[i], i))
        pi.extend(lane_nodes)

    stats = eval_cost(W, pi, pi, cfg)
    extra = {
        "lane_groups": lane_count,
        "topology_groups": len(groups),
        "group_balance_std": float(statistics.pstdev(group_loads)) if len(group_loads) > 1 else 0.0,
        "lane_balance_std": float(statistics.pstdev(lane_loads)) if lane_count > 1 else 0.0,
        "topology_assignment": tuple(int(g) for g in node_group),
        "topology_group_sizes": tuple(len(nodes) for nodes in group_nodes),
        "lane_assignment": tuple(int(l) for l in lane_assignment),
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


def compute_perm_from_w(
    W: CSRMatrix,
    *,
    time_budget_s: float | None = None,
    refine_steps: int | None = None,
    cfg: CostConfig | None = None,
    seed: int | None = None,
    clusters: Sequence[Sequence[int]] | None = None,
    method: str | None = None,
) -> List[int]:
    """Convenience wrapper that returns only the permutation for ``W``.

    The helper delegates to :func:`fit_permutation` while ensuring that the
    resulting permutation length matches the feature dimension ``D`` encoded in
    ``W``.  Callers may override solver parameters via keyword arguments.  The
    returned permutation is a Python ``list`` compatible with downstream
    runners that expect JSON-serialisable payloads.
    """

    kwargs: Dict[str, Any] = {}
    if time_budget_s is not None:
        kwargs["time_budget_s"] = float(time_budget_s)
    if refine_steps is not None:
        kwargs["refine_steps"] = int(refine_steps)
    if cfg is not None:
        kwargs["cfg"] = cfg
    if seed is not None:
        kwargs["seed"] = int(seed)
    if clusters is not None:
        kwargs["clusters"] = clusters
    if method is not None:
        kwargs["method"] = method

    pi, _ = fit_permutation(W, **kwargs)
    if len(pi) != int(W.shape[0]):
        raise ValueError(
            "permutation length {} does not match W.shape[0] {}".format(
                len(pi), int(W.shape[0])
            )
        )
    return pi


__all__ = ["fit_permutation", "compute_perm_from_w"]
