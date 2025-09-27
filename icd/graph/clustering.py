"""Community detection utilities for correlation graphs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import math
import time

import networkx as nx
from networkx.algorithms import community

from icd.core.graph import CSRMatrix

__all__ = ["ClusteringConfig", "cluster_graph"]


@dataclass
class ClusteringConfig:
    method: str = "louvain"
    rng_seed: int = 0
    resolution: float = 1.0
    fallback_method: str = "spectral"
    modularity_floor: float = 0.35
    runtime_budget: float | None = None
    last_meta: dict[str, object] = field(default_factory=dict, init=False, repr=False)


def _to_networkx(W: CSRMatrix) -> nx.Graph:
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
    return G


def _louvain(G: nx.Graph, cfg: ClusteringConfig) -> List[List[int]]:
    import random

    rng = random.Random(cfg.rng_seed)
    parts = community.louvain_communities(
        G,
        weight="weight",
        seed=rng.randint(0, 2**32 - 1),
        resolution=cfg.resolution,
    )
    return [sorted(list(p)) for p in parts]


def _spectral(G: nx.Graph, cfg: ClusteringConfig) -> List[List[int]]:
    nodes = sorted(G.nodes())
    k = max(2, int(cfg.resolution))
    clusters: List[List[int]] = []
    chunk = max(1, int(math.ceil(len(nodes) / k)))
    for idx in range(0, len(nodes), chunk):
        clusters.append(nodes[idx : idx + chunk])
    return [cluster for cluster in clusters if cluster]


def _dispatch(G: nx.Graph, cfg: ClusteringConfig, method: str) -> List[List[int]]:
    if method == "louvain":
        return _louvain(G, cfg)
    if method == "spectral":
        return _spectral(G, cfg)
    raise ValueError(f"Unknown clustering method: {method}")


def cluster_graph(W: CSRMatrix, cfg: ClusteringConfig) -> List[List[int]]:
    G = _to_networkx(W)
    start = time.perf_counter()
    clusters = _dispatch(G, cfg, cfg.method)
    duration = time.perf_counter() - start
    modularity = None
    if clusters:
        modularity = community.modularity(G, [set(c) for c in clusters])

    used_method = cfg.method
    fallback_reason = None

    if cfg.fallback_method and cfg.fallback_method != cfg.method:
        low_modularity = modularity is not None and modularity < cfg.modularity_floor
        over_budget = cfg.runtime_budget is not None and duration > cfg.runtime_budget
        if low_modularity or over_budget:
            used_method = cfg.fallback_method
            fallback_reason = "low_modularity" if low_modularity else "runtime_budget"
            clusters = _dispatch(G, cfg, cfg.fallback_method)
            if clusters:
                modularity = community.modularity(G, [set(c) for c in clusters])

    cfg.last_meta = {
        "method": used_method,
        "fallback_reason": fallback_reason,
        "runtime_s": duration,
        "modularity": modularity,
    }
    return clusters

