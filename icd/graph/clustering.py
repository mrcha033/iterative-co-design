"""Community detection utilities for correlation graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import networkx as nx
from networkx.algorithms import community

from icd.core.graph import CSRMatrix

__all__ = ["ClusteringConfig", "cluster_graph"]


@dataclass
class ClusteringConfig:
    method: str = "louvain"
    rng_seed: int = 0
    resolution: float = 1.0


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
    k = max(2, int(cfg.resolution))
    labels = community.spectral_clustering(G, k=k, weight="weight", seed=cfg.rng_seed)
    clusters: dict[int, list[int]] = {}
    for node, label in enumerate(labels):
        clusters.setdefault(int(label), []).append(node)
    return [sorted(v) for v in clusters.values()]


def cluster_graph(W: CSRMatrix, cfg: ClusteringConfig) -> List[List[int]]:
    G = _to_networkx(W)
    if cfg.method == "louvain":
        return _louvain(G, cfg)
    if cfg.method == "spectral":
        return _spectral(G, cfg)
    raise ValueError(f"Unknown clustering method: {cfg.method}")

