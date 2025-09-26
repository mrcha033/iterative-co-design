import numpy as np
import torch

from icd.core.graph import CSRMatrix
from icd.core.solver import fit_permutation
from icd.graph.clustering import ClusteringConfig, cluster_graph


def _block_csr(n_block: int, size: int) -> CSRMatrix:
    data = []
    indices = []
    indptr = [0]
    for i in range(size):
        row = []
        for j in range(size):
            if i == j:
                continue
            same = i // n_block == j // n_block
            weight = 1.0 if same else 0.01
            if j > i:
                row.append((j, weight))
        row.sort()
        for j, w in row:
            indices.append(j)
            data.append(w)
        indptr.append(len(indices))
    return CSRMatrix(indptr=indptr, indices=indices, data=data, shape=(size, size), meta={})


def test_cluster_graph_louvain():
    W = _block_csr(n_block=2, size=8)
    cfg = ClusteringConfig(method="louvain", rng_seed=0)
    clusters = cluster_graph(W, cfg)
    assert len(clusters) >= 2
    lengths = sorted(len(c) for c in clusters)
    assert sum(lengths) == 8
    assert all(length >= 2 for length in lengths)


def test_fit_permutation_with_clusters():
    W = _block_csr(n_block=2, size=8)
    cfg = ClusteringConfig(method="louvain", rng_seed=0)
    clusters = cluster_graph(W, cfg)

    pi, stats = fit_permutation(W, clusters=clusters, time_budget_s=0.1, refine_steps=10)
    assert sorted(pi) == list(range(8))
    assert stats["clusters"] == len(clusters)
    assert stats["Q_cluster"] >= 0.0
    assert stats["Q_final"] >= stats["Q_cluster"]
