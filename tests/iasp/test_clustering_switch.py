import math

from icd.core.graph import CSRMatrix
from icd.graph.clustering import ClusteringConfig, cluster_graph


def _random_graph(n: int) -> CSRMatrix:
    data = []
    indices = []
    indptr = [0]
    for i in range(n):
        row = []
        for j in range(n):
            if j <= i:
                continue
            weight = 1.0 if (i + j) % 2 == 0 else 0.01
            row.append((j, weight))
        row.sort()
        for j, w in row:
            indices.append(j)
            data.append(w)
        indptr.append(len(indices))
    return CSRMatrix(indptr=indptr, indices=indices, data=data, shape=(n, n), meta={})


def test_runtime_budget_triggers_fallback(monkeypatch):
    W = _random_graph(10)
    cfg = ClusteringConfig(runtime_budget=0.0, fallback_method="spectral")

    times = iter([0.0, 10.0])

    def fake_perf_counter():
        return next(times, 10.0)

    monkeypatch.setattr("icd.graph.clustering.time.perf_counter", fake_perf_counter)

    clusters = cluster_graph(W, cfg)
    assert cfg.last_meta["method"] == "spectral"
    assert cfg.last_meta["fallback_reason"] == "runtime_budget"
    assert clusters

