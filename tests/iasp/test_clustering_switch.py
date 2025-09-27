import math
import sys
import types
from importlib.machinery import SourceFileLoader
from pathlib import Path

from icd.core.graph import CSRMatrix

import icd  # noqa: F401  # ensure package registered


class _FakeGraph:
    def __init__(self) -> None:
        self._nodes: set[int] = set()

    def add_nodes_from(self, nodes) -> None:
        self._nodes.update(nodes)

    def add_edge(self, u: int, v: int, weight: float) -> None:  # pragma: no cover - structure only
        self.add_nodes_from([u, v])

    def nodes(self):
        return list(sorted(self._nodes))


def _fake_louvain(G, weight=None, seed=None, resolution=None):  # pragma: no cover - deterministic
    return [[n] for n in G.nodes()]


def _fake_modularity(G, communities):  # pragma: no cover - deterministic
    return 0.5


networkx = types.ModuleType("networkx")
networkx.Graph = _FakeGraph  # type: ignore[attr-defined]

community_mod = types.ModuleType("networkx.algorithms.community")
community_mod.louvain_communities = _fake_louvain  # type: ignore[attr-defined]
community_mod.modularity = _fake_modularity  # type: ignore[attr-defined]

algorithms_mod = types.ModuleType("networkx.algorithms")
algorithms_mod.community = community_mod  # type: ignore[attr-defined]

sys.modules.setdefault("networkx", networkx)
sys.modules.setdefault("networkx.algorithms", algorithms_mod)
sys.modules.setdefault("networkx.algorithms.community", community_mod)

graph_pkg = sys.modules.setdefault("icd.graph", types.ModuleType("icd.graph"))
graph_pkg.__path__ = []  # type: ignore[attr-defined]
setattr(icd, "graph", graph_pkg)

_clustering_path = Path(__file__).resolve().parents[2] / "icd" / "graph" / "clustering.py"
_loader = SourceFileLoader("icd.graph.clustering", str(_clustering_path))
_clustering_mod = types.ModuleType("icd.graph.clustering")
sys.modules["icd.graph.clustering"] = _clustering_mod
_loader.exec_module(_clustering_mod)
graph_pkg.clustering = _clustering_mod  # type: ignore[attr-defined]

ClusteringConfig = _clustering_mod.ClusteringConfig
cluster_graph = _clustering_mod.cluster_graph


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
    assert cfg.last_meta["modularity"] is not None
    assert cfg.last_meta["runtime_s"] >= 0.0
    assert clusters


def test_low_modularity_triggers_fallback_and_records_meta():
    W = _random_graph(8)
    cfg = ClusteringConfig(
        method="louvain",
        fallback_method="spectral",
        modularity_floor=0.99,
    )

    clusters = cluster_graph(W, cfg)

    assert clusters
    assert cfg.last_meta["method"] == "spectral"
    assert cfg.last_meta["fallback_reason"] == "low_modularity"
    assert cfg.last_meta["modularity"] is not None
    assert cfg.last_meta["runtime_s"] >= 0.0

