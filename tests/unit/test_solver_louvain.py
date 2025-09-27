import math
import sys
import types

import pytest

from icd.core.cost import CostConfig
from icd.core.graph import build_w
from icd.core.solver import fit_permutation


@pytest.fixture(autouse=True)
def stub_networkx(monkeypatch: pytest.MonkeyPatch) -> None:
    if "networkx" in sys.modules:
        return

    module = types.ModuleType("networkx")

    class Graph:
        def __init__(self) -> None:
            self._nodes: list[int] = []
            self._edges: list[tuple[int, int, float]] = []

        def add_nodes_from(self, nodes) -> None:
            for node in nodes:
                if node not in self._nodes:
                    self._nodes.append(int(node))

        def add_edge(self, u: int, v: int, weight: float = 1.0) -> None:
            self._edges.append((int(u), int(v), float(weight)))

        def number_of_edges(self) -> int:
            return len(self._edges)

        @property
        def nodes(self) -> list[int]:
            return list(self._nodes)

    module.Graph = Graph

    algorithms = types.ModuleType("networkx.algorithms")
    community = types.ModuleType("networkx.algorithms.community")
    quality = types.ModuleType("networkx.algorithms.community.quality")

    def louvain_communities(G: Graph, seed=None, weight=None, resolution: float = 1.0):
        nodes = sorted(G.nodes)
        if not nodes:
            return []
        split = max(1, len(nodes) // 2)
        return [set(nodes[:split]), set(nodes[split:])]

    def modularity(_G: Graph, _partition, weight=None) -> float:
        return 0.42

    quality.modularity = modularity
    community.louvain_communities = louvain_communities
    community.quality = quality
    algorithms.community = community
    module.algorithms = algorithms

    monkeypatch.setitem(sys.modules, "networkx", module)
    monkeypatch.setitem(sys.modules, "networkx.algorithms", algorithms)
    monkeypatch.setitem(sys.modules, "networkx.algorithms.community", community)
    monkeypatch.setitem(sys.modules, "networkx.algorithms.community.quality", quality)


def _make_graph():
    W = build_w(source="mock", D=128, blocks=4, noise=0.02, seed=1)
    return W


def test_louvain_accepts_when_thresholds_met() -> None:
    W = _make_graph()
    cfg = CostConfig(
        louvain_time_budget_s=math.inf,
        louvain_modularity_floor=-math.inf,
    )

    pi, stats = fit_permutation(
        W,
        time_budget_s=5.0,
        refine_steps=0,
        cfg=cfg,
        seed=7,
        method="louvain",
    )

    assert len(pi) == W.shape[0]
    assert stats["method"] == "louvain"
    assert stats.get("Q_louvain", -math.inf) >= cfg.louvain_modularity_floor
    assert stats.get("louvain_runtime_s", 0.0) >= 0.0


def test_louvain_falls_back_when_modularity_too_low() -> None:
    W = _make_graph()
    warm_cfg = CostConfig(louvain_time_budget_s=1.0, louvain_modularity_floor=-math.inf)
    _, warm_stats = fit_permutation(
        W,
        time_budget_s=5.0,
        refine_steps=0,
        cfg=warm_cfg,
        seed=11,
        method="louvain",
    )

    floor = float(warm_stats.get("Q_louvain", 0.0) + 1e-3)
    cfg = CostConfig(
        louvain_time_budget_s=1.0,
        louvain_modularity_floor=floor,
    )

    _, stats = fit_permutation(
        W,
        time_budget_s=5.0,
        refine_steps=0,
        cfg=cfg,
        seed=11,
        method="louvain",
    )

    assert stats["method"] == "spectral_refine"


def test_louvain_is_deterministic_for_seed() -> None:
    W = _make_graph()
    cfg = CostConfig(louvain_time_budget_s=1.0, louvain_modularity_floor=-0.5)

    pi_a, stats_a = fit_permutation(
        W,
        time_budget_s=5.0,
        refine_steps=0,
        cfg=cfg,
        seed=5,
        method="louvain",
    )

    pi_b, stats_b = fit_permutation(
        W,
        time_budget_s=5.0,
        refine_steps=0,
        cfg=cfg,
        seed=5,
        method="louvain",
    )

    assert pi_a == pi_b
    stats_a_clean = dict(stats_a)
    stats_b_clean = dict(stats_b)
    stats_a_clean.pop("louvain_runtime_s", None)
    stats_b_clean.pop("louvain_runtime_s", None)
    assert stats_a_clean == stats_b_clean
