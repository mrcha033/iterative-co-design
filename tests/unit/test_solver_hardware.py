from icd.core.cost import CostConfig
from icd.core.graph import CSRMatrix
from icd.core.solver import fit_permutation


def _make_heterogeneous_csr(group_sizes, intra_weight=4.0, inter_weight=0.1):
    n = sum(group_sizes)
    indptr = [0]
    indices = []
    data = []
    group_boundaries = []
    acc = 0
    for size in group_sizes:
        group_boundaries.append((acc, acc + size))
        acc += size
    for i in range(n):
        start_idx = next(start for start, end in group_boundaries if start <= i < end)
        for j in range(i + 1, n):
            group_j = next(start for start, end in group_boundaries if start <= j < end)
            weight = intra_weight if start_idx == group_j else inter_weight
            indices.append(j)
            data.append(float(weight))
        indptr.append(len(indices))
    return CSRMatrix(indptr=indptr, indices=indices, data=data, shape=(n, n), meta={})


def test_hardware_topology_groups_follow_cache_and_memory_affinity():
    W = _make_heterogeneous_csr([4, 4])
    topology = {
        "lanes": [
            {"id": 0, "l2_slice": 0, "memory_channel": 0},
            {"id": 1, "l2_slice": 0, "memory_channel": 0},
            {"id": 2, "l2_slice": 1, "memory_channel": 1},
            {"id": 3, "l2_slice": 1, "memory_channel": 1},
        ]
    }
    cfg = CostConfig(vec_width=4, hardware_topology=topology)
    pi, stats = fit_permutation(W, cfg=cfg, method="hardware")

    assert stats["method"] == "hardware_aware"
    assert stats["topology_groups"] == 2

    assignment = stats.get("topology_assignment")
    lane_assignment = stats.get("lane_assignment")
    assert assignment is not None and lane_assignment is not None

    group_to_nodes = {}
    group_to_lanes = {}
    for node, group in enumerate(assignment):
        group_to_nodes.setdefault(group, []).append(node)
        group_to_lanes.setdefault(group, set()).add(lane_assignment[node])

    observed_node_sets = {frozenset(nodes) for nodes in group_to_nodes.values()}
    expected_node_sets = {frozenset(range(4)), frozenset(range(4, 8))}
    assert observed_node_sets == expected_node_sets

    observed_lane_sets = {frozenset(lanes) for lanes in group_to_lanes.values()}
    expected_lane_sets = {frozenset({0, 1}), frozenset({2, 3})}
    assert observed_lane_sets == expected_lane_sets


def test_hardware_topology_capacity_balances_groups():
    W = _make_heterogeneous_csr([4, 2])
    topology = {
        "lanes": [
            {"id": 0, "l2_slice": 0, "memory_channel": 0},
            {"id": 1, "l2_slice": 0, "memory_channel": 0},
            {"id": 2, "l2_slice": 1, "memory_channel": 1},
        ]
    }
    cfg = CostConfig(vec_width=3, hardware_topology=topology)
    _, stats = fit_permutation(W, cfg=cfg, method="hardware")

    group_sizes = stats.get("topology_group_sizes")
    assert group_sizes is not None
    assert group_sizes[0] >= group_sizes[1]
    assert sum(group_sizes) == 6

    group_std = stats.get("group_balance_std")
    assert group_std is not None
    assert group_std >= 0
