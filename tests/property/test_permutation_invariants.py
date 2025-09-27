import math
from typing import List

import hypothesis.strategies as st
import numpy as np
from hypothesis import given, settings

from icd.core.cost import CostConfig, eval_cost
from icd.core.graph import CSRMatrix
from icd.core.solver import fit_permutation


def _dense_to_csr(matrix: np.ndarray) -> CSRMatrix:
    n = matrix.shape[0]
    indptr: List[int] = [0]
    indices: List[int] = []
    data: List[float] = []
    for i in range(n):
        row_nnz = 0
        for j in range(i + 1, n):
            val = float(matrix[i, j])
            if math.isclose(val, 0.0, abs_tol=1e-9):
                continue
            indices.append(j)
            data.append(val)
            row_nnz += 1
        indptr.append(indptr[-1] + row_nnz)
    return CSRMatrix(indptr=indptr, indices=indices, data=data, shape=(n, n), meta={})


@st.composite
def symmetric_csr(draw):
    size = draw(st.integers(min_value=4, max_value=9))
    upper = draw(st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=size * (size - 1) // 2, max_size=size * (size - 1) // 2))
    mat = np.zeros((size, size), dtype=float)
    idx = 0
    for i in range(size):
        for j in range(i + 1, size):
            val = upper[idx]
            idx += 1
            mat[i, j] = val
            mat[j, i] = val
    return _dense_to_csr(mat)


@settings(max_examples=25, deadline=None)
@given(symmetric_csr())
def test_fit_permutation_returns_valid_permutation(W: CSRMatrix):
    pi, stats = fit_permutation(W, time_budget_s=0.05, refine_steps=32, seed=3)
    assert sorted(pi) == list(range(W.shape[0]))
    assert len(set(pi)) == W.shape[0]
    assert "J" in stats


@settings(max_examples=20, deadline=None)
@given(symmetric_csr())
def test_iterative_cost_never_worse_than_identity(W: CSRMatrix):
    cfg = CostConfig()
    identity = list(range(W.shape[0]))
    stats_identity = eval_cost(W, identity, identity, cfg)
    pi, stats = fit_permutation(W, time_budget_s=0.05, refine_steps=64, seed=7)
    stats_perm = eval_cost(W, pi, pi, cfg)
    assert stats_perm["J"] <= stats_identity["J"] + 1e-6
    if stats.get("Q_final") is not None and stats.get("Q_cluster") is not None:
        assert stats["Q_final"] >= stats["Q_cluster"] - 1e-6
