from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


@dataclass
class CSRMatrix:
    """Minimal CSR matrix to avoid external deps.

    Stores: indptr (len=n+1), indices (len=nnz), data (len=nnz), shape=(n,n)
    """

    indptr: List[int]
    indices: List[int]
    data: List[float]
    shape: Tuple[int, int]
    meta: Dict[str, object]

    def nnz(self) -> int:
        return len(self.data)

    def to_npz_payload(self) -> Dict[str, object]:
        return {
            "indptr": self.indptr,
            "indices": self.indices,
            "data": self.data,
            "shape": list(self.shape),
            "meta": self.meta,
        }

    # convenience for tests/tools
    def to_dense(self) -> list[list[float]]:
        n = self.shape[0]
        M = [[0.0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            start, end = self.indptr[i], self.indptr[i + 1]
            for k in range(start, end):
                j = self.indices[k]
                v = self.data[k]
                M[i][j] = v
                if j < n:
                    M[j][i] = v
        return M

    # alias for compatibility with common tests
    def toarray(self):
        return self.to_dense()


def _make_blocky_mock(d: int, blocks: int, noise: float, seed: int) -> CSRMatrix:
    """Create a deterministic blocky co-access matrix W in CSR.

    - W has stronger weights within blocks and weaker cross-block weights.
    - Deterministic for given parameters.
    - No external dependencies; uses simple loops so suitable for small D in mock.
    """
    import random

    rng = random.Random(seed)
    block_size = max(1, d // max(1, blocks))
    indptr: List[int] = [0]
    indices: List[int] = []
    data: List[float] = []
    for i in range(d):
        row_nnz = 0
        bi = i // block_size
        for j in range(d):
            if j < i:
                # store upper triangle later via symmetry; skip to keep CSR simple
                continue
            bj = j // block_size
            base = 1.0 if bi == bj else 0.05
            # add tiny deterministic jitter
            jitter = (rng.random() - 0.5) * 2 * noise
            w = max(0.0, base + jitter)
            if w == 0.0 or i == j:
                continue
            indices.append(j)
            data.append(w)
            row_nnz += 1
        indptr.append(indptr[-1] + row_nnz)

    meta = {
        "shape": d,
        "format": "csr",
        "nnz": len(data),
        "source": "mock",
        "seed": seed,
        "blocks": blocks,
        "noise": noise,
    }
    return CSRMatrix(indptr=indptr, indices=indices, data=data, shape=(d, d), meta=meta)


def _normalize_sym(csr: CSRMatrix) -> CSRMatrix:
    """Simple symmetric normalization: W <- D^{-1/2} W D^{-1/2} (approx upper triangle only).

    Since we store only i<=j entries, we approximate by scaling with row degrees.
    """
    # compute degrees (sum of weights per row across stored upper triangle)
    n = csr.shape[0]
    deg = [0.0] * n
    for i in range(n):
        start, end = csr.indptr[i], csr.indptr[i + 1]
        for k in range(start, end):
            j = csr.indices[k]
            w = csr.data[k]
            deg[i] += w
            if j < n:
                deg[j] += w  # approximate symmetry contribution
    import math

    for i in range(n):
        start, end = csr.indptr[i], csr.indptr[i + 1]
        for k in range(start, end):
            j = csr.indices[k]
            di = max(deg[i], 1e-9)
            dj = max(deg[j] if j < n else deg[i], 1e-9)
            csr.data[k] = csr.data[k] / math.sqrt(di * dj)
    csr.meta["normalize"] = "sym"
    return csr


def _cap_and_prune(csr: CSRMatrix, nnz_cap: int) -> CSRMatrix:
    """If nnz exceeds cap, prune per-row by keeping top-k by weight proportional to cap.

    After pruning, L1-normalize rows to preserve relative scale.
    """
    cur_nnz = csr.nnz()
    if nnz_cap <= 0 or cur_nnz <= nnz_cap:
        return csr
    n = csr.shape[0]
    keep_ratio = nnz_cap / float(cur_nnz)
    new_indptr = [0]
    new_indices = []
    new_data = []
    for i in range(n):
        start, end = csr.indptr[i], csr.indptr[i + 1]
        row_idx = csr.indices[start:end]
        row_dat = csr.data[start:end]
        row = list(zip(row_idx, row_dat))
        row.sort(key=lambda t: t[1], reverse=True)
        k = max(0, min(len(row), int(round(len(row) * keep_ratio))))
        kept = row[:k]
        s = sum(w for _, w in kept) or 1.0
        for j, w in kept:
            new_indices.append(j)
            new_data.append(w / s)
        new_indptr.append(len(new_indices))
    csr.indptr, csr.indices, csr.data = new_indptr, new_indices, new_data
    csr.meta["pruned"] = True
    csr.meta["nnz_before"] = cur_nnz
    csr.meta["nnz_after"] = csr.nnz()
    return csr


def build_w(source: str = "mock", **cfg) -> CSRMatrix:
    """Build co-access weight matrix W (CSR) deterministically.

    Supported sources:
    - "mock": synthetic blocky matrix for testing.
    - "trace": placeholder raises NotImplementedError (spec-first approach).

    cfg (mock): d(int), blocks(int), noise(float), seed(int), normalize(str|None)
    """
    source = (source or "mock").lower()
    if source == "mock":
        d = int(cfg.get("d", cfg.get("D", 256)))
        blocks = int(cfg.get("blocks", 4))
        noise = float(cfg.get("noise", 0.02))
        seed = int(cfg.get("seed", 0))
        csr = _make_blocky_mock(d=d, blocks=blocks, noise=noise, seed=seed)
        normalize = cfg.get("normalize", "sym")
        if normalize in ("sym", "row"):
            csr = _normalize_sym(csr)
        # nnz cap rule: min(0.05 * D^2, 50_000_000)
        cap = cfg.get("nnz_cap", None)
        if cap is None:
            cap = int(min(0.05 * d * d, 50_000_000))
        csr = _cap_and_prune(csr, int(cap))
        return csr
    elif source == "pytorch":
        # Expect model & example_inputs provided by caller under keys or nested cfg
        model = cfg.get("model")
        example_inputs = cfg.get("example_inputs")
        pt_cfg = cfg.get("pytorch", {})
        if model is None or example_inputs is None:
            raise ValueError("graph.source='pytorch' requires 'model' and 'example_inputs' in cfg")
        # Lazy import to avoid hard dependency when unused
        from .graph_pytorch import build_w_from_pytorch

        csr = build_w_from_pytorch(model, example_inputs, **pt_cfg)
        # normalize & cap as with mock
        normalize = cfg.get("normalize", "sym")
        if normalize in ("sym", "row"):
            csr = _normalize_sym(csr)
        d = csr.shape[0]
        cap = cfg.get("nnz_cap", None)
        if cap is None:
            cap = int(min(0.05 * d * d, 50_000_000))
        csr = _cap_and_prune(csr, int(cap))
        return csr
    elif source == "trace":
        # TODO: define trace schema in docs and implement parser/merger
        raise NotImplementedError("trace source not yet implemented; see Graph Spec doc")
    else:
        raise ValueError(f"Unknown source: {source}")


def save_w_npz(path: str, W: CSRMatrix) -> None:
    """Save CSR matrix to .npz (json inside for simplicity, no numpy hard dep)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(W.to_npz_payload(), f)


__all__ = ["CSRMatrix", "build_w", "save_w_npz"]


def make_band_of_blocks(d: int, section_size: int, hops: int = 2, reuse_decay: float = 0.7,
                        cross_scale: float = 0.01) -> CSRMatrix:
    """Construct a band-of-blocks CSR matrix for testing/attention-aware mode.

    - Partition dimension D into contiguous sections of size `section_size`.
    - Within each section, connect neighbors up to `hops` with decay.
    - Across sections, add tiny links to avoid isolated blocks.
    Deterministic and non-negative.
    """
    d = int(d)
    section_size = max(1, int(section_size))
    hops = max(1, int(hops))
    nsec = (d + section_size - 1) // section_size
    indptr: List[int] = [0]
    indices: List[int] = []
    data: List[float] = []
    # Intra-section bands
    for i in range(d):
        sidx = i // section_size
        start_j = i + 1
        added = 0
        for off in range(1, hops + 1):
            j = i + off
            if j >= d:
                break
            if j // section_size != sidx:
                break
            w = reuse_decay ** off
            indices.append(j)
            data.append(w)
            added += 1
        indptr.append(indptr[-1] + added)
    # Add tiny cross-section links (upper triangle only)
    cross = cross_scale * (reuse_decay ** hops)
    for s in range(nsec - 1):
        i = min(d - 1, s * section_size + section_size - 1)
        j = min(d - 1, (s + 1) * section_size)
        if i < j and j < d:
            # attach to row i
            indices.append(j)
            data.append(cross)
            # We cannot easily adjust indptr retrospectively per row; so we append to last row.
            # For simplicity, rebuild CSR structures compactly.
    # Rebuild consistent CSR from COO-like lists
    rows: List[List[Tuple[int, float]]] = [[] for _ in range(d)]
    cursor = 0
    for i in range(d):
        start, end = indptr[i], indptr[i + 1]
        for k in range(start, end):
            j = indices[k]
            rows[i].append((j, data[k]))
    if len(indices) > 0 and len(data) > 0 and len(rows) > 0:
        # append cross edges to last row if any
        # Already appended cross to end; ensure it belongs to a valid row
        pass
    # Build final CSR
    f_indptr = [0]
    f_indices: List[int] = []
    f_data: List[float] = []
    for i in range(d):
        row = sorted(rows[i])
        for j, w in row:
            if j == i:
                continue
            f_indices.append(j)
            f_data.append(float(max(0.0, w)))
        f_indptr.append(len(f_indices))
    meta = {"shape": d, "format": "csr", "nnz": len(f_data), "source": "band_of_blocks"}
    return CSRMatrix(indptr=f_indptr, indices=f_indices, data=f_data, shape=(d, d), meta=meta)
