import pytest

from icd.core.graph import make_band_of_blocks


def test_sectioning_intra_stronger_than_inter():
    D, sec = 128, 32
    W = make_band_of_blocks(D, section_size=sec, hops=2, reuse_decay=0.7)
    A = W.toarray()
    intra = 0.0
    inter = 0.0
    ic = 0
    oc = 0
    for i in range(10, 20):
        for j in range(10, 20):
            if i != j:
                intra += A[i][j]
                ic += 1
        for j in range(70, 80):
            inter += A[i][j]
            oc += 1
    assert ic > 0 and oc > 0
    assert (intra / ic) > (inter / oc)


def test_sectioning_cross_links_present():
    D, sec = 64, 16
    hops = 2
    reuse = 0.7
    cross_scale = 0.01
    W = make_band_of_blocks(D, section_size=sec, hops=hops,
                            reuse_decay=reuse, cross_scale=cross_scale)
    cross_weight = cross_scale * (reuse ** hops)

    boundary_row = sec - 1
    cross_col = sec
    start, end = W.indptr[boundary_row], W.indptr[boundary_row + 1]
    row_entries = {W.indices[k]: W.data[k] for k in range(start, end)}

    assert cross_col in row_entries
    assert row_entries[cross_col] == pytest.approx(cross_weight)

