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

