# smoke determinism/no-NaN for mock W

from icd.core.graph import build_w


def test_mock_w_determinism_no_nan():
    W1 = build_w(source="mock", D=128, blocks=4, noise=0.02, seed=7)
    W2 = build_w(source="mock", D=128, blocks=4, noise=0.02, seed=7)
    A1, A2 = W1.toarray(), W2.toarray()
    # exact equality may differ after normalization; compare elementwise
    assert A1 == A2
    # non-negative and finite
    for row in A1:
        for v in row:
            assert v >= 0.0
            assert v == v  # not NaN

