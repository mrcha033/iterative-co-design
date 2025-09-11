from icd.core.graph import build_w


def test_build_w_from_triples():
    trace = [(0, 1, 1.0), (1, 2, 2.0), (2, 3, 3.0)]
    W = build_w(source="trace", trace=trace, D=4, normalize="sym")
    assert W.shape == (4, 4)
    assert W.nnz() > 0
    # determinism
    W2 = build_w(source="trace", trace=trace, D=4, normalize="sym")
    assert W.to_npz_payload() == W2.to_npz_payload()


def test_build_w_from_jsonl(tmp_path):
    p = tmp_path / "trace.jsonl"
    p.write_text("\n".join([
        '{"src":0, "dst": 1, "w": 1.0}',
        '{"src":1, "dst": 3, "w": 1.0}',
    ]))
    W = build_w(source="trace", trace=str(p), normalize="row")
    assert W.shape[0] >= 2
    assert W.nnz() > 0

