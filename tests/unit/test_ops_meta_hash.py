from icd.core.graph import build_w


class _DummyModel:
    def eval(self):
        return self

    def __call__(self, *a, **k):  # CI-safe dummy
        return None


def test_trace_meta_or_hash_present_in_pytorch_fallback():
    # Use pytorch source but rely on fallback path (no torch required)
    W = build_w(
        source="pytorch",
        model=_DummyModel(),
        example_inputs=(object(),),
        normalize="sym",
        pytorch={"hops": 1},
    )
    meta = getattr(W, "meta", {}).get("pytorch", {})
    assert ("trace_hash" in meta and isinstance(meta["trace_hash"], str)) or ("used_ops" in meta)

