from icd.runtime.compare import decide


def test_decide_accepts_on_j_and_latency():
    base = {
        "latency_ms": {"mean": 100.0},
        "l2_hit_pct": 80.0,
        "ept_j_per_tok": 1.0,
        "acceptance": {"delta_J": 0.0},
    }
    trial = {
        "latency_ms": {"mean": 75.0},
        "l2_hit_pct": 92.0,
        "ept_j_per_tok": 0.8,
        "acceptance": {"delta_J": -0.05},
    }
    v = decide(base, trial, fixed_clock=True, eps_J=0.01)
    assert v.get("accepted") is True
    assert isinstance(v.get("delta", {}).get("lat_rel"), float)
    assert v.get("missing") == []


def test_decide_respects_quality_gate():
    base = {
        "latency_ms": {"mean": 100.0},
        "l2_hit_pct": 80.0,
        "ept_j_per_tok": 1.0,
        "acceptance": {"delta_J": -0.1},
        "quality": {"metric": "accuracy", "after": 0.90},
    }
    # Trial worsens accuracy by more than 0.1pp → fail quality_ok
    trial = {
        "latency_ms": {"mean": 80.0},
        "l2_hit_pct": 95.0,
        "ept_j_per_tok": 0.7,
        "acceptance": {"delta_J": -0.2},
        "quality": {"metric": "accuracy", "after": 0.898},
    }
    v = decide(base, trial, fixed_clock=True, eps_J=0.01)
    assert v.get("quality_ok") is False
    assert v.get("accepted") is False

