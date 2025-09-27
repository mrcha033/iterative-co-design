import math

import pytest

from icd.measure.gates import make_pairwise_summary
from icd.measure.significance import compute_prd_significance, paired_statistics


def test_paired_statistics_zero_variance_delta():
    baseline = [10.0, 12.0, 11.0, 13.0]
    trial = [9.5, 11.5, 10.5, 12.5]
    result = paired_statistics(baseline, trial)
    assert result["sample_size"] == 4
    assert result["mean_baseline"] == pytest.approx(11.5)
    assert result["mean_trial"] == pytest.approx(11.0)
    assert result["mean_diff"] == pytest.approx(-0.5)
    assert result["method"] == "zero_variance"
    assert result["ci_low"] == pytest.approx(-0.5)
    assert result["ci_high"] == pytest.approx(-0.5)
    assert result["p_value"] == 0.0
    assert result["effect_size"] is None


def test_paired_statistics_normal_approximation():
    baseline = [10.0, 12.0, 11.0, 13.0]
    trial = [9.0, 11.0, 11.0, 14.0]
    result = paired_statistics(baseline, trial)
    assert result["sample_size"] == 4
    assert result["method"] == "normal_approx"
    assert result["mean_diff"] == pytest.approx(-0.25)
    assert result["statistic"] == pytest.approx(-0.5222329678)
    assert result["p_value"] == pytest.approx(0.6015081344)
    assert result["effect_size"] == pytest.approx(-0.2611164839)
    assert result["ci_low"] == pytest.approx(-1.1882613245)
    assert result["ci_high"] == pytest.approx(0.6882613245)


def test_compute_prd_significance_with_metrics():
    baseline = {
        "mode": "linear",
        "latency_ms": {"mean": 11.5, "samples": [10.0, 12.0, 11.0, 13.0]},
        "latency_ms_mean": 11.5,
        "l2_hit_pct": 0.60,
        "ept_j_per_tok": 0.50,
    }
    trial = {
        "mode": "iterative",
        "latency_ms": {"mean": 11.0, "samples": [9.0, 11.0, 11.0, 14.0]},
        "latency_ms_mean": 11.0,
        "l2_hit_pct": 0.58,
        "ept_j_per_tok": 0.48,
    }
    stats = compute_prd_significance(baseline, trial)
    lat = stats["latency_ms"]
    assert lat["sample_size"] == 4
    assert lat["mean_diff"] == pytest.approx(-0.25)
    assert stats["l2_hit_pct"]["sample_size"] == 1
    assert stats["l2_hit_pct"]["mean_diff"] == pytest.approx(-0.020000000000000018)
    assert stats["l2_hit_pct"]["method"] == "insufficient_samples"
    assert stats["ept_j_per_tok"]["mean_diff"] == pytest.approx(-0.020000000000000018)


def test_make_pairwise_summary_includes_significance():
    dense = {
        "mode": "dense",
        "latency_ms": {"mean": 20.0, "samples": [20.0, 20.5, 19.5, 20.5]},
        "latency_ms_mean": 20.0,
        "l2_hit_pct": 0.55,
        "ept_j_per_tok": 0.62,
    }
    linear = {
        "mode": "linear",
        "latency_ms": {"mean": 15.0, "samples": [15.0, 15.5, 14.5, 15.0]},
        "latency_ms_mean": 15.0,
        "l2_hit_pct": 0.60,
        "ept_j_per_tok": 0.60,
    }
    iterative = {
        "mode": "iterative",
        "latency_ms": {"mean": 12.0, "samples": [12.0, 11.5, 12.5, 12.0]},
        "latency_ms_mean": 12.0,
        "l2_hit_pct": 0.74,
        "ept_j_per_tok": 0.48,
    }
    summary = make_pairwise_summary([dense, linear, iterative])
    iter_summary = next(m for m in summary if m.get("mode") == "iterative")
    assert "significance" in iter_summary
    sig_vs_linear = iter_summary["significance"]["linear"]
    latency_stats = sig_vs_linear["latency_ms"]
    assert latency_stats["sample_size"] == 4
    assert math.isfinite(latency_stats["mean_diff"])


def test_make_pairwise_summary_three_baselines_all_pairs():
    def make_metrics(mode: str, latency: float, l2: float, ept: float) -> dict:
        return {
            "mode": mode,
            "latency_ms": {
                "mean": latency,
                "samples": [latency - 0.5, latency, latency + 0.5],
            },
            "latency_ms_mean": latency,
            "l2_hit_pct": l2,
            "l2_hit_pct_samples": [l2 - 0.01, l2, l2 + 0.01],
            "ept_j_per_tok": ept,
            "ept_j_per_tok_samples": [ept - 0.02, ept, ept + 0.02],
        }

    dense = make_metrics("dense", 22.0, 0.52, 0.61)
    linear = make_metrics("linear", 18.0, 0.58, 0.57)
    louvain = make_metrics("linear:louvain", 19.5, 0.56, 0.59)
    iterative = make_metrics("iterative", 14.0, 0.74, 0.48)

    summary = make_pairwise_summary([dense, linear, louvain, iterative])
    by_mode = {str(entry.get("mode", "")).lower(): entry for entry in summary}

    assert set(by_mode.keys()) == {"dense", "linear", "linear:louvain", "iterative"}

    for mode, entry in by_mode.items():
        sig = entry.get("significance")
        assert isinstance(sig, dict)
        expected_keys = {m for m in by_mode if m != mode}
        assert set(sig.keys()) == expected_keys

    dense_vs_linear = by_mode["dense"]["significance"]["linear"]["latency_ms"]["mean_diff"]
    linear_vs_dense = by_mode["linear"]["significance"]["dense"]["latency_ms"]["mean_diff"]
    assert dense_vs_linear == pytest.approx(-linear_vs_dense)
