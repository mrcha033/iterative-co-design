from __future__ import annotations

import pytest

from icd.measure.gates import make_pairwise_summary, verdict


def test_verdict_sst2_pass():
    metrics = {"task": "sst2", "mode": "dense", "latency_ms_mean": 10.0, "accuracy": 0.95}
    verdict(metrics)
    assert metrics["verdict"] == "pass"


def test_verdict_sst2_fail():
    metrics = {"task": "sst2", "mode": "dense", "latency_ms_mean": 10.0, "accuracy": 0.80}
    verdict(metrics)
    assert metrics["verdict"] == "fail"


def test_verdict_iterative_latency_gate():
    dense = {
        "task": "sst2",
        "mode": "dense",
        "latency_ms_mean": 25.0,
        "accuracy": 0.95,
        "l2_hit_pct": 0.70,
        "ept_j_per_tok": 1.10,
    }
    linear = {
        "task": "sst2",
        "mode": "linear",
        "latency_ms_mean": 15.0,
        "accuracy": 0.949,
        "l2_hit_pct": 0.70,
        "ept_j_per_tok": 1.00,
    }
    iterative = {
        "task": "sst2",
        "mode": "iterative",
        "latency_ms_mean": 12.0,
        "accuracy": 0.949,
        "l2_hit_pct": 0.82,
        "ept_j_per_tok": 0.80,
    }
    verdict(iterative, dense_metrics=dense, linear_metrics=linear)
    assert iterative["verdict"] == "pass"
    gates = iterative["gates"]
    thresholds = gates["thresholds"]
    status = gates["status"]
    observed = gates["observed"]
    missing = gates.get("missing", [])

    assert thresholds["iter.latency_rel"] == pytest.approx(-0.20)
    assert status["iter.latency_rel"] is True
    assert observed["iter.latency_rel"] == pytest.approx(-0.2)
    assert status["iter.l2_pp"] is True
    assert observed["iter.l2_pp"] == pytest.approx(12.0)
    assert status["iter.ept_rel"] is True
    assert observed["iter.ept_rel"] == pytest.approx(-0.2)
    assert status["quality.acc_drop_pp"] is True
    assert missing == []


def test_verdict_iterative_gate_failure():
    dense = {
        "task": "sst2",
        "mode": "dense",
        "latency_ms_mean": 20.0,
        "accuracy": 0.95,
    }
    linear = {
        "task": "sst2",
        "mode": "linear",
        "latency_ms_mean": 15.0,
        "accuracy": 0.94,
        "l2_hit_pct": 70.0,
        "ept_j_per_tok": 0.60,
    }
    iterative = {
        "task": "sst2",
        "mode": "iterative",
        "latency_ms_mean": 13.0,
        "accuracy": 0.94,
        "l2_hit_pct": 75.0,
        "ept_j_per_tok": 0.59,
    }
    verdict(iterative, dense_metrics=dense, linear_metrics=linear)
    assert iterative["verdict"] == "fail"


def test_make_pairwise_summary_deltas():
    metrics = [
        {
            "task": "sst2",
            "mode": "dense",
            "latency_ms_mean": 20.0,
            "accuracy": 0.95,
            "l2_hit_pct": 55.0,
            "ept_j_per_tok": 0.62,
        },
        {
            "task": "sst2",
            "mode": "linear",
            "latency_ms_mean": 15.0,
            "accuracy": 0.94,
            "l2_hit_pct": 60.0,
            "ept_j_per_tok": 0.60,
        },
        {
            "task": "sst2",
            "mode": "iterative",
            "latency_ms_mean": 12.0,
            "accuracy": 0.951,
            "l2_hit_pct": 74.0,
            "ept_j_per_tok": 0.48,
        },
    ]
    summary = make_pairwise_summary(metrics)
    assert summary[2]["delta_latency_vs_dense"] == -8.0
    assert summary[2]["delta_accuracy_vs_dense"] == pytest.approx(0.001)
    assert summary[2]["delta_l2_hit_vs_linear"] == pytest.approx(14.0)
    assert summary[2]["delta_ept_vs_linear"] == pytest.approx(-0.12)
