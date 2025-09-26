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
    dense = {"task": "sst2", "mode": "dense", "latency_ms_mean": 20.0, "accuracy": 0.95}
    linear = {"task": "sst2", "mode": "linear", "latency_ms_mean": 15.0, "accuracy": 0.94}
    iterative = {"task": "sst2", "mode": "iterative", "latency_ms_mean": 14.0, "accuracy": 0.94}
    verdict(iterative, dense_metrics=dense, linear_metrics=linear)
    assert iterative["verdict"] == "pass"


def test_make_pairwise_summary_deltas():
    metrics = [
        {"task": "sst2", "mode": "dense", "latency_ms_mean": 20.0, "accuracy": 0.95},
        {"task": "sst2", "mode": "iterative", "latency_ms_mean": 14.0, "accuracy": 0.96},
    ]
    summary = make_pairwise_summary(metrics)
    assert summary[1]["delta_latency_vs_dense"] == -6.0
    assert summary[1]["delta_accuracy_vs_dense"] == pytest.approx(0.01)
