from icd.measure.metrics_bridge import bridge_metrics


def test_bridge_metrics_merges_values():
    hlo = {"latency_delta": -0.5, "accuracy_delta": 0.0}
    runtime = {"latency_ms_mean": 10.0}
    result = bridge_metrics(hlo, runtime)
    assert result.passed is True
    assert result.metrics["stablehlo.latency_delta"] == -0.5


def test_bridge_metrics_detects_failure():
    hlo = {"latency_delta": 0.1}
    runtime = {}
    result = bridge_metrics(hlo, runtime)
    assert result.passed is False
