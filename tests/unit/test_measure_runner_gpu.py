import torch

from icd.measure.runner_gpu import BenchmarkConfig, benchmark_inference


def test_benchmark_inference_cpu():
    model = torch.nn.Linear(4, 4)
    example = torch.ones(1, 4)
    cfg = BenchmarkConfig(repeats=5, warmup=1, sync=True, use_cuda_events=False)
    metrics = benchmark_inference(model, example, cfg)

    assert metrics["repeats"] == 5
    assert metrics["warmup"] == 1
    assert "latency_ms_mean" in metrics
    assert metrics["latency_ms_mean"] > 0
    assert metrics["device"].startswith("cpu")


def test_benchmark_inference_tokens():
    model = torch.nn.Linear(2, 2)
    example = torch.ones(1, 2)
    cfg = BenchmarkConfig(repeats=3, warmup=0, tokens_per_batch=16)
    metrics = benchmark_inference(model, example, cfg)
    assert metrics["tokens"] == 48
    assert metrics["throughput_toks_s"] > 0

