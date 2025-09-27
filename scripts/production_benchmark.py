#!/usr/bin/env python3
"""Load test script for Triton deployment.

The script issues inference requests to a Triton HTTP endpoint and records
latency/throughput statistics for continuous monitoring during a canary rollout.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Any, Dict, List

import numpy as np
import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("endpoint", help="Triton HTTP endpoint, e.g. http://localhost:8000")
    parser.add_argument("model", help="Triton model name")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--duration", type=int, default=60, help="Duration of the benchmark in seconds")
    parser.add_argument("--input-shape", nargs="*", type=int, default=[1, 3, 224, 224])
    parser.add_argument("--dtype", default="FP32", help="Triton tensor dtype (FP32, FP16, INT64, ...)")
    parser.add_argument("--report", type=str, default=None, help="Path to write JSON results")
    return parser.parse_args()


def build_request(model: str, batch: np.ndarray, dtype: str) -> Dict[str, Any]:
    return {
        "model_name": model,
        "inputs": [
            {
                "name": "INPUT__0",
                "shape": list(batch.shape),
                "datatype": dtype,
                "data": batch.flatten().tolist(),
            }
        ],
        "outputs": [
            {
                "name": "OUTPUT__0",
            }
        ],
    }


def main() -> None:
    args = parse_args()

    endpoint = args.endpoint.rstrip("/") + "/v2/models/{}/infer".format(args.model)
    rng = np.random.default_rng()
    latencies: List[float] = []
    completed = 0
    start = time.perf_counter()
    deadline = start + args.duration

    while time.perf_counter() < deadline:
        batch = rng.standard_normal((args.batch_size, *args.input_shape), dtype=np.float32)
        payload = build_request(args.model, batch, args.dtype)
        send = time.perf_counter()
        response = requests.post(endpoint, json=payload, timeout=30)
        response.raise_for_status()
        latency = time.perf_counter() - send
        latencies.append(latency)
        completed += args.batch_size

    total_time = time.perf_counter() - start
    rps = completed / total_time if total_time > 0 else 0
    summary = {
        "requests": completed,
        "duration_s": total_time,
        "throughput_rps": rps,
        "latency_ms": {
            "mean": statistics.mean(latencies) * 1000 if latencies else 0,
            "p95": statistics.quantiles(latencies, n=20)[-1] * 1000 if len(latencies) >= 20 else 0,
        },
    }

    print(json.dumps(summary, indent=2))
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
