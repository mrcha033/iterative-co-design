import json
import types

import torch
import torch.nn as nn
import pytest

from icd.core.graph import build_w
from icd.core.graph_instrumented import (
    DimensionAccessTracker,
    InstrumentedGraphConfig,
    build_w_from_instrumented_access,
    extract_hidden_dimensions,
)


class TinyMLP(nn.Module):
    def __init__(self, hidden_size: int = 16):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.config = types.SimpleNamespace(hidden_size=hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.linear1(x))
        return self.linear2(h)


def test_dimension_access_tracker_builds_dense_edges():
    cfg = InstrumentedGraphConfig(
        temporal_window_ns=1_000_000,
        min_coaccesses=1,
        num_samples=1,
        normalize=False,
    )
    tracker = DimensionAccessTracker(hidden_size=4, config=cfg)
    timestamp = 1_000
    tracker.record_access({0, 1}, timestamp, "layer")
    tracker.record_access({2, 3}, timestamp + 1, "layer")
    tracker.increment_sample_count()
    W = tracker.build_coacccess_graph()
    assert W.shape == (4, 4)
    assert W.meta["source"] == "instrumented"
    # Each dimension should connect to at least one other dimension
    for i in range(4):
        start, end = W.indptr[i], W.indptr[i + 1]
        assert end > start


def test_tracker_raises_when_dimensions_exceed_hidden_size():
    cfg = InstrumentedGraphConfig(
        temporal_window_ns=1_000,
        min_coaccesses=1,
        num_samples=1,
        normalize=False,
    )
    tracker = DimensionAccessTracker(hidden_size=4, config=cfg)
    timestamp = 42
    tracker.record_access({0, 4, 5}, timestamp, "layer")
    tracker.increment_sample_count()

    with pytest.raises(ValueError, match="hidden_size=4"):
        tracker.build_coacccess_graph()


def test_extract_hidden_dimensions_handles_nested_outputs():
    torch.manual_seed(0)
    hidden_size = 4
    tensor = torch.randn(2, hidden_size)
    subset = torch.randn(2, hidden_size // 2)

    dims = extract_hidden_dimensions((tensor, subset), hidden_size)

    assert dims == set(range(hidden_size))


def test_extract_hidden_dimensions_threshold_filters_values():
    tensor = torch.tensor([[0.0, 0.6, 0.0, 0.2]], dtype=torch.float32)

    dims = extract_hidden_dimensions(
        tensor,
        hidden_size=4,
        activation_threshold=0.5,
        cache_line_bytes=4,
    )

    assert dims == {1}


def test_build_w_from_instrumented_access_creates_graph(tmp_path):
    torch.manual_seed(0)
    model = TinyMLP(hidden_size=8).eval()
    x = torch.randn(2, 8)
    trace_path = tmp_path / "trace.jsonl"

    cfg = InstrumentedGraphConfig(
        temporal_window_ns=1_000_000,
        min_coaccesses=1,
        num_samples=3,
        store_access_log=True,
        max_events_to_store=8,
        activation_threshold=0.25,
        cache_line_bytes=4,
        trace_output_path=str(trace_path),
    )

    W = build_w_from_instrumented_access(model, x, hidden_size=8, config=cfg)

    assert W.shape == (8, 8)
    assert W.meta["source"] == "instrumented"
    assert W.meta["num_samples"] == cfg.num_samples
    assert len(W.data) > 0
    assert "instrumented_events" in W.meta
    assert len(W.meta["instrumented_events"]) <= cfg.max_events_to_store

    # Trace file should exist and contain JSON lines
    contents = trace_path.read_text(encoding="utf-8").strip().splitlines()
    assert contents, "trace output should not be empty"
    first_event = json.loads(contents[0])
    assert {"dimension", "timestamp_ns", "layer", "operation"}.issubset(first_event)


def test_build_w_integration_instrumented_source():
    torch.manual_seed(1)
    model = TinyMLP(hidden_size=6).eval()
    x = torch.randn(1, 6)

    W = build_w(
        source="instrumented",
        model=model,
        example_inputs=x,
        hidden_size=6,
        instrumented={
            "num_samples": 2,
            "min_coaccesses": 1,
            "temporal_window_ns": 1_000_000,
        },
    )

    assert W.shape == (6, 6)
    assert W.meta["source"] == "instrumented"
    assert W.meta.get("normalize") == "sym"
    assert W.meta.get("nnz_after", len(W.data)) >= len(W.data)
