from __future__ import annotations

import pytest

from icd.runtime.runner import prepare_runner_context, resolve_runner


def dummy_runner(mode: str, context: dict) -> dict:
    return {"mode": mode, "ctx": context}


def test_resolve_runner_from_callable():
    runner = resolve_runner({"runner": dummy_runner})
    assert runner is dummy_runner


def test_resolve_runner_from_dotted_path():
    runner = resolve_runner({"runner": "icd.runtime.runners.mock_inference"})
    result = runner("linear", {"tokens": 4, "provide_l2": True})
    assert "l2_hit_pct" in result


def test_resolve_runner_invalid_path():
    with pytest.raises(ValueError):
        resolve_runner({"runner": "nonexistent.module:call"})


def test_prepare_runner_context_returns_copy():
    ctx = prepare_runner_context(foo=1)
    ctx["foo"] = 2
    other = prepare_runner_context(foo=1)
    assert other["foo"] == 1
