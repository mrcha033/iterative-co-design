from __future__ import annotations

import types

import importlib
import importlib.util

import pytest

from icd.runtime.apply_pi import (
    detect_stablehlo_capability,
    require_stablehlo_capability,
)


def test_detect_stablehlo_disabled_via_env(monkeypatch):
    # Guard that no import is attempted when explicitly disabled.
    called = False

    def _sentinel(name):  # pragma: no cover - should not run
        nonlocal called
        called = True
        return object()

    monkeypatch.setattr(importlib.util, "find_spec", _sentinel)

    cap = detect_stablehlo_capability(env={"ICD_DISABLE_STABLEHLO": "1"})
    assert not cap.available
    assert "disabled" in (cap.reason or "").lower()
    assert not called


def test_detect_stablehlo_missing_package(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

    cap = detect_stablehlo_capability(env={})
    assert not cap.available
    assert "stablehlo" in (cap.reason or "").lower()


def test_detect_stablehlo_success(monkeypatch):
    sentinel_spec = types.SimpleNamespace()
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: sentinel_spec)

    fake_module = types.SimpleNamespace(__name__="stablehlo")
    monkeypatch.setattr(importlib, "import_module", lambda name: fake_module)

    import icd.runtime.apply_pi as apply_pi

    monkeypatch.setattr(apply_pi.metadata, "version", lambda name: "0.15.0")

    cap = detect_stablehlo_capability(env={})
    assert cap.available
    assert cap.details["module"] == "stablehlo"
    assert cap.details["version"] == "0.15.0"


def test_require_stablehlo_capability_raises(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

    with pytest.raises(RuntimeError) as excinfo:
        require_stablehlo_capability(env={})

    assert "stablehlo" in str(excinfo.value).lower()
