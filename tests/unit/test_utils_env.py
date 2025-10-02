from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

import icd.utils.env as env


class _StubNvml:
    NVML_CLOCK_GRAPHICS = 1
    NVML_CLOCK_SM = 2
    NVML_CLOCK_MEM = 3
    NVML_CLOCK_VIDEO = 4
    NVML_TEMPERATURE_GPU = 10
    NVML_TEMPERATURE_MEMORY = 11

    class NVMLError(Exception):
        pass

    def __init__(self) -> None:
        self.init_called = False
        self.shutdown_called = False

    def nvmlInit(self) -> None:
        self.init_called = True

    def nvmlShutdown(self) -> None:
        self.shutdown_called = True

    def nvmlSystemGetDriverVersion(self):
        return b"555.55"

    def nvmlDeviceGetHandleByIndex(self, index: int) -> object:
        assert index == 0
        return object()

    def nvmlDeviceGetName(self, handle: object):
        return b"Stub GPU"

    def nvmlDeviceGetClockInfo(self, handle: object, domain: int) -> int:
        return {self.NVML_CLOCK_GRAPHICS: 1200, self.NVML_CLOCK_SM: 1300, self.NVML_CLOCK_MEM: 1400, self.NVML_CLOCK_VIDEO: 1500}[domain]

    def nvmlDeviceGetTemperature(self, handle: object, target: int) -> int:
        return {self.NVML_TEMPERATURE_GPU: 55, self.NVML_TEMPERATURE_MEMORY: 60}[target]


def test_collect_env_fingerprint_with_stubbed_nvml(monkeypatch):
    stub = _StubNvml()
    monkeypatch.setattr(env, "pynvml", stub, raising=False)
    monkeypatch.setattr(env, "DEFAULT_PACKAGES", (("pytest", "pytest"),))

    fingerprint = env.collect_env_fingerprint()
    assert fingerprint["gpu"]["name"] == "Stub GPU"
    assert fingerprint["host"]["driver"] == "555.55"
    clocks = fingerprint["gpu"].get("clocks", {})
    assert clocks.get("graphics") == pytest.approx(1200.0)
    assert stub.init_called and stub.shutdown_called


def test_resolve_package_versions_prefers_module_version(monkeypatch):
    fake_module = SimpleNamespace(__version__="1.2.3")
    monkeypatch.setitem(sys.modules, "fake_pkg", fake_module)

    def fake_version(dist: str) -> str:
        if dist == "other":
            return "0.9.0"
        raise env.importlib_metadata.PackageNotFoundError

    monkeypatch.setattr(env.importlib_metadata, "version", fake_version)

    entries = env._resolve_package_versions([("fake-dist", "fake_pkg"), ("other", None), ("missing", "missing")])
    assert {"name": "fake-dist", "version": "1.2.3"} in entries
    assert {"name": "other", "version": "0.9.0"} in entries
    assert all(item["name"] != "missing" for item in entries)


def test_collect_gpu_info_handles_handle_failure(monkeypatch):
    class FailingNvml(_StubNvml):
        def nvmlDeviceGetHandleByIndex(self, index: int) -> object:
            raise self.NVMLError()

    stub = FailingNvml()
    monkeypatch.setattr(env, "pynvml", stub, raising=False)

    gpu, driver_hint = env._collect_gpu_info()
    assert gpu["name"] == "unavailable"
    assert driver_hint == "555.55"


def test_collect_gpu_info_shutdown_error(monkeypatch):
    class ShutdownNvml(_StubNvml):
        def nvmlShutdown(self) -> None:
            raise self.NVMLError()

    stub = ShutdownNvml()
    monkeypatch.setattr(env, "pynvml", stub, raising=False)

    gpu, driver_hint = env._collect_gpu_info()
    assert gpu["clocks"]
    assert driver_hint == "555.55"
