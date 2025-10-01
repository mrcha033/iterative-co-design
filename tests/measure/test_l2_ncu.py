from pathlib import Path

import pytest
import torch

from icd.measure import l2_ncu


class DummyCompletedProcess:
    def __init__(self):
        self.returncode = 0
        self.stdout = ""
        self.stderr = ""


@pytest.mark.skipif(not hasattr(torch, "cuda"), reason="PyTorch without CUDA module")
def test_generated_script_invokes_real_model(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    calls = []

    def fake_run(cmd, *args, **kwargs):
        calls.append((cmd, kwargs))
        return DummyCompletedProcess()

    monkeypatch.setattr(l2_ncu.subprocess, "run", fake_run)

    model = torch.nn.Linear(4, 4)
    inputs = (torch.randn(2, 4),)

    result = l2_ncu.collect_l2_metrics(model, inputs, ncu_path="ncu")
    assert isinstance(result, dict)

    assert calls, "expected subprocess.run to be called"
    script_path = Path(calls[0][0][-1])
    assert script_path.exists(), "profiling script should be created"

    script_text = script_path.read_text()
    assert "torch.load" in script_text
    assert "model(*inputs)" in script_text
    assert "torch.nn.functional.linear" not in script_text
