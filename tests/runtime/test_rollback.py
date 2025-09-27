import importlib.util
import sys
import types

if "torch" not in sys.modules:  # pragma: no cover - exercised on CPU-only CI
    torch_stub = types.ModuleType("torch")

    class _TorchDType:
        pass

    class _TorchTensor:
        pass

    class _TorchParameter:
        pass

    def _no_grad():
        def decorator(fn):
            return fn

        return decorator

    torch_stub.nn = types.SimpleNamespace(Parameter=_TorchParameter)
    torch_stub.Tensor = _TorchTensor
    torch_stub.dtype = _TorchDType
    torch_stub.float32 = _TorchDType()
    torch_stub.float64 = _TorchDType()
    torch_stub.float16 = _TorchDType()
    torch_stub.no_grad = _no_grad
    torch_stub.__spec__ = importlib.util.spec_from_loader("torch", loader=None)

    sys.modules["torch"] = torch_stub

from icd.runtime.orchestrator import evaluate_acceptance


def test_evaluate_acceptance_pass():
    decision = evaluate_acceptance(delta_J=-0.02, epsilon_J=0.01, retry_budget=0)
    assert decision == {"accepted": True, "rolled_back": False, "retry": False}


def test_evaluate_acceptance_retry():
    decision = evaluate_acceptance(delta_J=0.005, epsilon_J=0.01, retry_budget=2)
    assert decision["accepted"] is False
    assert decision["rolled_back"] is True
    assert decision["retry"] is True
