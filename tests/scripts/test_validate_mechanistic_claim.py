import torch

from scripts.validate_mechanistic_claim import measure_cache_and_latency_for_permutation


class DummyConfig:
    model_type = "mamba"


class DummyMambaLayer(torch.nn.Module):
    def __init__(self, hidden: int = 4) -> None:
        super().__init__()
        self.A = torch.nn.Linear(hidden, hidden, bias=True)
        self.B = torch.nn.Linear(hidden, hidden, bias=True)
        self.C = torch.nn.Linear(hidden, hidden, bias=True)
        self.x0 = torch.nn.Parameter(torch.zeros(hidden))


class DummyMambaModel(torch.nn.Module):
    def __init__(self, hidden: int = 4) -> None:
        super().__init__()
        self.config = DummyConfig()
        self.block = DummyMambaLayer(hidden)


def _clone_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def test_permutation_applies_and_restores(monkeypatch):
    model = DummyMambaModel(hidden=4)
    inputs = torch.randn(1, 4)
    permutation = [2, 0, 3, 1]
    original_state = _clone_state_dict(model)

    permuted_flags: dict[str, bool] = {}

    def fake_latency(model, inputs, **kwargs):  # type: ignore[override]
        permuted_flags["latency"] = not torch.equal(
            model.block.A.weight, original_state["block.A.weight"]
        )
        return {"mean": 1.23}

    def fake_collect(model, inputs, **kwargs):  # type: ignore[override]
        permuted_flags["cache"] = not torch.equal(
            model.block.A.weight, original_state["block.A.weight"]
        )
        return {"l2_hit_rate_pct": 98.7}

    monkeypatch.setattr(
        "scripts.validate_mechanistic_claim.measure_latency_with_stats",
        fake_latency,
    )
    monkeypatch.setattr(
        "scripts.validate_mechanistic_claim.collect_l2_metrics",
        fake_collect,
    )

    l2_hit_rate, latency = measure_cache_and_latency_for_permutation(
        model,
        inputs,
        permutation,
        device="cpu",
        num_warmup=0,
        num_samples=1,
    )

    assert permuted_flags.get("latency")
    assert permuted_flags.get("cache")
    assert l2_hit_rate == 98.7
    assert latency == 1.23

    restored_state = _clone_state_dict(model)
    for key, tensor in original_state.items():
        assert torch.equal(restored_state[key], tensor)
