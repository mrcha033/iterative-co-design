from pathlib import Path

import pytest
import yaml

torch = pytest.importorskip("torch")

import icd.graph.correlation as corr_mod
from icd.graph.correlation import CorrelationConfig, collect_correlations, correlation_to_csr
from icd.runtime.orchestrator import _make_correlation_config


class IdentityLinear(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(dim, dim, bias=False)
        torch.nn.init.eye_(self.linear.weight)

    def forward(self, x):
        return self.linear(x)


def test_activation_correlation_matches_manual():
    model = IdentityLinear(2)
    inputs = [
        (torch.tensor([[1.0, 2.0], [3.0, 4.0]]),),
        (torch.tensor([[0.0, 1.0], [1.0, 0.0]]),),
    ]
    cfg = CorrelationConfig(
        layers=["linear"],
        samples=len(inputs),
        dtype=torch.float64,
        transfer_batch_size=1,
    )
    matrix, meta = collect_correlations(model, inputs, cfg=cfg)

    all_samples = torch.cat([x for (x,) in inputs], dim=0)
    mean = all_samples.mean(dim=0)
    manual = (all_samples - mean).t().mm(all_samples - mean) / all_samples.shape[0]
    manual = manual.to(matrix.dtype)

    assert torch.allclose(matrix, manual, atol=1e-6)
    assert meta["samples"] == len(inputs)
    assert meta["layers"]
    layer_meta = meta["layers"][0]
    assert layer_meta["name"].endswith("linear")
    assert layer_meta["samples"] == all_samples.shape[0]
    assert layer_meta["feature_dim"] == manual.shape[0]
    assert layer_meta["count"] == all_samples.shape[0]


def test_correlation_to_csr_threshold():
    matrix = torch.tensor(
        [
            [0.0, 0.5, 0.1],
            [0.5, 0.0, 0.2],
            [0.1, 0.2, 0.0],
        ]
    )
    cfg = CorrelationConfig(threshold=0.2, normalize="none", nnz_cap=10)
    csr = correlation_to_csr(matrix, cfg=cfg)
    assert csr.shape == (3, 3)
    assert csr.nnz() == 4  # pairs (0,1),(1,0),(1,2),(2,1)


def test_whiten_option_sets_unit_diagonal():
    model = IdentityLinear(3)
    inputs = [
        (torch.randn(4, 3),),
        (torch.randn(4, 3),),
    ]
    cfg = CorrelationConfig(layers=["linear"], samples=len(inputs), dtype=torch.float64, whiten=True)
    matrix, meta = collect_correlations(model, inputs, cfg=cfg)
    assert torch.allclose(matrix.diagonal(), torch.ones_like(matrix.diagonal()), atol=1e-6)
    assert meta["whiten"] is True


def test_yaml_config_enables_whiten_and_transfer(monkeypatch):
    raw_cfg = yaml.safe_load(Path("configs/iasp_defaults.yaml").read_text(encoding="utf-8"))
    corr_data = dict(raw_cfg.get("correlation", {}))
    corr_data["samples"] = 1
    corr_data["layers"] = ["linear"]

    updates: list[int] = []
    original_update = corr_mod._ActivationStats.update

    def recording_update(self, activations):
        updates.append(int(activations.shape[0]))
        return original_update(self, activations)

    monkeypatch.setattr(corr_mod._ActivationStats, "update", recording_update)

    model = IdentityLinear(3)
    inputs = [(torch.randn(9, 3),)]

    cfg = _make_correlation_config(corr_data)
    matrix, meta = collect_correlations(model, inputs, cfg=cfg)

    assert cfg.whiten is True
    assert cfg.transfer_batch_size == corr_data["transfer_batch_size"]
    assert torch.allclose(matrix.diagonal(), torch.ones_like(matrix.diagonal()), atol=1e-6)
    assert meta["whiten"] is True
    assert meta["transfer_batch_size"] == cfg.transfer_batch_size
    assert updates
    assert max(updates) <= cfg.transfer_batch_size


class MixedFeatureModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear4 = torch.nn.Linear(4, 4, bias=False)
        torch.nn.init.eye_(self.linear4.weight)
        self.heads = torch.nn.ModuleList([
            torch.nn.Linear(4, 3, bias=False),
            torch.nn.Linear(4, 3, bias=False),
        ])
        for head in self.heads:
            torch.nn.init.eye_(head.weight)

    def forward(self, x):  # type: ignore[override]
        out = self.linear4(x)
        # Execute additional heads to surface alternate feature dimensions for hooks.
        for head in self.heads:
            head(x)
        return out


def test_expected_dim_filters_layers_with_mismatched_feature_dim():
    model = MixedFeatureModel()
    inputs = [(torch.randn(2, 4),) for _ in range(3)]

    cfg = CorrelationConfig(
        samples=len(inputs),
        expected_dim=4,
        layers=["linear4", "heads.0", "heads.1"],
    )

    matrix, meta = collect_correlations(model, inputs, cfg=cfg)

    assert matrix.shape == (4, 4)
    assert meta["feature_dim"] == 4
    assert meta["expected_dim"] == 4
    assert meta["selection"]["matched_expected_dim"] is True
    assert 3 in meta["available_feature_dims"]

    selected_names = {layer["name"] for layer in meta["layers"] if layer.get("selected")}
    assert any(layer.get("feature_dim") == 3 and not layer.get("selected") for layer in meta["layers"])
    assert any(layer.get("ignored_reason") == "feature_dim_mismatch" for layer in meta["layers"] if layer.get("feature_dim") == 3)
    assert "linear4" in selected_names
