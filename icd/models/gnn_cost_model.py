"""Graph neural network based latency predictor."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from icd.core.graph import CSRMatrix


@dataclass
class PermutationGraphExample:
    adjacency: CSRMatrix
    permutation: Sequence[int]
    sparsity: Sequence[float]
    latency_ms: float

    @staticmethod
    def from_json(payload: dict) -> "PermutationGraphExample":
        csr = CSRMatrix(
            indptr=list(payload["adjacency"]["indptr"]),
            indices=list(payload["adjacency"]["indices"]),
            data=list(payload["adjacency"]["data"]),
            shape=tuple(payload["adjacency"]["shape"]),
            meta={},
        )
        return PermutationGraphExample(
            adjacency=csr,
            permutation=list(payload["permutation"]),
            sparsity=list(payload.get("sparsity_mask", [])),
            latency_ms=float(payload["latency_ms"]),
        )


class PermutationGraphDataset(Dataset[PermutationGraphExample]):
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.examples: List[PermutationGraphExample] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                self.examples.append(PermutationGraphExample.from_json(payload))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> PermutationGraphExample:
        return self.examples[idx]


def _csr_to_dense_tensor(csr: CSRMatrix) -> torch.Tensor:
    dense = torch.tensor(csr.to_dense(), dtype=torch.float32)
    return dense


def _permutation_features(example: PermutationGraphExample) -> torch.Tensor:
    n = example.adjacency.shape[0]
    perm = torch.tensor(example.permutation, dtype=torch.float32)
    perm = perm / max(float(n - 1), 1.0)
    if example.sparsity:
        mask = torch.tensor(example.sparsity, dtype=torch.float32)
    else:
        mask = torch.zeros(n, dtype=torch.float32)
    return torch.stack([perm, mask], dim=-1)


class GNNCostModel(nn.Module):
    def __init__(self, hidden_dim: int = 64, layers: int = 3) -> None:
        super().__init__()
        self.input_proj = nn.Linear(2, hidden_dim)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_dim * 2, hidden_dim) for _ in range(layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(layers)])
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, adjacency: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.input_proj(features))
        adjacency = adjacency / torch.clamp(adjacency.sum(dim=-1, keepdim=True), min=1.0)
        for layer, norm in zip(self.layers, self.norms):
            agg = adjacency @ h
            h = torch.relu(layer(torch.cat([h, agg], dim=-1)))
            h = norm(h)
        pooled = h.mean(dim=0)
        return self.readout(pooled).squeeze(-1)


def gnn_collate(batch: Iterable[PermutationGraphExample]):
    adjacencies = []
    features = []
    latencies = []
    for example in batch:
        adjacencies.append(_csr_to_dense_tensor(example.adjacency))
        features.append(_permutation_features(example))
        latencies.append(torch.tensor(example.latency_ms, dtype=torch.float32))
    return adjacencies, features, torch.stack(latencies)


def train_cost_model(
    dataset: PermutationGraphDataset,
    *,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 4,
    device: str = "cpu",
) -> GNNCostModel:
    model = GNNCostModel().to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=gnn_collate)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for _ in range(epochs):
        for adj_list, feat_list, lat in loader:
            preds = []
            for adj, feat in zip(adj_list, feat_list):
                pred = model(adj.to(device), feat.to(device))
                preds.append(pred)
            preds_tensor = torch.stack(preds)
            loss = criterion(preds_tensor, lat.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


__all__ = [
    "PermutationGraphDataset",
    "PermutationGraphExample",
    "GNNCostModel",
    "train_cost_model",
    "gnn_collate",
]
