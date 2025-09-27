"""Graph neural network experiment loaders leveraging PyTorch Geometric."""

from __future__ import annotations

import os
from typing import Any, Tuple

from ._torch_utils import resolve_device, resolve_dtype


def _default_root(dataset_name: str) -> str:
    safe = dataset_name.replace("/", "_")
    return os.path.join(os.getcwd(), "data", safe)


def _load_dataset(dataset_name: str, root: str | None, dataset_kwargs: dict[str, Any] | None):
    name = dataset_name.lower()
    kwargs = dict(dataset_kwargs or {})
    target_root = root or _default_root(dataset_name)

    try:
        from torch_geometric.datasets import OgbnArxiv, Planetoid
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("torch_geometric is required for the graph loaders") from exc

    if name in {"ogbn-arxiv", "arxiv", "ogbn_arxiv"}:
        dataset = OgbnArxiv(root=target_root, **kwargs)
    elif name in {"cora", "citeseer", "pubmed"}:
        dataset = Planetoid(root=target_root, name=dataset_name.capitalize(), **kwargs)
    else:
        raise ValueError(f"unsupported dataset '{dataset_name}' for GNN loader")
    if len(dataset) == 0:  # pragma: no cover - defensive guard
        raise RuntimeError(f"dataset '{dataset_name}' did not return any samples")
    data = dataset[0]
    return dataset, data


def _prepare_example(data, device_name: str, torch_dtype) -> Tuple[Any, Any]:
    x = data.x.to(device=device_name, dtype=torch_dtype)
    edge_index = data.edge_index.to(device=device_name)
    return x, edge_index


def _build_gnn_model(
    model_type: str,
    num_features: int,
    num_classes: int,
    *,
    hidden_channels: int,
    num_layers: int,
    dropout: float,
):
    model_type = model_type.lower()
    try:
        from torch_geometric.nn.models import GCN, GraphSAGE
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("torch_geometric is required for the graph loaders") from exc

    if model_type == "gcn":
        return GCN(
            in_channels=num_features,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=num_classes,
            dropout=dropout,
        )
    if model_type in {"graphsage", "sage"}:
        return GraphSAGE(
            in_channels=num_features,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=num_classes,
            dropout=dropout,
        )
    raise ValueError(f"unsupported model type '{model_type}'")


def _load_gnn(
    model_type: str,
    dataset_name: str,
    *,
    hidden_channels: int = 256,
    num_layers: int = 2,
    dropout: float = 0.5,
    device: str | None = None,
    dtype: str | None = None,
    root: str | None = None,
    dataset_kwargs: dict[str, Any] | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> Tuple[Any, Tuple[Any, ...]]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("torch is required for the graph loaders") from exc

    torch_dtype = resolve_dtype(dtype, torch)
    device_name = resolve_device(device, torch)

    dataset, data = _load_dataset(dataset_name, root, dataset_kwargs)

    model = _build_gnn_model(
        model_type,
        dataset.num_features,
        dataset.num_classes,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        dropout=dropout,
        **(model_kwargs or {}),
    )
    model.to(device=device_name, dtype=torch_dtype)
    model.eval()

    example_inputs = _prepare_example(data, device_name, torch_dtype)
    return model, example_inputs


def load_pyg_gcn(
    dataset_name: str = "ogbn-arxiv",
    **kwargs: Any,
) -> Tuple[Any, Tuple[Any, ...]]:
    """Load a GCN model and node features for the requested dataset."""

    return _load_gnn("gcn", dataset_name, **kwargs)


def load_pyg_graphsage(
    dataset_name: str = "ogbn-arxiv",
    **kwargs: Any,
) -> Tuple[Any, Tuple[Any, ...]]:
    """Load a GraphSAGE model and node features for the requested dataset."""

    return _load_gnn("graphsage", dataset_name, **kwargs)


__all__ = ["load_pyg_gcn", "load_pyg_graphsage"]
