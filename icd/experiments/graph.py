"""Graph experiment loaders for PyTorch Geometric models."""

from __future__ import annotations

import importlib
from typing import Any, Tuple

from icd.utils.imports import load_object

from ._torch_utils import resolve_device, resolve_dtype

__all__ = ["load_pyg_gcn", "load_pyg_graphsage", "load_gcn", "load_graphsage"]


def _import_pyg_modules():
    """Import PyTorch Geometric dependencies with a helpful error."""

    try:
        torch = importlib.import_module("torch")
        datasets_mod = importlib.import_module("torch_geometric.datasets")
        models_mod = importlib.import_module("torch_geometric.nn.models")
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "torch and torch_geometric are required for the graph experiment loaders"
        ) from exc

    return torch, datasets_mod, models_mod


def _canonicalize_dataset_class(dataset_name: str) -> str:
    """Return the torch_geometric.datasets attribute for a dataset name."""

    dataset_key = dataset_name.replace("-", "_").lower()
    if dataset_key.startswith("ogbn_"):
        suffix = dataset_key.split("_", 1)[1]
        return "Ogbn" + "".join(part.capitalize() for part in suffix.split("_"))
    return "Planetoid"


def _load_dataset(
    dataset_name: str,
    *,
    data_root: str,
    datasets_mod: Any,
    dataset_loader: str | None,
    dataset_loader_kwargs: dict[str, Any] | None,
) -> Any:
    """Instantiate a PyG dataset for the requested graph benchmark."""

    kwargs: dict[str, Any] = dict(dataset_loader_kwargs or {})
    kwargs.setdefault("root", data_root)

    if dataset_loader:
        loader = load_object(dataset_loader)
        call_kwargs = dict(kwargs)
        call_kwargs.setdefault("dataset_name", dataset_name)
        call_kwargs.setdefault("name", dataset_name)
        return loader(**call_kwargs)

    attr_name = _canonicalize_dataset_class(dataset_name)
    loader = getattr(datasets_mod, attr_name, None)
    if loader is None:
        raise ValueError(
            f"torch_geometric.datasets does not provide a loader named '{attr_name}'"
        )

    call_kwargs = dict(kwargs)
    if attr_name == "Planetoid":
        call_kwargs.setdefault("name", dataset_name)
    return loader(**call_kwargs)


def _prepare_example_tensors(data: Any, *, device: str, torch_dtype: Any) -> Tuple[Any, Any]:
    """Move node features and edges to the requested device/dtype."""

    x = getattr(data, "x", None)
    edge_index = getattr(data, "edge_index", None)
    if x is None or edge_index is None:
        raise ValueError("dataset example must contain 'x' features and 'edge_index' graph structure")

    if hasattr(x, "to"):
        x = x.to(device=device, dtype=torch_dtype)
    if hasattr(edge_index, "to"):
        edge_index = edge_index.to(device=device)

    return x, edge_index


def _infer_in_out_channels(dataset: Any, data: Any) -> tuple[int, int]:
    """Derive model input/output sizes from the dataset."""

    features = getattr(data, "x", None)
    if features is None or not hasattr(features, "shape") or len(features.shape) < 2:
        raise ValueError("graph datasets must expose node features with shape (nodes, channels)")

    in_channels = int(features.shape[1])
    num_classes = getattr(dataset, "num_classes", None)
    if num_classes is None:
        raise ValueError("dataset object must define 'num_classes' for graph loaders")

    return in_channels, int(num_classes)


def _resolve_model_loader(
    *,
    default_attr: str,
    models_mod: Any,
    model_loader: str | None,
) -> Any:
    """Return the callable used to instantiate the graph neural network."""

    if model_loader:
        return load_object(model_loader)

    loader = getattr(models_mod, default_attr, None)
    if loader is None:
        raise ValueError(
            f"torch_geometric.nn.models does not provide a '{default_attr}' constructor"
        )
    return loader


def _load_pyg_model(
    *,
    default_model_attr: str,
    dataset_name: str,
    hidden_channels: int,
    num_layers: int,
    device: str | None,
    dtype: str | None,
    dropout: float | None,
    aggr: str | None,
    data_root: str,
    model_loader: str | None,
    model_loader_kwargs: dict[str, Any] | None,
    dataset_loader: str | None,
    dataset_loader_kwargs: dict[str, Any] | None,
) -> Tuple[Any, Tuple[Any, ...]]:
    """Shared implementation for the PyTorch Geometric experiment loaders."""

    torch, datasets_mod, models_mod = _import_pyg_modules()
    torch_dtype = resolve_dtype(dtype, torch)
    device_name = resolve_device(device, torch)

    dataset = _load_dataset(
        dataset_name,
        data_root=data_root,
        datasets_mod=datasets_mod,
        dataset_loader=dataset_loader,
        dataset_loader_kwargs=dataset_loader_kwargs,
    )
    if len(dataset) == 0:  # pragma: no cover - defensive
        raise ValueError(f"dataset '{dataset_name}' returned no samples")

    data = dataset[0]
    in_channels, out_channels = _infer_in_out_channels(dataset, data)

    loader = _resolve_model_loader(
        default_attr=default_model_attr,
        models_mod=models_mod,
        model_loader=model_loader,
    )

    model_kwargs: dict[str, Any] = dict(model_loader_kwargs or {})
    model_kwargs.setdefault("in_channels", in_channels)
    model_kwargs.setdefault("hidden_channels", hidden_channels)
    model_kwargs.setdefault("num_layers", num_layers)
    model_kwargs.setdefault("out_channels", out_channels)
    if dropout is not None:
        model_kwargs.setdefault("dropout", dropout)
    if aggr is not None:
        model_kwargs.setdefault("aggr", aggr)

    model = loader(**model_kwargs)
    if hasattr(model, "to"):
        model = model.to(device=device_name, dtype=torch_dtype)
    if hasattr(model, "eval"):
        model = model.eval()

    example = _prepare_example_tensors(data, device=device_name, torch_dtype=torch_dtype)
    return model, example


def load_pyg_gcn(
    *,
    dataset_name: str = "ogbn-arxiv",
    hidden_channels: int = 256,
    num_layers: int = 3,
    dropout: float | None = None,
    device: str | None = None,
    dtype: str | None = None,
    data_root: str = "data",
    model_loader: str | None = None,
    model_loader_kwargs: dict[str, Any] | None = None,
    dataset_loader: str | None = None,
    dataset_loader_kwargs: dict[str, Any] | None = None,
) -> Tuple[Any, Tuple[Any, ...]]:
    """Load a Graph Convolutional Network and example batch via PyTorch Geometric."""

    return _load_pyg_model(
        default_model_attr="GCN",
        dataset_name=dataset_name,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        device=device,
        dtype=dtype,
        dropout=dropout,
        aggr=None,
        data_root=data_root,
        model_loader=model_loader,
        model_loader_kwargs=model_loader_kwargs,
        dataset_loader=dataset_loader,
        dataset_loader_kwargs=dataset_loader_kwargs,
    )


def load_pyg_graphsage(
    *,
    dataset_name: str = "ogbn-arxiv",
    hidden_channels: int = 256,
    num_layers: int = 3,
    aggr: str = "mean",
    dropout: float | None = None,
    device: str | None = None,
    dtype: str | None = None,
    data_root: str = "data",
    model_loader: str | None = None,
    model_loader_kwargs: dict[str, Any] | None = None,
    dataset_loader: str | None = None,
    dataset_loader_kwargs: dict[str, Any] | None = None,
) -> Tuple[Any, Tuple[Any, ...]]:
    """Load a GraphSAGE model and example batch via PyTorch Geometric."""

    return _load_pyg_model(
        default_model_attr="GraphSAGE",
        dataset_name=dataset_name,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        device=device,
        dtype=dtype,
        dropout=dropout,
        aggr=aggr,
        data_root=data_root,
        model_loader=model_loader,
        model_loader_kwargs=model_loader_kwargs,
        dataset_loader=dataset_loader,
        dataset_loader_kwargs=dataset_loader_kwargs,
    )


def load_gcn(*, dataset: str = "ogbn-arxiv", **kwargs: Any) -> Tuple[Any, Tuple[Any, ...]]:
    """Backward compatible alias for :func:`load_pyg_gcn`."""

    kwargs = dict(kwargs)
    if "dataset_name" not in kwargs:
        kwargs["dataset_name"] = dataset
    return load_pyg_gcn(**kwargs)


def load_graphsage(*, dataset: str = "ogbn-arxiv", **kwargs: Any) -> Tuple[Any, Tuple[Any, ...]]:
    """Backward compatible alias for :func:`load_pyg_graphsage`."""

    kwargs = dict(kwargs)
    if "dataset_name" not in kwargs:
        kwargs["dataset_name"] = dataset
    return load_pyg_graphsage(**kwargs)
