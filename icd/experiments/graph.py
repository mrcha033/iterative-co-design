"""Graph neural network loaders for GCN, GraphSAGE, and other GNN architectures.

Provides loaders for PyTorch Geometric models with OGB dataset integration.
"""

from __future__ import annotations

import logging
from typing import Any, Tuple

__all__ = ["load_gcn", "load_graphsage"]

logger = logging.getLogger(__name__)


def load_gcn(
    *,
    dataset: str = "ogbn-arxiv",
    hidden_channels: int = 256,
    num_layers: int = 3,
    device: str | None = None,
    dtype: str | None = None,
    **kwargs: Any,
) -> Tuple[Any, Tuple[Any, ...]]:
    """Load GCN model on OGB dataset.

    Args:
        dataset: OGB dataset name (default: "ogbn-arxiv").
        hidden_channels: Hidden layer dimension.
        num_layers: Number of GCN layers.
        device: Target device.
        dtype: Data type.
        **kwargs: Additional arguments.

    Returns:
        Tuple of (model, (x, edge_index))
    """
    try:
        import torch
        import torch_geometric.nn as geom_nn
        from ogb.nodeproppred import PygNodePropPredDataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "torch_geometric and ogb are required for graph loaders. "
            "Install with: pip install torch_geometric ogb"
        ) from exc

    logger.info(f"Loading GCN on {dataset}")

    # Load dataset
    dataset_obj = PygNodePropPredDataset(name=dataset, root=kwargs.get("data_root", "data"))
    data = dataset_obj[0]

    # Create model
    class GCN(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
            super().__init__()
            self.convs = torch.nn.ModuleList()
            self.convs.append(geom_nn.GCNConv(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(geom_nn.GCNConv(hidden_channels, hidden_channels))
            self.convs.append(geom_nn.GCNConv(hidden_channels, out_channels))

        def forward(self, x, edge_index):
            for conv in self.convs[:-1]:
                x = conv(x, edge_index).relu()
            return self.convs[-1](x, edge_index)

    model = GCN(
        in_channels=data.x.shape[1],
        hidden_channels=hidden_channels,
        out_channels=dataset_obj.num_classes,
        num_layers=num_layers,
    )

    # Move to device
    if device:
        device_obj = torch.device(device)
        model = model.to(device_obj)
        data = data.to(device_obj)

    model.eval()

    return model, (data.x, data.edge_index)


def load_graphsage(
    *,
    dataset: str = "ogbn-arxiv",
    hidden_channels: int = 256,
    num_layers: int = 3,
    aggr: str = "mean",
    device: str | None = None,
    dtype: str | None = None,
    **kwargs: Any,
) -> Tuple[Any, Tuple[Any, ...]]:
    """Load GraphSAGE model on OGB dataset.

    Args:
        dataset: OGB dataset name (default: "ogbn-arxiv").
        hidden_channels: Hidden layer dimension.
        num_layers: Number of GraphSAGE layers.
        aggr: Aggregation function ("mean", "max", "add").
        device: Target device.
        dtype: Data type.
        **kwargs: Additional arguments.

    Returns:
        Tuple of (model, (x, edge_index))
    """
    try:
        import torch
        import torch_geometric.nn as geom_nn
        from ogb.nodeproppred import PygNodePropPredDataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "torch_geometric and ogb are required for graph loaders. "
            "Install with: pip install torch_geometric ogb"
        ) from exc

    logger.info(f"Loading GraphSAGE on {dataset} (aggr={aggr})")

    # Load dataset
    dataset_obj = PygNodePropPredDataset(name=dataset, root=kwargs.get("data_root", "data"))
    data = dataset_obj[0]

    # Create model
    class GraphSAGE(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels, num_layers, aggr="mean"):
            super().__init__()
            self.convs = torch.nn.ModuleList()
            self.convs.append(geom_nn.SAGEConv(in_channels, hidden_channels, aggr=aggr))
            for _ in range(num_layers - 2):
                self.convs.append(geom_nn.SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
            self.convs.append(geom_nn.SAGEConv(hidden_channels, out_channels, aggr=aggr))

        def forward(self, x, edge_index):
            for conv in self.convs[:-1]:
                x = conv(x, edge_index).relu()
            return self.convs[-1](x, edge_index)

    model = GraphSAGE(
        in_channels=data.x.shape[1],
        hidden_channels=hidden_channels,
        out_channels=dataset_obj.num_classes,
        num_layers=num_layers,
        aggr=aggr,
    )

    # Move to device
    if device:
        device_obj = torch.device(device)
        model = model.to(device_obj)
        data = data.to(device_obj)

    model.eval()

    return model, (data.x, data.edge_index)