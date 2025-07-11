"""
Simple GCN model implementation for the iterative co-design framework.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNModel(nn.Module):
    """
    Simple Graph Convolutional Network for node classification.
    Used for the ogbn-arxiv dataset experiments.
    """
    
    def __init__(
        self,
        num_features: int = 128,
        hidden_dim: int = 256,
        num_classes: int = 40,
        num_layers: int = 3,
        dropout: float = 0.5
    ):
        """
        Initialize the GCN model.
        
        Args:
            num_features: Number of input features
            hidden_dim: Hidden dimension size
            num_classes: Number of output classes
            num_layers: Number of GCN layers
            dropout: Dropout probability
        """
        super().__init__()
        
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Create GCN layers
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(GCNConv(num_features, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
        
        # Output layer
        self.layers.append(GCNConv(hidden_dim, num_classes))
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass through the GCN.
        
        Args:
            x: Node features [num_nodes, num_features]
            edge_index: Edge indices [2, num_edges]
            batch: Batch vector (optional)
            
        Returns:
            Node embeddings or predictions
        """
        # Apply GCN layers
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            
            # Apply activation and dropout (except for last layer)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = self.dropout_layer(x)
        
        return x
    
    def get_embeddings(self, x, edge_index, layer_idx=-2):
        """
        Get node embeddings from a specific layer.
        
        Args:
            x: Node features
            edge_index: Edge indices
            layer_idx: Layer index to extract embeddings from
            
        Returns:
            Node embeddings
        """
        # Forward pass up to specified layer
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            
            if i == layer_idx:
                return x
            
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = self.dropout_layer(x)
        
        return x