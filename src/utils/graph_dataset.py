"""
Graph dataset utilities for node classification tasks.
"""
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple
import numpy as np


class GraphDataset(Dataset):
    """
    Dataset wrapper for graph data (OGB datasets).
    Handles node classification tasks with train/valid/test splits.
    """
    
    def __init__(
        self,
        dataset,
        split: str = 'train',
        num_samples: Optional[int] = None
    ):
        """
        Initialize the graph dataset.
        
        Args:
            dataset: OGB dataset object
            split: Dataset split ('train', 'valid', 'test')
            num_samples: Maximum number of samples to use
        """
        self.dataset = dataset
        self.split = split
        self.num_samples = num_samples
        
        # Get the graph data
        self.graph, self.labels = dataset[0]
        
        # Get split indices
        split_idx = dataset.get_idx_split()
        self.split_indices = split_idx[split]
        
        # Limit samples if requested
        if num_samples is not None:
            self.split_indices = self.split_indices[:num_samples]
        
        print(f"Loaded {len(self.split_indices)} samples for {split} split")
    
    def __len__(self):
        """Get dataset length."""
        return len(self.split_indices)
    
    def __getitem__(self, idx):
        """Get a graph sample."""
        # Get node index
        node_idx = self.split_indices[idx]
        
        # Get node features
        x = self.graph.x[node_idx]
        
        # Get node label
        y = self.labels[node_idx]
        
        return {
            'x': x,
            'y': y,
            'node_idx': node_idx,
            'edge_index': self.graph.edge_index,
            'num_nodes': self.graph.num_nodes
        }
    
    def get_full_graph(self):
        """Get the full graph for batch processing."""
        return {
            'x': self.graph.x,
            'y': self.labels,
            'edge_index': self.graph.edge_index,
            'num_nodes': self.graph.num_nodes,
            'split_indices': self.split_indices
        }
    
    def get_graph_stats(self):
        """Get statistics about the graph."""
        return {
            'num_nodes': self.graph.num_nodes,
            'num_edges': self.graph.num_edges,
            'num_features': self.graph.num_features,
            'num_classes': self.dataset.num_classes,
            'split_size': len(self.split_indices),
            'avg_degree': float(self.graph.num_edges) / self.graph.num_nodes
        }