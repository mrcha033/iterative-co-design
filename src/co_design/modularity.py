"""
Modularity calculation module for community detection and graph analysis.

This module provides tools for calculating modularity scores of graph partitions,
which is essential for evaluating the quality of community detection algorithms.
Modularity measures how well a partition divides a network into communities by
comparing the density of edges within communities to what would be expected in
a random network with the same degree distribution.

Key functions:
- calculate_modularity: Computes Newman's modularity score for a given partition
"""

import numpy as np
from typing import List


def calculate_modularity(
    correlation_matrix: np.ndarray, partition: List[List[int]]
) -> float:
    """
    Calculates the modularity of a partitioned graph for a signed network.

    This function implements the extended modularity definition for networks
    with both positive and negative weights (e.g., correlation matrices),
    as proposed by Gomez, Jensen, and Sarle (2009).

    Args:
        correlation_matrix: A square NumPy array representing the weighted adjacency
                            matrix of the graph.
        partition: A list of lists, where each inner list contains the indices of
                   nodes in a community.

    Returns:
        The modularity score (Q) of the given partition.
    """
    if correlation_matrix.shape[0] != correlation_matrix.shape[1]:
        raise ValueError("Correlation matrix must be square.")

    num_nodes = correlation_matrix.shape[0]
    
    # --- 1. Pre-process the graph ---
    # Create a copy to avoid modifying the original matrix
    W = correlation_matrix.copy()
    # Exclude self-loops from the calculation, as is standard
    np.fill_diagonal(W, 0)
    
    # Separate positive and negative weights
    W_plus = np.maximum(0, W)
    W_minus = np.maximum(0, -W)

    # --- 2. Calculate graph-level statistics ---
    # Total weight of all positive and negative edges
    m_plus = np.sum(W_plus) / 2.0
    m_minus = np.sum(W_minus) / 2.0
    
    # Epsilon for numerical stability
    eps = np.finfo(float).eps

    if m_plus + m_minus < eps:
        return 0.0

    # Strength of each node (sum of positive/negative weights)
    s_plus = np.sum(W_plus, axis=1)
    s_minus = np.sum(W_minus, axis=1)

    # --- 3. Build community membership matrix ---
    community_membership = np.full(num_nodes, -1, dtype=int)
    for i, community in enumerate(partition):
        if not community: continue # Skip empty communities
        community_membership[community] = i
    
    if np.any(community_membership == -1):
        # Find which nodes were not in the partition for a better error message
        missing_nodes = np.where(community_membership == -1)[0]
        raise ValueError(f"Partition must include all nodes. Missing nodes: {missing_nodes}")

    # Create a boolean matrix where S_ij is True if node i and j are in the same community
    same_community_matrix = community_membership[:, np.newaxis] == community_membership
    
    # --- 4. Calculate modularity using the Newman-Girvan formula extended for signed networks ---
    # Expected number of edges in the null model
    null_model_plus = np.outer(s_plus, s_plus) / (2 * m_plus + eps)
    null_model_minus = np.outer(s_minus, s_minus) / (2 * m_minus + eps)

    # Sum of (Observed - Expected) for pairs within the same community
    mod_matrix = (W_plus - null_model_plus) - (W_minus - null_model_minus)
    
    # Apply the community mask
    Q_matrix = mod_matrix * same_community_matrix
    
    # Final modularity score is the sum over the normalization factor
    modularity_score = np.sum(Q_matrix) / (2 * (m_plus + m_minus) + eps)
    
    return float(modularity_score)
