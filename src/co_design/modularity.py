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
    Calculates the modularity of a partitioned graph using vectorized operations.

    The graph's adjacency matrix is represented by the given correlation matrix.
    This function uses the formula for modularity defined by Newman (2006),
    optimized with NumPy vectorization for better performance on large graphs.

    Args:
        correlation_matrix: A square NumPy array representing the weighted adjacency
                            matrix of the graph (e.g., Pearson correlation).
        partition: A list of lists, where each inner list contains the indices of
                   nodes in a community. E.g., [[0, 1], [2, 3]].

    Returns:
        The modularity score (Q) of the given partition.
    """
    if correlation_matrix.shape[0] != correlation_matrix.shape[1]:
        raise ValueError("Correlation matrix must be square.")

    num_nodes = correlation_matrix.shape[0]

    # Create a community membership array for vectorized operations
    community_membership = np.full(num_nodes, -1, dtype=int)
    for community_idx, community in enumerate(partition):
        for node in community:
            if node >= num_nodes or node < 0:
                raise ValueError(
                    f"Node index {node} is out of range [0, {num_nodes - 1}]."
                )
            if community_membership[node] != -1:
                raise ValueError(f"Node {node} appears in multiple communities.")
            community_membership[node] = community_idx

    if np.any(community_membership == -1):
        raise ValueError("Partition must include all nodes from 0 to N-1 exactly once.")

    # Total weight of all edges in the graph
    # For a weighted, undirected graph, m is half the sum of all matrix elements.
    m = np.sum(correlation_matrix) / 2.0

    if m == 0:
        # If there are no edges, modularity is conventionally zero.
        return 0.0

    # Sum of weights of edges attached to each node (degree)
    k = np.sum(correlation_matrix, axis=1)

    # Create a matrix indicating which pairs of nodes are in the same community
    # This is the key vectorization: instead of nested loops, we use broadcasting
    same_community_matrix = (
        community_membership[:, np.newaxis] == community_membership[np.newaxis, :]
    )

    # Calculate the null model term: k[i] * k[j] / (2 * m)
    # Using outer product for vectorized computation
    null_model = np.outer(k, k) / (2 * m)

    # Calculate modularity: sum over all pairs where nodes are in same community
    # of (A[i,j] - k[i]*k[j]/(2*m))
    modularity_terms = (correlation_matrix - null_model) * same_community_matrix
    modularity_sum = np.sum(modularity_terms)

    return modularity_sum / (2 * m)
