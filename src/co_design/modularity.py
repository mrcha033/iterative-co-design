import numpy as np
from typing import List


def calculate_modularity(
    correlation_matrix: np.ndarray, partition: List[List[int]]
) -> float:
    """
    Calculates the modularity of a partitioned graph.

    The graph's adjacency matrix is represented by the given correlation matrix.
    This function uses the formula for modularity defined by Newman (2006).

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

    # Create a mapping from node index to community index
    node_to_community = {}
    for community_idx, community in enumerate(partition):
        for node in community:
            node_to_community[node] = community_idx

    if len(node_to_community) != num_nodes:
        raise ValueError("Partition must include all nodes from 0 to N-1 exactly once.")

    # Total weight of all edges in the graph
    # For a weighted, undirected graph, m is half the sum of all matrix elements.
    # We handle negative weights by just summing them.
    m = np.sum(correlation_matrix) / 2.0

    if m == 0:
        # If there are no edges, modularity is conventionally zero.
        return 0.0

    # Sum of weights of edges attached to each node (degree)
    k = np.sum(correlation_matrix, axis=1)

    modularity_sum = 0.0
    for i in range(num_nodes):
        for j in range(num_nodes):
            # Check if nodes i and j are in the same community
            if node_to_community.get(i) == node_to_community.get(j):
                modularity_sum += correlation_matrix[i, j] - (k[i] * k[j]) / (2 * m)

    return modularity_sum / (2 * m)
