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


def calculate_modularity(W: np.ndarray, partition: List[List[int]]) -> float:
    """
    Calculates the modularity of a partitioned graph for a signed network
    in a memory-efficient and vectorized way.
    """
    n = W.shape[0]
    if W.shape[1] != n:
        raise ValueError("Matrix must be square")

    # Work on a copy to avoid modifying the original matrix
    W = W.copy()
    # Exclude self-loops from the calculation
    np.fill_diagonal(W, 0)

    # Split into positive and negative weights once
    Wp, Wn = np.maximum(W, 0), np.maximum(-W, 0)
    mp, mn = Wp.sum() / 2.0, Wn.sum() / 2.0

    if mp + mn == 0:
        return 0.0

    sp, sn = Wp.sum(axis=1), Wn.sum(axis=1)

    # --- Vectorized community lookup (avoids N² matrices) ---
    com_id = -np.ones(n, dtype=int)
    # Use len() for safety, as partition elements are lists/arrays
    for cid, nodes in enumerate(partition):
        if len(nodes) == 0:
            continue
        com_id[nodes] = cid
        
    if (com_id == -1).any():
        missing_nodes = np.where(com_id == -1)[0].tolist()
        raise ValueError(f"Partition must include all nodes. Missing nodes: {missing_nodes}")

    # Sort nodes by community ID to create contiguous blocks
    order = np.argsort(com_id)
    Wp, Wn = Wp[order][:, order], Wn[order][:, order]
    sp, sn = sp[order], sn[order]

    # Get the cumulative borders of each community block
    # Filter out empty communities before calculating borders
    partition_sizes = [len(p) for p in partition if len(p) > 0]
    borders = np.cumsum(partition_sizes)
    borders = np.insert(borders, 0, 0)

    Q = 0.0
    eps = 1e-9  # Epsilon for numerical stability

    # Iterate over community blocks
    for i in range(len(borders) - 1):
        sl, sr = borders[i], borders[i+1]
        comm_slice = slice(sl, sr)
        
        # Sum of weights within the current community
        Wpc = Wp[comm_slice, comm_slice].sum()
        Wnc = Wn[comm_slice, comm_slice].sum()
        
        # Sum of strengths for nodes in the current community
        spc = sp[comm_slice].sum()
        snc = sn[comm_slice].sum()

        # Add this community's contribution to the total modularity
        Q += (Wpc - (spc**2) / (2 * mp + eps)) - (Wnc - (snc**2) / (2 * mn + eps))

    return Q / (2 * (mp + mn) + eps)
