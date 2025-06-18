import numpy as np
from co_design.modularity import calculate_modularity


def test_modularity_perfect_clusters():
    """
    Tests the modularity calculation with a graph that has two perfect, disconnected clusters.
    The modularity score should be high.
    """
    # A correlation matrix representing two perfect clusters:
    # Cluster 1: nodes {0, 1}
    # Cluster 2: nodes {2, 3}
    correlation_matrix = np.array(
        [
            [1.0, 0.9, 0.1, 0.1],
            [0.9, 1.0, 0.1, 0.1],
            [0.1, 0.1, 1.0, 0.9],
            [0.1, 0.1, 0.9, 1.0],
        ]
    )

    # The correct partition that aligns with the clusters
    correct_partition = [[0, 1], [2, 3]]

    # The incorrect partition that splits the clusters
    incorrect_partition = [[0, 2], [1, 3]]

    modularity_correct = calculate_modularity(correlation_matrix, correct_partition)
    modularity_incorrect = calculate_modularity(correlation_matrix, incorrect_partition)

    # The modularity of the correct partition should be significantly higher
    assert modularity_correct > modularity_incorrect
    # It should also be a positive value, indicating good community structure
    assert modularity_correct > 0


def test_modularity_single_cluster():
    """
    Tests that the modularity of a graph with only one community is zero.
    """
    correlation_matrix = np.random.rand(4, 4)
    partition = [[0, 1, 2, 3]]  # All nodes in one cluster

    modularity = calculate_modularity(correlation_matrix, partition)

    assert np.isclose(modularity, 0.0)
