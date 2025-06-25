import numpy as np
import pytest
from co_design.modularity import calculate_modularity


@pytest.mark.parametrize("cluster_strength,expected_higher", [
    (0.9, True),   # Strong correlation within clusters
    (0.8, True),   # Moderate correlation
    (0.6, True),   # Weak but still detectable
    (0.5, False),  # Too weak to significantly matter
])
def test_modularity_with_varying_cluster_strength(cluster_strength, expected_higher):
    """Test modularity calculation with different correlation strengths."""
    # Create correlation matrix with variable cluster strength
    weak_correlation = 0.1
    correlation_matrix = np.array([
        [1.0, cluster_strength, weak_correlation, weak_correlation],
        [cluster_strength, 1.0, weak_correlation, weak_correlation],
        [weak_correlation, weak_correlation, 1.0, cluster_strength],
        [weak_correlation, weak_correlation, cluster_strength, 1.0],
    ])
    
    correct_partition = [[0, 1], [2, 3]]
    incorrect_partition = [[0, 2], [1, 3]]
    
    modularity_correct = calculate_modularity(correlation_matrix, correct_partition)
    modularity_incorrect = calculate_modularity(correlation_matrix, incorrect_partition)
    
    if expected_higher:
        assert modularity_correct > modularity_incorrect
    else:
        # With very weak correlations, the difference might be negligible
        assert abs(modularity_correct - modularity_incorrect) < 0.1


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


@pytest.mark.parametrize("matrix_size,partition_type,expected_modularity", [
    (4, "single", 0.0),      # All nodes in one cluster
    (6, "single", 0.0),      # Larger single cluster
    (8, "individuals", -1.0), # Each node in its own cluster (approximate)
])
def test_modularity_edge_cases(matrix_size, partition_type, expected_modularity):
    """Test modularity calculation for edge cases."""
    # Generate random correlation matrix
    np.random.seed(42)  # For reproducibility
    correlation_matrix = np.random.rand(matrix_size, matrix_size)
    # Make it symmetric
    correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
    np.fill_diagonal(correlation_matrix, 1.0)
    
    if partition_type == "single":
        partition = [list(range(matrix_size))]
    elif partition_type == "individuals":
        partition = [[i] for i in range(matrix_size)]
    
    modularity = calculate_modularity(correlation_matrix, partition)
    
    if partition_type == "single":
        assert np.isclose(modularity, expected_modularity, atol=1e-10)
    elif partition_type == "individuals":
        # For individual nodes, modularity should be negative but exact value varies
        assert modularity < 0
