import torch

def permute_tensor(tensor: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
    """
    Permutes the first two dimensions of a square tensor according to a permutation array.

    Args:
        tensor: The input tensor (at least 2D and square).
        permutation: A 1D tensor representing the permutation.

    Returns:
        The permuted tensor.
    """
    assert tensor.dim() >= 2, "Tensor must be at least 2D"
    assert tensor.size(0) == tensor.size(1), "Tensor must be square"
    assert tensor.size(0) == len(permutation), "Tensor dimension and permutation length must match"
    
    # Ensure permutation tensor is on the same device as the input tensor
    perm_device = permutation.to(tensor.device)

    # Permute rows and columns
    permuted_tensor = tensor.index_select(0, perm_device)
    permuted_tensor = permuted_tensor.index_select(1, perm_device)
    return permuted_tensor 