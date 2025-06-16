import torch
import torch.nn as nn
import torch.nn.functional as F

def gumbel_topk(logits: torch.Tensor, k: int, temperature: float = 1.0) -> torch.Tensor:
    """
    Differentiable Top-K selection using the Gumbel-Softmax trick.
    
    Args:
        logits: A tensor of raw scores (..., N).
        k: The number of items to select from N.
        temperature: The Gumbel-Softmax temperature. A lower temperature makes the
                     selection closer to a true one-hot encoding.
                     
    Returns:
        A tensor of the same shape as logits with a binary mask of K selected items.
        The gradients can flow back to the original logits.
    """
    # Gumbel-Softmax sampling
    gumbels = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
    gumbels = (logits + gumbels) / temperature
    y_soft = F.softmax(gumbels, dim=-1)

    # Straight-Through Estimator for Top-K
    # Get the top-k indices from the softened probabilities
    _, top_k_indices = torch.topk(y_soft, k, dim=-1)
    # Create a hard, binary mask
    y_hard = torch.zeros_like(logits).scatter_(-1, top_k_indices, 1.0)
    
    # Combine the hard mask with the soft probabilities for gradient flow
    y_out = y_hard - y_soft.detach() + y_soft
    return y_out

class HDSSparseLinear(nn.Module):
    """
    A Linear layer with Hardware-Native Differentiable Sparsity (HDS).
    This layer applies a learned N:M structured sparsity mask to its weight
    during the forward pass, enabling end-to-end fine-tuning for structured sparsity.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, n: int = 2, m: int = 4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n = n
        self.m = m

        if in_features % m != 0:
            raise ValueError(f"in_features ({in_features}) must be divisible by M ({m}).")

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        # Define the learnable logits for the sparsity mask.
        # There is one set of M logits for each block of M input features.
        num_blocks = in_features // m
        self.sparsity_logits = nn.Parameter(torch.randn(num_blocks, m))

    def get_mask(self, temperature: float = 1.0) -> torch.Tensor:
        """
        Generates the N:M sparsity mask from the internal logits.
        The mask is broadcast across the output features.
        """
        # Generate a mask for the blocks, shape (num_blocks, M)
        mask_blocks = gumbel_topk(self.sparsity_logits, self.n, temperature)
        
        # Reshape and repeat to match the full weight matrix dimensions
        # The mask is repeated for each of the `out_features` rows.
        mask = mask_blocks.repeat(self.out_features, 1) # Shape: (out_features * num_blocks, M)
        return mask.view(self.out_features, self.in_features) # Reshape to (out_features, in_features)

    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Performs the forward pass with the learned sparse weight.
        """
        mask = self.get_mask(temperature=temperature)
        sparse_weight = self.linear.weight * mask
        
        return F.linear(x, sparse_weight, self.linear.bias) 