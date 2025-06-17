import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

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

class HDSLinear(nn.Module):
    """
    A wrapper for a linear layer that applies Hardware-Native Differentiable Sparsity (HDS)
    using the Gumbel-Top-K trick for N:M structured sparsity.
    """
    def __init__(self, linear_layer: nn.Linear, n: int = 2, m: int = 4, gumbel_temp: float = 1.0):
        super().__init__()
        self.linear = linear_layer
        self.n = n
        self.m = m
        self.gumbel_temp = gumbel_temp
        
        # Determine padding
        self.in_features = linear_layer.in_features
        self.padding = (self.m - (self.in_features % self.m)) % self.m
        
        # Scores are created for the padded dimension
        padded_features = self.in_features + self.padding
        self.scores = nn.Parameter(torch.randn(self.linear.out_features, padded_features))

    def get_sparsity_mask(self):
        """
        Generates the N:M structured sparsity mask from the learnable scores.
        """
        # Pad the scores to be divisible by M
        padded_scores = F.pad(self.scores, (0, self.padding))
        
        # Reshape for grouped selection
        grouped_scores = padded_scores.view(-1, self.m)
        
        # Apply Gumbel noise for stochasticity
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(grouped_scores)))
        
        # Select top N scores in each group
        _, top_indices = torch.topk(grouped_scores + gumbel_noise, self.n, dim=-1)
        
        # Create a binary mask from the top-k indices
        mask = torch.zeros_like(grouped_scores)
        mask.scatter_(-1, top_indices, 1)
        
        # Reshape and un-pad the mask to match the original weight dimensions
        full_mask = mask.view(self.linear.out_features, -1)
        return full_mask[:, :self.in_features]

    def forward(self, x):
        sparsity_mask = self.get_sparsity_mask()
        sparse_weight = self.linear.weight * sparsity_mask
        return F.linear(x, sparse_weight, self.linear.bias)

def _replace_linear_with_hds(module, hds_config):
    has_replaced = False
    target_layer_names = hds_config.get('target_layers', [])
    n = hds_config.get('n', 2)
    m = hds_config.get('m', 4)

    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and any(target in name for target in target_layer_names):
            print(f"  - Wrapping layer: {name} with {n}:{m} HDSLinear")
            setattr(module, name, HDSLinear(child, n=n, m=m))
            has_replaced = True
        else:
            has_replaced = _replace_linear_with_hds(child, hds_config) or has_replaced
    return has_replaced

def apply_hds(model: nn.Module, data_loader: torch.utils.data.DataLoader, config: dict):
    """
    Applies HDS to the model by replacing target linear layers and fine-tuning.
    """
    print(">>> Applying HDS by replacing Linear layers and fine-tuning...")

    hds_config = config.get('hds', {})
    _replace_linear_with_hds(model, hds_config)

    # Fine-tune the model to learn the sparsity masks
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get('learning_rate', 1e-5))
    num_epochs = hds_config.get('fine_tuning_epochs', 1)

    device = next(model.parameters()).device
    model.train()

    for epoch in range(num_epochs):
        print(f"  - HDS Fine-tuning Epoch {epoch + 1}/{num_epochs}")
        for batch in tqdm(data_loader, desc="Fine-tuning"):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch.get('labels', input_ids).to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

    model.eval()
    print(">>> HDS application complete.")
    return model 