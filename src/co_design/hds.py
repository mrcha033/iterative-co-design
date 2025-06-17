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
    using the Gumbel-Top-K trick for 2:4 structured sparsity.
    """
    def __init__(self, linear_layer: nn.Linear, gumbel_temp: float = 1.0):
        super().__init__()
        self.linear = linear_layer
        self.scores = nn.Parameter(torch.randn_like(linear_layer.weight))
        self.gumbel_temp = gumbel_temp
    
    def get_sparsity_mask(self):
        """
        Generates the 2:4 structured sparsity mask from the learnable scores.
        """
        # Ensure the weights can be reshaped into groups of 4
        if self.scores.shape[1] % 4 != 0:
            # If not divisible by 4, we can't apply 2:4 sparsity.
            # In a real scenario, you might pad or handle this differently.
            # For now, we'll return a dense mask.
            return torch.ones_like(self.scores)
            
        # Reshape scores into groups of 4 for Top-K selection
        grouped_scores = self.scores.view(-1, 4)
        
        # Apply Gumbel-Softmax for differentiable sampling
        gumbel_samples = F.gumbel_softmax(grouped_scores, tau=self.gumbel_temp, hard=True, dim=-1)
        
        # Select top 2 values in each group to create the mask
        # Gumbel-softmax with hard=True already gives one-hot vectors. To get Top-K, 
        # we can relax and use topk on the scores with gumbel noise.
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(grouped_scores)))
        _, top_indices = torch.topk(grouped_scores + gumbel_noise, 2, dim=-1)
        
        mask = torch.zeros_like(grouped_scores)
        mask.scatter_(-1, top_indices, 1)
        
        return mask.view_as(self.scores)

    def forward(self, x):
        sparsity_mask = self.get_sparsity_mask()
        sparse_weight = self.linear.weight * sparsity_mask
        return F.linear(x, sparse_weight, self.linear.bias)

def apply_hds(model: nn.Module, data_loader: torch.utils.data.DataLoader, config: dict):
    """
    Applies HDS to the model by replacing target linear layers and fine-tuning.
    """
    print(">>> Applying HDS by replacing Linear layers and fine-tuning...")

    # Recursively find and replace target linear layers
    hds_config = config.get('hds', {})
    target_layer_names = hds_config.get('target_layers', ['output.dense']) # Default target
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(target_name in name for target_name in target_layer_names):
            print(f"  - Wrapping layer: {name}")
            # This requires structured replacement, e.g., using a helper function
            # For simplicity, we'll assume a direct replacement is possible.
            # In a real transformer, you need to set attributes carefully.
            # e.g., parent_module.child = HDSLinear(module)
            
            # This is a simplified example of replacement.
            # A robust implementation would parse the name and set the attribute.
            name_parts = name.split('.')
            parent = model
            for part in name_parts[:-1]:
                parent = getattr(parent, part)
            
            original_layer = getattr(parent, name_parts[-1])
            setattr(parent, name_parts[-1], HDSLinear(original_layer))

    # Fine-tune the model to learn the sparsity masks
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get('learning_rate', 1e-5))
    num_epochs = hds_config.get('fine_tuning_epochs', 1) # Just one epoch for demonstration

    device = next(model.parameters()).device
    model.train()

    for epoch in range(num_epochs):
        print(f"  - HDS Fine-tuning Epoch {epoch + 1}/{num_epochs}")
        for batch in tqdm(data_loader, desc="Fine-tuning"):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch.get('labels', input_ids).to(device) # For LM or classification

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

    model.eval()
    print(">>> HDS application complete.")
    return model 