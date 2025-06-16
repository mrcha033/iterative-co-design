import torch
import torch.nn as nn
from collections import OrderedDict
from .utils import permute_tensor

class ModelWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def permute_model_weights(self, permutation: list | torch.Tensor):
        """
        Permutes the weights of the model's layers based on a permutation array.

        This method identifies 2D square weight tensors in the model's state_dict
        that match the permutation's dimension and applies the given permutation to their
        rows and columns.

        Args:
            permutation (list or torch.Tensor): The permutation to apply.
        """
        if isinstance(permutation, list):
            # Using model's device to create the tensor
            example_param = next(self.model.parameters())
            permutation = torch.tensor(permutation, dtype=torch.long, device=example_param.device)

        target_dim = len(permutation)
        current_state_dict = self.model.state_dict()
        new_state_dict = OrderedDict()

        permuted_layers = []
        for name, params in current_state_dict.items():
            # Identify square 2D tensors that match the permutation dimension
            if params.dim() == 2 and params.size(0) == target_dim and params.size(1) == target_dim:
                permuted_layers.append(name)
                permuted_params = permute_tensor(params, permutation)
                new_state_dict[name] = permuted_params
            else:
                new_state_dict[name] = params
        
        if not permuted_layers:
            print(f"Warning: No layers found with dimension {target_dim}x{target_dim}. No weights were permuted.")
        else:
            print(f"Permuted layers: {permuted_layers}")

        self.model.load_state_dict(new_state_dict) 