import torch
import torch.nn as nn
from collections import OrderedDict
from .utils import permute_tensor
from typing import List


class ModelWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.device = (
            next(model.parameters()).device
            if list(model.parameters())
            else torch.device("cpu")
        )

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def permute_model_weights(self, permutation: List[int]):
        """
        Permutes the weights of the model's linear layers and their corresponding
        biases according to the given permutation.
        """
        perm_tensor = torch.LongTensor(permutation).to(self.device)
        d_model = len(permutation)

        # Create a map of layers to permute
        layers_to_permute = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                layers_to_permute[name] = module

        for name, layer in layers_to_permute.items():
            # Permute columns (input features)
            if layer.weight.shape[1] == d_model:
                layer.weight.data = layer.weight.data[:, perm_tensor]
                print(f"  - Permuted columns of layer: {name}")

            # Permute rows (output features)
            if layer.weight.shape[0] == d_model:
                layer.weight.data = layer.weight.data[perm_tensor, :]
                print(f"  - Permuted rows of layer: {name}")
                # Permute corresponding bias if it exists
                if layer.bias is not None:
                    layer.bias.data = layer.bias.data[perm_tensor]
                    print(f"  - Permuted bias of layer: {name}")

    def cuda(self, *args, **kwargs):
        self.device = torch.device("cuda")
        self.model.cuda(*args, **kwargs)
        return self

    def cpu(self, *args, **kwargs):
        self.device = torch.device("cpu")
        self.model.cpu(*args, **kwargs)
        return self


