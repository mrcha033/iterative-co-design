import pytest
import torch
import torch.nn as nn
import copy
from torch.utils.data import DataLoader, TensorDataset
from src.co_design.iasp import run_iasp_on_mamba

# A more realistic mock Mamba block for testing
class MockMambaBlock(nn.Module):
    def __init__(self, d_model, d_inner, d_state, dt_rank):
        super().__init__()
        self.d_inner = d_inner
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=True)
        # Projects from d_inner -> dt_rank
        self.dt_proj = nn.Linear(d_inner, dt_rank, bias=True)
        # Parameter shapes that depend on d_inner as an input dimension
        self.A_log = nn.Parameter(torch.randn(d_state, d_inner))
        self.D = nn.Parameter(torch.randn(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        # Dummy attributes to satisfy the permutation function's layer check
        self.x_proj = None
        self.conv1d = nn.Conv1d(in_channels=d_inner, out_channels=d_inner, kernel_size=3, padding=1)

    def forward(self, x):
        x_all = self.in_proj(x)
        x_proj = x_all[..., :self.d_inner]
        out = self.out_proj(x_proj)
        return out

class MockMambaForTest(nn.Module):
    def __init__(self, d_model=32, d_inner=64, d_state=8, n_layers=2, dt_rank=4):
        super().__init__()
        self.embedding = nn.Embedding(100, d_model)
        self.layers = nn.ModuleList(
            [MockMambaBlock(d_model, d_inner, d_state, dt_rank) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = x + layer(x)
        return self.norm(x)

@pytest.fixture
def mamba_model():
    return MockMambaForTest()

@pytest.fixture
def mamba_dataloader():
    # Dataloader must yield a dictionary
    input_ids = torch.randint(0, 100, (16, 10))
    dataset = TensorDataset(input_ids)
    
    class DictDataLoader(DataLoader):
        def __iter__(self):
            for batch in super().__iter__():
                yield {'input_ids': batch[0]}

    return DictDataLoader(dataset, batch_size=4)


def test_run_iasp_on_mamba_pipeline(mamba_model, mamba_dataloader):
    """
    Tests the end-to-end IASP pipeline for Mamba, ensuring it runs and
    maintains mathematical equivalence.
    """
    # 1. Get original output
    mamba_model.eval()
    input_sample = next(iter(mamba_dataloader))['input_ids']
    with torch.no_grad():
        original_output = mamba_model(input_sample)

    # 2. Create a copy and run the IASP pipeline
    permuted_model = copy.deepcopy(mamba_model)
    run_iasp_on_mamba(permuted_model, mamba_dataloader, cluster_size_range=(4, 16))

    # 3. Get output from permuted model
    permuted_model.eval()
    with torch.no_grad():
        permuted_output = permuted_model(input_sample)

    # 4. Verify outputs are the same
    assert torch.allclose(original_output, permuted_output, atol=1e-5), \
        "Model output changed after IASP pipeline."

    # 5. Verify weights have actually been permuted in the first block
    original_w = mamba_model.layers[0].in_proj.weight.data
    permuted_w = permuted_model.layers[0].in_proj.weight.data
    assert not torch.equal(original_w, permuted_w), \
        "in_proj weights were not permuted."
    
    original_out_w = mamba_model.layers[0].out_proj.weight.data
    permuted_out_w = permuted_model.layers[0].out_proj.weight.data
    assert not torch.equal(original_out_w, permuted_out_w), \
        "out_proj weights were not permuted."
    
    original_D = mamba_model.layers[0].D.data
    permuted_D = permuted_model.layers[0].D.data
    assert not torch.equal(original_D, permuted_D), \
        "D parameter was not permuted."
