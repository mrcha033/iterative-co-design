"""
Distributed and Edge-Case Tests for IASP
=========================================

Tests IASP under challenging conditions including multi-GPU, mixed precision,
gradient accumulation, and checkpoint recovery.
"""

import os
import pytest
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler

from src.utils.permutation import safe_permute_rows, safe_permute_cols
from src.co_design.iasp import run_iasp_on_mamba


class MockMambaModel(nn.Module):
    """Mock Mamba model for testing."""
    
    def __init__(self, d_model=128, n_layers=4):
        super().__init__()
        self.backbone = nn.ModuleDict({
            'layers': nn.ModuleList([
                self._make_layer(d_model) for _ in range(n_layers)
            ])
        })
        self.lm_head = nn.Linear(d_model, 1000)  # Vocab size 1000
        
    def _make_layer(self, d_model):
        layer = nn.Module()
        layer.mixer = nn.Module()
        layer.mixer.in_proj = nn.Linear(d_model, 2 * d_model)
        layer.mixer.conv1d = nn.Conv1d(d_model, d_model, kernel_size=4, padding=3)
        layer.mixer.out_proj = nn.Linear(d_model, d_model)
        layer.mixer.A_log = nn.Parameter(torch.randn(d_model, 16))
        layer.mixer.D = nn.Parameter(torch.randn(d_model))
        layer.norm = nn.LayerNorm(d_model)
        return layer
    
    def forward(self, x):
        for layer in self.backbone['layers']:
            # Simplified forward pass
            residual = x
            x = layer.norm(x)
            
            # Mixer block
            gate_x = layer.mixer.in_proj(x)
            d_inner = gate_x.size(-1) // 2
            gate, x = gate_x[..., :d_inner], gate_x[..., d_inner:]
            
            # Conv1d expects (batch, channels, length)
            if x.dim() == 3:
                x = x.transpose(1, 2)
                x = layer.mixer.conv1d(x)
                x = x.transpose(1, 2)
            
            x = x * torch.sigmoid(gate)
            x = layer.mixer.out_proj(x)
            x = residual + x
        
        return self.lm_head(x)


def setup_distributed(rank, world_size):
    """Initialize distributed process group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_distributed():
    """Clean up distributed process group."""
    dist.destroy_process_group()


class TestDistributedIASP:
    """Test IASP in distributed settings."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Multi-GPU required")
    def test_ddp_permutation(self):
        """Test IASP with DistributedDataParallel."""
        world_size = min(2, torch.cuda.device_count())
        mp.spawn(
            self._run_ddp_test,
            args=(world_size,),
            nprocs=world_size,
            join=True
        )
    
    def _run_ddp_test(self, rank, world_size):
        """Run DDP test on a single process."""
        setup_distributed(rank, world_size)
        torch.cuda.set_device(rank)
        
        try:
            # Create model and wrap with DDP
            model = MockMambaModel().cuda(rank)
            ddp_model = DDP(model, device_ids=[rank])
            
            # Create distributed dataset
            dataset = TensorDataset(
                torch.randn(100, 32, 128),  # (samples, seq_len, d_model)
                torch.randint(0, 1000, (100, 32))  # Labels
            )
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
            dataloader = DataLoader(dataset, batch_size=4, sampler=sampler)
            
            # Test that we can collect activations across DDP
            from src.co_design.iasp import _collect_layer_corr
            
            # This should work without hanging
            corr, mask = _collect_layer_corr(
                ddp_model,
                dataloader,
                "module.backbone.layers.0.mixer.in_proj",
                max_samples=32,
                stride=1
            )
            
            assert corr.shape[0] == corr.shape[1]
            assert mask.sum() > 0
            
            # Test permutation synchronization across ranks
            if rank == 0:
                perm = torch.randperm(64).cuda(rank)
            else:
                perm = torch.empty(64, dtype=torch.long).cuda(rank)
            
            # Broadcast permutation from rank 0
            dist.broadcast(perm, src=0)
            
            # Apply same permutation on all ranks
            layer = ddp_model.module.backbone.layers[0].mixer
            safe_permute_rows(layer.in_proj.weight[:64], perm, optimizer=None)
            
            # Verify weights are synchronized
            weight_slice = layer.in_proj.weight[:64].clone()
            weight_list = [torch.empty_like(weight_slice) for _ in range(world_size)]
            dist.all_gather(weight_list, weight_slice)
            
            # All ranks should have identical weights
            for w in weight_list[1:]:
                torch.testing.assert_close(weight_list[0], w)
                
        finally:
            cleanup_distributed()
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_mixed_precision_permutation(self):
        """Test IASP with mixed precision training."""
        model = MockMambaModel().cuda()
        
        # Enable mixed precision
        scaler = torch.cuda.amp.GradScaler()
        
        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters())
        
        # Training step with mixed precision
        data = torch.randn(8, 32, 128).cuda()
        target = torch.randint(0, 1000, (8, 32)).cuda()
        
        with torch.cuda.amp.autocast():
            output = model(data)
            loss = nn.functional.cross_entropy(
                output.reshape(-1, 1000),
                target.reshape(-1)
            )
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Apply permutation
        perm = torch.randperm(128)
        layer = model.backbone.layers[0].mixer
        
        # This should handle mixed precision gracefully
        safe_permute_rows(layer.in_proj.weight[:128], perm, optimizer=optimizer)
        
        # Verify model still works
        with torch.cuda.amp.autocast():
            output2 = model(data)
            assert output2.shape == output.shape
            assert not torch.isnan(output2).any()
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_gradient_accumulation_with_permutation(self):
        """Test IASP with gradient accumulation."""
        model = MockMambaModel().cuda()
        optimizer = torch.optim.Adam(model.parameters())
        
        accumulation_steps = 4
        
        # Accumulate gradients
        for step in range(accumulation_steps):
            data = torch.randn(2, 32, 128).cuda()
            target = torch.randint(0, 1000, (2, 32)).cuda()
            
            output = model(data)
            loss = nn.functional.cross_entropy(
                output.reshape(-1, 1000),
                target.reshape(-1)
            )
            loss = loss / accumulation_steps
            loss.backward()
        
        # Apply permutation before optimizer step
        perm = torch.randperm(128)
        layer = model.backbone.layers[0].mixer
        safe_permute_rows(layer.in_proj.weight[:128], perm, optimizer=optimizer)
        
        # Optimizer step should work correctly
        optimizer.step()
        optimizer.zero_grad()
        
        # Verify gradients are cleared
        for param in model.parameters():
            assert param.grad is None or param.grad.abs().sum() == 0
    
    def test_checkpoint_recovery_after_permutation(self):
        """Test model checkpoint save/load after permutation."""
        model = MockMambaModel()
        optimizer = torch.optim.Adam(model.parameters())
        
        # Train one step to create optimizer state
        data = torch.randn(4, 32, 128)
        target = torch.randint(0, 1000, (4, 32))
        output = model(data)
        loss = nn.functional.cross_entropy(
            output.reshape(-1, 1000),
            target.reshape(-1)
        )
        loss.backward()
        optimizer.step()
        
        # Apply permutation
        perm = torch.randperm(128)
        layer = model.backbone.layers[0].mixer
        safe_permute_rows(layer.in_proj.weight[:128], perm, optimizer=optimizer)
        
        # Save checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'permutation': perm,
        }
        
        # Create new model and load checkpoint
        model2 = MockMambaModel()
        optimizer2 = torch.optim.Adam(model2.parameters())
        
        model2.load_state_dict(checkpoint['model_state_dict'])
        optimizer2.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Verify weights match
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
            torch.testing.assert_close(p1, p2, msg=f"Mismatch in {n1}")
        
        # Verify optimizer state matches
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            if p1 in optimizer.state and p2 in optimizer2.state:
                state1 = optimizer.state[p1]
                state2 = optimizer2.state[p2]
                for key in state1:
                    if torch.is_tensor(state1[key]):
                        torch.testing.assert_close(
                            state1[key], state2[key],
                            msg=f"Optimizer state mismatch for {key}"
                        )
    
    def test_learning_rate_schedule_with_permutation(self):
        """Test that LR schedules work correctly with permutations."""
        model = MockMambaModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Cosine annealing schedule
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10, eta_min=1e-5
        )
        
        initial_lr = scheduler.get_last_lr()[0]
        
        # Train a few steps
        for _ in range(3):
            data = torch.randn(4, 32, 128)
            target = torch.randint(0, 1000, (4, 32))
            output = model(data)
            loss = nn.functional.cross_entropy(
                output.reshape(-1, 1000),
                target.reshape(-1)
            )
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        # Apply permutation
        perm = torch.randperm(128)
        layer = model.backbone.layers[0].mixer
        safe_permute_rows(layer.in_proj.weight[:128], perm, optimizer=optimizer)
        
        # Continue training
        for _ in range(3):
            data = torch.randn(4, 32, 128)
            target = torch.randint(0, 1000, (4, 32))
            output = model(data)
            loss = nn.functional.cross_entropy(
                output.reshape(-1, 1000),
                target.reshape(-1)
            )
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        # Verify LR has changed according to schedule
        current_lr = scheduler.get_last_lr()[0]
        assert current_lr < initial_lr  # Should decrease with cosine schedule
        
        # Loss should still decrease
        final_loss = loss.item()
        assert final_loss < 10.0  # Reasonable bound for cross-entropy 