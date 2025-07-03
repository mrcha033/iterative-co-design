"""
IASP Rollback and Safety Mechanisms
====================================

Provides checkpoint/rollback functionality for IASP permutations with
live perplexity monitoring to prevent model degradation.
"""

import copy
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.evaluation import evaluate_perplexity

logger = logging.getLogger(__name__)


@dataclass
class PermutationCheckpoint:
    """Stores information needed to rollback a permutation."""
    layer_name: str
    permutation: torch.Tensor
    inverse_permutation: torch.Tensor
    modularity: float
    timestamp: float
    param_shapes: Dict[str, torch.Size]


class IASPRollbackManager:
    """
    Manages permutation checkpoints and provides rollback functionality.
    
    Monitors perplexity after each permutation and can rollback if degradation
    is detected.
    """
    
    def __init__(
        self,
        model: nn.Module,
        eval_dataloader: DataLoader,
        max_ppl_increase: float = 0.05,  # 5% max perplexity increase
        checkpoint_interval: int = 5,      # Checkpoint every N layers
    ):
        self.model = model
        self.eval_dataloader = eval_dataloader
        self.max_ppl_increase = max_ppl_increase
        self.checkpoint_interval = checkpoint_interval
        
        self.checkpoints: List[PermutationCheckpoint] = []
        self.baseline_ppl: Optional[float] = None
        self.current_ppl: Optional[float] = None
        
        # Store original state dict for full rollback
        self.original_state = copy.deepcopy(model.state_dict())
        
    def set_baseline_perplexity(self, ppl: Optional[float] = None) -> float:
        """Set or compute baseline perplexity before permutations."""
        if ppl is not None:
            self.baseline_ppl = ppl
        else:
            logger.info("Computing baseline perplexity...")
            self.baseline_ppl = evaluate_perplexity(self.model, self.eval_dataloader)
        
        self.current_ppl = self.baseline_ppl
        logger.info(f"Baseline perplexity: {self.baseline_ppl:.4f}")
        return self.baseline_ppl
    
    def add_checkpoint(
        self,
        layer_name: str,
        permutation: torch.Tensor,
        modularity: float,
        affected_params: Dict[str, torch.nn.Parameter]
    ) -> None:
        """Add a permutation checkpoint."""
        checkpoint = PermutationCheckpoint(
            layer_name=layer_name,
            permutation=permutation.clone(),
            inverse_permutation=torch.argsort(permutation),
            modularity=modularity,
            timestamp=torch.cuda.Event(enable_timing=True).elapsed_time(torch.cuda.Event())
            if torch.cuda.is_available() else 0.0,
            param_shapes={name: param.shape for name, param in affected_params.items()}
        )
        self.checkpoints.append(checkpoint)
        
        # Check if we should evaluate perplexity
        if len(self.checkpoints) % self.checkpoint_interval == 0:
            self._check_perplexity()
    
    def _check_perplexity(self) -> None:
        """Check current perplexity and rollback if necessary."""
        logger.info(f"Evaluating perplexity after {len(self.checkpoints)} permutations...")
        self.current_ppl = evaluate_perplexity(self.model, self.eval_dataloader)
        
        ppl_increase = (self.current_ppl - self.baseline_ppl) / self.baseline_ppl
        logger.info(f"Current perplexity: {self.current_ppl:.4f} "
                   f"(+{ppl_increase*100:.2f}% from baseline)")
        
        if ppl_increase > self.max_ppl_increase:
            logger.warning(f"Perplexity increased by {ppl_increase*100:.2f}%, "
                          f"exceeding threshold of {self.max_ppl_increase*100:.2f}%")
            self._rollback_to_last_good_state()
    
    def _rollback_to_last_good_state(self) -> None:
        """Rollback to the last checkpoint with acceptable perplexity."""
        logger.warning("Initiating rollback to last good state...")
        
        # Binary search for the last good checkpoint
        left, right = 0, len(self.checkpoints) - 1
        last_good = -1
        
        while left <= right:
            mid = (left + right) // 2
            
            # Apply permutations up to mid
            self._apply_checkpoints_up_to(mid)
            
            # Check perplexity
            test_ppl = evaluate_perplexity(self.model, self.eval_dataloader)
            ppl_increase = (test_ppl - self.baseline_ppl) / self.baseline_ppl
            
            if ppl_increase <= self.max_ppl_increase:
                last_good = mid
                left = mid + 1
            else:
                right = mid - 1
        
        # Rollback to last good state
        if last_good >= 0:
            self._apply_checkpoints_up_to(last_good)
            self.checkpoints = self.checkpoints[:last_good + 1]
            logger.info(f"Rolled back to checkpoint {last_good + 1}/{len(self.checkpoints)}")
        else:
            # Full rollback to original state
            self.full_rollback()
    
    def _apply_checkpoints_up_to(self, index: int) -> None:
        """Apply permutations from checkpoints up to given index."""
        # Start from original state
        self.model.load_state_dict(self.original_state)
        
        # Apply each checkpoint in order
        for checkpoint in self.checkpoints[:index + 1]:
            # This would need the actual permutation application logic
            # For now, this is a placeholder
            logger.debug(f"Reapplying permutation to {checkpoint.layer_name}")
    
    def full_rollback(self) -> None:
        """Completely rollback to original model state."""
        logger.warning("Performing full rollback to original model state")
        self.model.load_state_dict(self.original_state)
        self.checkpoints.clear()
        self.current_ppl = self.baseline_ppl
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of permutation history and current state."""
        return {
            'num_checkpoints': len(self.checkpoints),
            'baseline_ppl': self.baseline_ppl,
            'current_ppl': self.current_ppl,
            'ppl_increase': ((self.current_ppl - self.baseline_ppl) / self.baseline_ppl * 100
                           if self.baseline_ppl and self.current_ppl else 0),
            'avg_modularity': (sum(c.modularity for c in self.checkpoints) / len(self.checkpoints)
                             if self.checkpoints else 0),
            'layers_permuted': [c.layer_name for c in self.checkpoints]
        }


def create_safe_iasp_wrapper(
    iasp_func,
    model: nn.Module,
    dataloader: DataLoader,
    eval_dataloader: DataLoader,
    config: dict,
    max_ppl_increase: float = 0.05
):
    """
    Wrapper that adds safety monitoring to any IASP function.
    
    Args:
        iasp_func: The IASP function to wrap (e.g., run_iasp_on_mamba)
        model: Model to permute
        dataloader: Training dataloader for correlation computation
        eval_dataloader: Evaluation dataloader for perplexity monitoring
        config: IASP configuration
        max_ppl_increase: Maximum allowed perplexity increase (default 5%)
    
    Returns:
        Tuple of (permutation, modularity) from the IASP function
    """
    rollback_manager = IASPRollbackManager(
        model, eval_dataloader, max_ppl_increase=max_ppl_increase
    )
    
    # Set baseline
    rollback_manager.set_baseline_perplexity()
    
    try:
        # Run IASP with monitoring
        result = iasp_func(model, dataloader, config)
        
        # Final perplexity check
        final_ppl = evaluate_perplexity(model, eval_dataloader)
        ppl_increase = (final_ppl - rollback_manager.baseline_ppl) / rollback_manager.baseline_ppl
        
        if ppl_increase > max_ppl_increase:
            logger.error(f"Final perplexity check failed: {ppl_increase*100:.2f}% increase")
            rollback_manager.full_rollback()
            raise RuntimeError(f"IASP caused excessive perplexity increase: {ppl_increase*100:.2f}%")
        
        logger.info(f"IASP completed successfully. Final perplexity: {final_ppl:.4f} "
                   f"(+{ppl_increase*100:.2f}%)")
        
        return result
        
    except Exception as e:
        logger.error(f"Error during IASP: {e}")
        rollback_manager.full_rollback()
        raise 