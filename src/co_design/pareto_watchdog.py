"""
Pareto-Optimal Watchdog for IASP
=================================

Monitors both perplexity and performance metrics to make optimal
permutation decisions based on the accuracy-latency tradeoff.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import deque

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass 
class PerformanceMetrics:
    """Performance metrics for a model state."""
    perplexity: float
    latency_ms: float  # Average inference latency in milliseconds
    memory_mb: float   # Peak memory usage in MB
    cache_hit_rate: float = 0.0  # L2 cache hit rate if available
    throughput: float = 0.0  # Samples per second
    
    def __post_init__(self):
        """Compute derived metrics."""
        if self.latency_ms > 0:
            self.throughput = 1000.0 / self.latency_ms


@dataclass
class ParetoConfig:
    """Configuration for Pareto-optimal decision making."""
    
    # Weights for different metrics (must sum to 1.0)
    ppl_weight: float = 0.5
    latency_weight: float = 0.3
    memory_weight: float = 0.2
    
    # Acceptable degradation thresholds
    max_ppl_increase: float = 0.05  # 5% max perplexity increase
    min_latency_improvement: float = 0.02  # 2% minimum latency improvement
    
    # Decision thresholds
    pareto_improvement_threshold: float = 0.01  # 1% overall improvement needed
    
    # Monitoring configuration
    warmup_iterations: int = 10  # Warmup iterations for latency measurement
    measure_iterations: int = 50  # Iterations for actual measurement
    
    def __post_init__(self):
        """Validate configuration."""
        total_weight = self.ppl_weight + self.latency_weight + self.memory_weight
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")


class ParetoWatchdog:
    """
    Monitors model performance across multiple dimensions and makes
    Pareto-optimal decisions about permutation acceptance.
    """
    
    def __init__(
        self,
        model: nn.Module,
        eval_dataloader: DataLoader,
        config: ParetoConfig,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.eval_dataloader = eval_dataloader
        self.config = config
        self.device = device or next(model.parameters()).device
        
        # Performance history
        self.baseline_metrics: Optional[PerformanceMetrics] = None
        self.current_metrics: Optional[PerformanceMetrics] = None
        self.metrics_history: List[PerformanceMetrics] = []
        
        # Cache profiling tools if available
        self._init_profiling_tools()
    
    def _init_profiling_tools(self):
        """Initialize hardware profiling tools if available."""
        self.has_cuda_events = torch.cuda.is_available()
        
        # Try to import Intel VTune for cache profiling
        try:
            import pyitt
            self.has_vtune = True
        except ImportError:
            self.has_vtune = False
            
        # Try to import NVIDIA NSight for GPU profiling
        try:
            import torch.cuda.nvtx as nvtx
            self.has_nvtx = True
        except ImportError:
            self.has_nvtx = False
    
    def measure_performance(
        self,
        compute_perplexity: bool = True
    ) -> PerformanceMetrics:
        """
        Measure comprehensive performance metrics.
        
        Args:
            compute_perplexity: Whether to compute perplexity (expensive)
            
        Returns:
            PerformanceMetrics instance
        """
        self.model.eval()
        
        # Measure perplexity if requested
        if compute_perplexity:
            from src.utils.evaluation import evaluate_perplexity
            ppl = evaluate_perplexity(self.model, self.eval_dataloader)
        else:
            ppl = self.current_metrics.perplexity if self.current_metrics else 0.0
        
        # Measure latency and memory
        latency_ms = self._measure_latency()
        memory_mb = self._measure_memory()
        
        # Measure cache hit rate if available
        cache_hit_rate = self._measure_cache_hit_rate()
        
        metrics = PerformanceMetrics(
            perplexity=ppl,
            latency_ms=latency_ms,
            memory_mb=memory_mb,
            cache_hit_rate=cache_hit_rate
        )
        
        return metrics
    
    def _measure_latency(self) -> float:
        """Measure average inference latency in milliseconds."""
        latencies = []
        
        # Get a single batch for consistent measurement
        batch = next(iter(self.eval_dataloader))
        if isinstance(batch, (list, tuple)):
            input_ids = batch[0].to(self.device)
        else:
            input_ids = batch['input_ids'].to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(self.config.warmup_iterations):
                _ = self.model(input_ids)
        
        # Measure
        if self.has_cuda_events and self.device.type == 'cuda':
            # Use CUDA events for GPU timing
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            with torch.no_grad():
                for _ in range(self.config.measure_iterations):
                    torch.cuda.synchronize()
                    start_event.record()
                    
                    _ = self.model(input_ids)
                    
                    end_event.record()
                    torch.cuda.synchronize()
                    
                    latencies.append(start_event.elapsed_time(end_event))
        else:
            # CPU timing
            with torch.no_grad():
                for _ in range(self.config.measure_iterations):
                    start_time = time.perf_counter()
                    _ = self.model(input_ids)
                    end_time = time.perf_counter()
                    latencies.append((end_time - start_time) * 1000)  # Convert to ms
        
        return float(np.median(latencies))
    
    def _measure_memory(self) -> float:
        """Measure peak memory usage in MB."""
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
            
            # Run inference
            with torch.no_grad():
                for i, batch in enumerate(self.eval_dataloader):
                    if i >= 10:  # Sample 10 batches
                        break
                    if isinstance(batch, (list, tuple)):
                        input_ids = batch[0].to(self.device)
                    else:
                        input_ids = batch['input_ids'].to(self.device)
                    _ = self.model(input_ids)
            
            peak_memory = torch.cuda.max_memory_allocated(self.device)
            return peak_memory / (1024 * 1024)  # Convert to MB
        else:
            # CPU memory measurement (less accurate)
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
    
    def _measure_cache_hit_rate(self) -> float:
        """Measure L2 cache hit rate if profiling tools are available."""
        # This is a placeholder - actual implementation would use
        # platform-specific profiling tools
        if self.has_vtune:
            # Would use Intel VTune APIs here
            return 0.0
        elif self.has_nvtx and self.device.type == 'cuda':
            # Would use NVIDIA profiling APIs here
            return 0.0
        else:
            return 0.0
    
    def set_baseline(self, metrics: Optional[PerformanceMetrics] = None) -> PerformanceMetrics:
        """Set baseline performance metrics."""
        if metrics is None:
            logger.info("Measuring baseline performance...")
            metrics = self.measure_performance(compute_perplexity=True)
        
        self.baseline_metrics = metrics
        self.current_metrics = metrics
        self.metrics_history = [metrics]
        
        logger.info(
            f"Baseline - PPL: {metrics.perplexity:.4f}, "
            f"Latency: {metrics.latency_ms:.2f}ms, "
            f"Memory: {metrics.memory_mb:.1f}MB"
        )
        
        return metrics
    
    def evaluate_permutation(
        self,
        permuted_model_state: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[bool, PerformanceMetrics, str]:
        """
        Evaluate whether a permutation should be accepted based on Pareto optimality.
        
        Args:
            permuted_model_state: Optional state dict to evaluate (if None, uses current model)
            
        Returns:
            Tuple of (should_accept, metrics, reason)
        """
        if self.baseline_metrics is None:
            raise RuntimeError("Baseline metrics not set. Call set_baseline() first.")
        
        # Temporarily load permuted state if provided
        original_state = None
        if permuted_model_state is not None:
            original_state = self.model.state_dict()
            self.model.load_state_dict(permuted_model_state)
        
        try:
            # Measure performance
            metrics = self.measure_performance(compute_perplexity=True)
            
            # Calculate relative changes
            ppl_change = (metrics.perplexity - self.baseline_metrics.perplexity) / self.baseline_metrics.perplexity
            latency_change = (metrics.latency_ms - self.baseline_metrics.latency_ms) / self.baseline_metrics.latency_ms
            memory_change = (metrics.memory_mb - self.baseline_metrics.memory_mb) / self.baseline_metrics.memory_mb
            
            # Check hard constraints
            if ppl_change > self.config.max_ppl_increase:
                return False, metrics, f"Perplexity increased by {ppl_change*100:.2f}% (max: {self.config.max_ppl_increase*100:.1f}%)"
            
            if latency_change > 0 and abs(latency_change) < self.config.min_latency_improvement:
                return False, metrics, f"Insufficient latency improvement: {-latency_change*100:.2f}% (min: {self.config.min_latency_improvement*100:.1f}%)"
            
            # Calculate Pareto score (lower is better)
            # Normalize changes to [0, 1] range
            normalized_ppl = ppl_change / self.config.max_ppl_increase
            normalized_latency = -latency_change  # Negative because lower latency is better
            normalized_memory = -memory_change / 0.1  # Assume 10% memory change is significant
            
            pareto_score = (
                self.config.ppl_weight * normalized_ppl +
                self.config.latency_weight * (1 - normalized_latency) +
                self.config.memory_weight * (1 - normalized_memory)
            )
            
            # Decision based on Pareto improvement
            baseline_pareto_score = self.config.ppl_weight  # Baseline normalized score
            improvement = baseline_pareto_score - pareto_score
            
            if improvement > self.config.pareto_improvement_threshold:
                reason = (
                    f"Pareto improvement: {improvement*100:.2f}% "
                    f"(PPL: {ppl_change*100:+.2f}%, Latency: {latency_change*100:+.2f}%, "
                    f"Memory: {memory_change*100:+.2f}%)"
                )
                return True, metrics, reason
            else:
                reason = (
                    f"Insufficient Pareto improvement: {improvement*100:.2f}% "
                    f"(threshold: {self.config.pareto_improvement_threshold*100:.1f}%)"
                )
                return False, metrics, reason
                
        finally:
            # Restore original state if needed
            if original_state is not None:
                self.model.load_state_dict(original_state)
    
    def update_metrics(self, metrics: PerformanceMetrics) -> None:
        """Update current metrics and history."""
        self.current_metrics = metrics
        self.metrics_history.append(metrics)
        
        # Keep only recent history
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of performance history."""
        if not self.metrics_history:
            return {}
        
        current = self.current_metrics
        baseline = self.baseline_metrics
        
        if not current or not baseline:
            return {}
        
        return {
            'baseline': {
                'perplexity': baseline.perplexity,
                'latency_ms': baseline.latency_ms,
                'memory_mb': baseline.memory_mb,
            },
            'current': {
                'perplexity': current.perplexity,
                'latency_ms': current.latency_ms,
                'memory_mb': current.memory_mb,
                'throughput': current.throughput,
            },
            'relative_change': {
                'perplexity_pct': ((current.perplexity - baseline.perplexity) / baseline.perplexity) * 100,
                'latency_pct': ((current.latency_ms - baseline.latency_ms) / baseline.latency_ms) * 100,
                'memory_pct': ((current.memory_mb - baseline.memory_mb) / baseline.memory_mb) * 100,
            },
            'history_length': len(self.metrics_history)
        } 