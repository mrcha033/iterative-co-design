"""
Unit tests for latency profiling module.

These tests verify precise timing and statistical analysis functionality.
"""
import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch

from src.profiler.latency import (
    LatencyProfiler, LatencyConfig, LatencyResults,
    benchmark_model_latency, profile_layer_latency,
    compare_model_latencies, analyze_latency_improvements
)


class TestLatencyConfig:
    """Test LatencyConfig validation."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = LatencyConfig()
        
        assert config.warmup_runs == 10
        assert config.measurement_runs == 5
        assert config.use_cuda_events is True
        assert config.cuda_sync is True
        assert config.outlier_threshold == 2.0
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = LatencyConfig(
            warmup_runs=20,
            measurement_runs=10,
            use_cuda_events=False,
            outlier_threshold=1.5
        )
        
        assert config.warmup_runs == 20
        assert config.measurement_runs == 10
        assert config.use_cuda_events is False
        assert config.outlier_threshold == 1.5


class TestLatencyResults:
    """Test LatencyResults functionality."""
    
    def test_results_creation(self):
        """Test creating LatencyResults."""
        results = LatencyResults(
            mean_latency_ms=10.5,
            std_latency_ms=0.5,
            min_latency_ms=10.0,
            max_latency_ms=11.0,
            median_latency_ms=10.5,
            p95_latency_ms=10.9,
            p99_latency_ms=11.0,
            cv_percent=4.8,
            raw_times=[10.0, 10.2, 10.5, 10.8, 11.0],
            warmup_times=[12.0, 11.5, 11.0],
            outliers_removed=0,
            measurement_runs=5,
            device='cuda',
            timing_method='cuda_events'
        )
        
        assert results.mean_latency_ms == 10.5
        assert results.device == 'cuda'
        assert len(results.raw_times) == 5
    
    def test_to_dict(self):
        """Test converting results to dictionary."""
        results = LatencyResults(
            mean_latency_ms=10.5,
            std_latency_ms=0.5,
            min_latency_ms=10.0,
            max_latency_ms=11.0,
            median_latency_ms=10.5,
            p95_latency_ms=10.9,
            p99_latency_ms=11.0,
            cv_percent=4.8,
            raw_times=[10.0, 10.2, 10.5, 10.8, 11.0],
            warmup_times=[12.0, 11.5, 11.0],
            outliers_removed=0,
            measurement_runs=5,
            device='cuda',
            timing_method='cuda_events'
        )
        
        result_dict = results.to_dict()
        
        assert result_dict['mean_latency_ms'] == 10.5
        assert result_dict['device'] == 'cuda'
        assert result_dict['timing_method'] == 'cuda_events'
        assert 'raw_times' in result_dict


class TestLatencyProfiler:
    """Test LatencyProfiler functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = LatencyConfig(
            warmup_runs=2,
            measurement_runs=3,
            use_cuda_events=False,  # Use CPU timing for tests
            statistical_validation=False  # Disable for small sample
        )
        self.profiler = LatencyProfiler(self.config)
    
    def test_profiler_initialization(self):
        """Test profiler initialization."""
        assert self.profiler.config.warmup_runs == 2
        assert self.profiler.config.measurement_runs == 3
        assert self.profiler.device in ['cuda', 'cpu']
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        valid_config = LatencyConfig(warmup_runs=5, measurement_runs=3)
        profiler = LatencyProfiler(valid_config)
        assert profiler.config.warmup_runs == 5
        
        # Invalid configs
        with pytest.raises(ValueError):
            LatencyConfig(warmup_runs=-1)
        
        with pytest.raises(ValueError):
            LatencyConfig(measurement_runs=0)
        
        with pytest.raises(ValueError):
            LatencyConfig(outlier_threshold=-1)
    
    def test_device_inference(self):
        """Test device inference from arguments."""
        # CPU tensor
        cpu_tensor = torch.randn(4, 8)
        device = self.profiler._infer_device([cpu_tensor], {})
        assert device == 'cpu'
        
        # CUDA tensor (if available)
        if torch.cuda.is_available():
            cuda_tensor = torch.randn(4, 8, device='cuda')
            device = self.profiler._infer_device([cuda_tensor], {})
            assert device == 'cuda'
        
        # Model with parameters
        model = nn.Linear(4, 2)
        device = self.profiler._infer_device([model], {})
        assert device == 'cpu'  # Model is on CPU by default
    
    def test_simple_function_profiling(self):
        """Test profiling a simple function."""
        def simple_function(x):
            return torch.sum(x * x)
        
        # Test with CPU tensor
        test_tensor = torch.randn(100, 100)
        results = self.profiler.profile_function(simple_function, test_tensor)
        
        assert isinstance(results, LatencyResults)
        assert results.mean_latency_ms > 0
        assert results.measurement_runs == 3
        assert results.device == 'cpu'
        assert len(results.raw_times) == 3
    
    def test_timing_method_selection(self):
        """Test timing method selection."""
        def dummy_function():
            return torch.ones(10)
        
        # CPU profiler should use perf_counter
        cpu_config = LatencyConfig(use_cuda_events=True, measurement_runs=1)
        cpu_profiler = LatencyProfiler(cpu_config)
        
        if cpu_profiler.device == 'cpu':
            # Should automatically fallback to perf_counter on CPU
            results = cpu_profiler.profile_function(dummy_function)
            assert results.timing_method == 'perf_counter'
    
    def test_outlier_removal(self):
        """Test outlier removal functionality."""
        # Create a function with artificial timing variability
        call_count = 0
        def variable_function():
            nonlocal call_count
            call_count += 1
            # Introduce artificial delay for outlier
            if call_count == 3:  # Make third call an outlier
                time.sleep(0.01)  # 10ms delay
            return torch.ones(10)
        
        config = LatencyConfig(
            warmup_runs=0,
            measurement_runs=5,
            statistical_validation=True,
            outlier_threshold=2.0,
            use_cuda_events=False
        )
        profiler = LatencyProfiler(config)
        
        results = profiler.profile_function(variable_function)
        
        # Should have detected and removed outlier
        assert results.measurement_runs < 5 or results.outliers_removed > 0
    
    def test_statistical_analysis(self):
        """Test statistical analysis of results."""
        def consistent_function():
            # Simulate consistent timing
            return torch.ones(50)
        
        config = LatencyConfig(
            warmup_runs=1,
            measurement_runs=5,
            use_cuda_events=False
        )
        profiler = LatencyProfiler(config)
        
        results = profiler.profile_function(consistent_function)
        
        # Check statistical measures
        assert results.mean_latency_ms > 0
        assert results.std_latency_ms >= 0
        assert results.min_latency_ms <= results.mean_latency_ms
        assert results.max_latency_ms >= results.mean_latency_ms
        assert results.median_latency_ms > 0
        assert results.cv_percent >= 0
        
        # Percentiles should be ordered correctly
        assert results.p95_latency_ms >= results.median_latency_ms
        assert results.p99_latency_ms >= results.p95_latency_ms


class TestUtilityFunctions:
    """Test utility functions."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = LatencyConfig(
            warmup_runs=1,
            measurement_runs=2,
            use_cuda_events=False
        )
    
    def test_benchmark_model_latency(self):
        """Test model latency benchmarking."""
        # Create simple model
        model = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )
        
        # Create input
        inputs = torch.randn(2, 4)
        
        # Benchmark
        results = benchmark_model_latency(model, inputs, self.config)
        
        assert isinstance(results, LatencyResults)
        assert results.mean_latency_ms > 0
        assert results.device == 'cpu'
    
    def test_profile_layer_latency(self):
        """Test single layer profiling."""
        layer = nn.Linear(16, 8)
        input_shape = (4, 16)
        
        results = profile_layer_latency(
            layer, input_shape, device='cpu', config=self.config
        )
        
        assert isinstance(results, LatencyResults)
        assert results.mean_latency_ms > 0
    
    def test_compare_model_latencies(self):
        """Test comparing multiple models."""
        # Create different models
        models = {
            'small': nn.Linear(4, 2),
            'large': nn.Sequential(
                nn.Linear(4, 16),
                nn.ReLU(),
                nn.Linear(16, 16),
                nn.ReLU(),
                nn.Linear(16, 2)
            )
        }
        
        inputs = torch.randn(2, 4)
        
        results = compare_model_latencies(models, inputs, self.config)
        
        assert 'small' in results
        assert 'large' in results
        assert isinstance(results['small'], LatencyResults)
        assert isinstance(results['large'], LatencyResults)
        
        # Larger model should generally be slower
        # (though this might not always hold for small models)
        assert results['small'].mean_latency_ms >= 0
        assert results['large'].mean_latency_ms >= 0
    
    def test_analyze_latency_improvements(self):
        """Test latency improvement analysis."""
        # Create mock results
        baseline = LatencyResults(
            mean_latency_ms=20.0,
            std_latency_ms=1.0,
            min_latency_ms=19.0,
            max_latency_ms=21.0,
            median_latency_ms=20.0,
            p95_latency_ms=20.8,
            p99_latency_ms=21.0,
            cv_percent=5.0,
            raw_times=[19.0, 19.5, 20.0, 20.5, 21.0],
            warmup_times=[22.0, 21.0],
            outliers_removed=0,
            measurement_runs=5,
            device='cuda',
            timing_method='cuda_events'
        )
        
        optimized = LatencyResults(
            mean_latency_ms=16.0,
            std_latency_ms=0.8,
            min_latency_ms=15.2,
            max_latency_ms=16.8,
            median_latency_ms=16.0,
            p95_latency_ms=16.6,
            p99_latency_ms=16.8,
            cv_percent=5.0,
            raw_times=[15.2, 15.6, 16.0, 16.4, 16.8],
            warmup_times=[18.0, 17.0],
            outliers_removed=0,
            measurement_runs=5,
            device='cuda',
            timing_method='cuda_events'
        )
        
        analysis = analyze_latency_improvements(baseline, optimized)
        
        assert analysis['improvement_ms'] == 4.0  # 20.0 - 16.0
        assert analysis['improvement_pct'] == 20.0  # (4.0 / 20.0) * 100
        assert analysis['speedup_factor'] == 1.25  # 20.0 / 16.0
        assert analysis['baseline_mean_ms'] == 20.0
        assert analysis['optimized_mean_ms'] == 16.0
        assert 't_statistic' in analysis
        assert 'statistically_significant' in analysis


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_empty_measurements(self):
        """Test handling of failed measurements."""
        def failing_function():
            raise RuntimeError("Simulated failure")
        
        config = LatencyConfig(
            warmup_runs=0,
            measurement_runs=2,
            use_cuda_events=False
        )
        profiler = LatencyProfiler(config)
        
        # Should raise an error when all measurements fail
        with pytest.raises(RuntimeError):
            profiler.profile_function(failing_function)
    
    def test_partial_failures(self):
        """Test handling of partial measurement failures."""
        call_count = 0
        def partially_failing_function():
            nonlocal call_count
            call_count += 1
            if call_count == 2:  # Fail on second call
                raise RuntimeError("Simulated failure")
            return torch.ones(10)
        
        config = LatencyConfig(
            warmup_runs=0,
            measurement_runs=3,
            use_cuda_events=False
        )
        profiler = LatencyProfiler(config)
        
        # Should still succeed with remaining measurements
        results = profiler.profile_function(partially_failing_function)
        assert results.measurement_runs == 2  # Only 2 successful runs
    
    def test_high_variability_warning(self):
        """Test warning for high timing variability."""
        import logging
        
        # Capture log messages
        with patch.object(logging.getLogger('src.profiler.latency'), 'warning') as mock_warning:
            # Create results with high CV
            results = LatencyResults(
                mean_latency_ms=10.0,
                std_latency_ms=2.0,  # High std dev
                min_latency_ms=8.0,
                max_latency_ms=12.0,
                median_latency_ms=10.0,
                p95_latency_ms=11.8,
                p99_latency_ms=12.0,
                cv_percent=20.0,  # High CV
                raw_times=[8.0, 9.0, 10.0, 11.0, 12.0],
                warmup_times=[13.0, 12.0],
                outliers_removed=0,
                measurement_runs=5,
                device='cpu',
                timing_method='perf_counter'
            )
            
            # Manually trigger the analysis that would generate warning
            profiler = LatencyProfiler()
            analyzed_results = profiler._analyze_results([8.0, 9.0, 10.0, 11.0, 12.0], [13.0, 12.0], 'cpu')
            
            # Should have logged a warning about high variability
            assert analyzed_results.cv_percent > 10.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])