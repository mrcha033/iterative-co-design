"""
Integration tests for profiling modules with toy models.

These tests verify end-to-end profiling functionality with realistic models
and validate the integration between latency and hardware profiling.
"""
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import pytest
import torch
import torch.nn as nn
import numpy as np

from src.profiler.latency import (
    LatencyProfiler, LatencyConfig, benchmark_model_latency,
    compare_model_latencies, analyze_latency_improvements
)
from src.profiler.ncu import (
    NsightComputeProfiler, NCUConfig, collect_hardware_metrics
)
from src.profiler.calibration import (
    SystemCalibrator, CalibrationConfig, validate_system_performance
)


class ToyModel(nn.Module):
    """Simple toy model for testing."""
    
    def __init__(self, input_size=64, hidden_size=32, output_size=16):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.layers(x)


class ComplexToyModel(nn.Module):
    """More complex toy model for comparison testing."""
    
    def __init__(self, input_size=64):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 8, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16)
        )
    
    def forward(self, x):
        # x shape: (batch, features) -> (batch, 1, features)
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


class TestLatencyProfilingIntegration:
    """Integration tests for latency profiling."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = LatencyConfig(
            warmup_runs=2,
            measurement_runs=3,
            use_cuda_events=torch.cuda.is_available(),
            cuda_sync=True,
            statistical_validation=True
        )
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def test_single_model_profiling(self):
        """Test profiling a single toy model."""
        model = ToyModel(input_size=64, hidden_size=32, output_size=16)
        model = model.to(self.device)
        model.eval()
        
        # Create input data
        batch_size = 4
        inputs = torch.randn(batch_size, 64, device=self.device)
        
        # Profile the model
        results = benchmark_model_latency(model, inputs, self.config)
        
        # Verify results
        assert results.mean_latency_ms > 0
        assert results.std_latency_ms >= 0
        assert results.measurement_runs == 3
        assert results.device == self.device
        assert len(results.raw_times) == 3
        
        # Check statistical measures
        assert results.min_latency_ms <= results.mean_latency_ms <= results.max_latency_ms
        assert results.cv_percent >= 0
        
        print(f"Model profiling: {results.mean_latency_ms:.2f}±{results.std_latency_ms:.2f} ms")
    
    def test_model_comparison_profiling(self):
        """Test comparing multiple models."""
        # Create models of different complexity
        models = {
            'simple': ToyModel(input_size=32, hidden_size=16, output_size=8),
            'medium': ToyModel(input_size=64, hidden_size=32, output_size=16),
            'complex': ComplexToyModel(input_size=64)
        }
        
        # Move models to device
        for name, model in models.items():
            models[name] = model.to(self.device)
            models[name].eval()
        
        # Create appropriate inputs for each model
        inputs = {
            'simple': torch.randn(2, 32, device=self.device),
            'medium': torch.randn(2, 64, device=self.device), 
            'complex': torch.randn(2, 64, device=self.device)
        }
        
        # Profile each model
        results = {}
        for name, model in models.items():
            results[name] = benchmark_model_latency(model, inputs[name], self.config)
        
        # Verify all models were profiled successfully
        for name, result in results.items():
            assert result.mean_latency_ms > 0
            assert result.device == self.device
            print(f"{name} model: {result.mean_latency_ms:.2f}±{result.std_latency_ms:.2f} ms")
        
        # Generally expect more complex models to be slower (though not guaranteed for toy models)
        assert all(r.mean_latency_ms > 0 for r in results.values())
    
    def test_batch_size_scaling(self):
        """Test latency scaling with different batch sizes."""
        model = ToyModel(input_size=64, hidden_size=32, output_size=16)
        model = model.to(self.device)
        model.eval()
        
        batch_sizes = [1, 2, 4, 8]
        results = {}
        
        for batch_size in batch_sizes:
            inputs = torch.randn(batch_size, 64, device=self.device)
            results[batch_size] = benchmark_model_latency(model, inputs, self.config)
        
        # Verify all batch sizes were profiled
        for batch_size, result in results.items():
            assert result.mean_latency_ms > 0
            print(f"Batch size {batch_size}: {result.mean_latency_ms:.2f} ms")
        
        # Generally expect latency to increase with batch size, but allow some variance
        latencies = [results[bs].mean_latency_ms for bs in batch_sizes]
        assert all(lat > 0 for lat in latencies)
    
    def test_improvement_analysis(self):
        """Test latency improvement analysis between models."""
        # Create baseline and "optimized" models
        baseline_model = ToyModel(input_size=64, hidden_size=64, output_size=16)
        optimized_model = ToyModel(input_size=64, hidden_size=32, output_size=16)  # Smaller
        
        baseline_model = baseline_model.to(self.device)
        optimized_model = optimized_model.to(self.device)
        
        baseline_model.eval()
        optimized_model.eval()
        
        # Profile both models
        inputs = torch.randn(4, 64, device=self.device)
        
        baseline_results = benchmark_model_latency(baseline_model, inputs, self.config)
        optimized_results = benchmark_model_latency(optimized_model, inputs, self.config)
        
        # Analyze improvements
        analysis = analyze_latency_improvements(baseline_results, optimized_results)
        
        # Verify analysis results
        assert 'improvement_ms' in analysis
        assert 'improvement_pct' in analysis
        assert 'speedup_factor' in analysis
        assert 'statistically_significant' in analysis
        
        assert analysis['baseline_mean_ms'] == baseline_results.mean_latency_ms
        assert analysis['optimized_mean_ms'] == optimized_results.mean_latency_ms
        
        print(f"Improvement analysis: {analysis['improvement_pct']:.1f}% improvement")
        print(f"Speedup factor: {analysis['speedup_factor']:.2f}x")
    
    def test_profiling_with_different_data_types(self):
        """Test profiling with different tensor data types."""
        model = ToyModel(input_size=32, hidden_size=16, output_size=8)
        model = model.to(self.device)
        model.eval()
        
        # Test different data types
        data_types = [torch.float32, torch.float16] if self.device == 'cuda' else [torch.float32]
        
        for dtype in data_types:
            # Convert model to the appropriate dtype
            model_typed = model.to(dtype)
            inputs = torch.randn(2, 32, device=self.device, dtype=dtype)
            
            results = benchmark_model_latency(model_typed, inputs, self.config)
            
            assert results.mean_latency_ms > 0
            print(f"Data type {dtype}: {results.mean_latency_ms:.2f} ms")


class TestHardwareProfilingIntegration:
    """Integration tests for hardware profiling (mocked)."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = NCUConfig(
            metrics=['lts__t_sector_hit_rate.pct', 'dram__bytes_read.sum'],
            timeout_seconds=60
        )
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    @patch.object(NsightComputeProfiler, '_check_availability')
    @patch.object(NsightComputeProfiler, '_run_ncu_profiling')
    def test_hardware_profiling_integration(self, mock_run_ncu, mock_check_availability):
        """Test hardware profiling integration (mocked)."""
        # Mock successful profiling
        mock_check_availability.return_value = True
        
        from src.profiler.ncu import NCUResults
        mock_results = NCUResults(
            l2_cache_hit_rate=87.5,
            dram_read_bytes=1000000,
            dram_write_bytes=500000,
            kernels_profiled=3,
            profiling_overhead_ms=2000.0
        )
        mock_run_ncu.return_value = mock_results
        
        # Create model
        model = ToyModel(input_size=64, hidden_size=32, output_size=16)
        if self.device == 'cuda':
            model = model.to(self.device)
        model.eval()
        
        inputs = torch.randn(4, 64)
        if self.device == 'cuda':
            inputs = inputs.to(self.device)
        
        # Test hardware profiling
        results = collect_hardware_metrics(model, inputs, self.config)
        
        # Verify results
        assert results.l2_cache_hit_rate == 87.5
        assert results.dram_read_bytes == 1000000
        assert results.kernels_profiled == 3
        
        print(f"Hardware profiling: L2 hit rate {results.l2_cache_hit_rate}%")
    
    def test_hardware_profiling_unavailable(self):
        """Test behavior when hardware profiling is unavailable."""
        # Create profiler that will detect NCU as unavailable
        profiler = NsightComputeProfiler(self.config)
        
        model = ToyModel(input_size=32, hidden_size=16, output_size=8)
        inputs = torch.randn(2, 32)
        
        # Should return error result when unavailable
        results = profiler.profile_model(model, inputs)
        
        if not profiler.available:
            assert results.error_message is not None
            print(f"Hardware profiling unavailable: {results.error_message}")


class TestSystemCalibrationIntegration:
    """Integration tests for system calibration."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = CalibrationConfig(
            baseline_tolerance_pct=20.0,  # More lenient for tests
            min_runs=2,
            max_runs=3,
            warmup_runs=1,
            save_baseline=False  # Don't save during tests
        )
    
    def test_system_calibration_creation(self):
        """Test creating a new system baseline."""
        calibrator = SystemCalibrator(self.config)
        
        # Run calibration (should create new baseline)
        results = calibrator.calibrate_system(force_recalibrate=True)
        
        # Verify calibration succeeded
        assert results.baseline_latency_ms is not None
        assert results.measured_latency_ms is not None
        assert results.is_calibrated is True
        assert results.deviation_pct == 0.0  # Perfect match for new baseline
        
        print(f"Calibration baseline: {results.baseline_latency_ms:.2f} ms")
    
    def test_system_calibration_validation(self):
        """Test validating against existing baseline."""
        calibrator = SystemCalibrator(self.config)
        
        # Create initial baseline
        initial_results = calibrator.calibrate_system(force_recalibrate=True)
        assert initial_results.is_calibrated
        
        # Validate against the baseline
        validation_results = calibrator.calibrate_system(force_recalibrate=False)
        
        # Should validate successfully (same system)
        assert validation_results.is_calibrated
        assert validation_results.baseline_latency_ms is not None
        assert validation_results.measured_latency_ms is not None
        assert validation_results.deviation_pct is not None
        
        print(f"Calibration validation: {validation_results.deviation_pct:.1f}% deviation")
    
    def test_calibration_with_mock_high_deviation(self):
        """Test calibration failure with high deviation."""
        # Create calibrator with strict tolerance
        strict_config = CalibrationConfig(
            baseline_tolerance_pct=1.0,  # Very strict
            min_runs=2,
            max_runs=3,
            save_baseline=False
        )
        calibrator = SystemCalibrator(strict_config)
        
        # Create baseline
        calibrator.calibrate_system(force_recalibrate=True)
        
        # Mock the calibration model to return different performance
        with patch.object(calibrator, '_create_calibration_model') as mock_model:
            # Create a model that will perform differently
            slow_model = nn.Sequential(
                nn.Linear(512, 1024),  # Larger model
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, 64),
                nn.ReLU(),
                nn.Linear(64, 32)
            )
            mock_model.return_value = slow_model
            
            # This should fail calibration due to performance difference
            results = calibrator.calibrate_system(force_recalibrate=False)
            
            # Verify failure or high deviation
            if not results.is_calibrated:
                assert results.deviation_pct > strict_config.baseline_tolerance_pct
                print(f"Calibration failed: {results.deviation_pct:.1f}% deviation")
            else:
                print(f"Calibration passed: {results.deviation_pct:.1f}% deviation")


class TestEndToEndProfilingIntegration:
    """End-to-end integration tests combining all profiling components."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create models for comparison
        self.models = {
            'baseline': ToyModel(input_size=64, hidden_size=64, output_size=16),
            'optimized': ToyModel(input_size=64, hidden_size=32, output_size=16)
        }
        
        for name, model in self.models.items():
            self.models[name] = model.to(self.device)
            self.models[name].eval()
        
        self.inputs = torch.randn(4, 64, device=self.device)
    
    def test_complete_profiling_workflow(self):
        """Test complete profiling workflow with calibration, latency, and hardware profiling."""
        # Step 1: System calibration
        calibration_config = CalibrationConfig(
            baseline_tolerance_pct=25.0,
            min_runs=2,
            max_runs=3,
            save_baseline=False
        )
        
        calibration_results = validate_system_performance(calibration_config, force_recalibrate=True)
        assert calibration_results.is_calibrated
        print(f"✓ System calibration passed")
        
        # Step 2: Latency profiling
        latency_config = LatencyConfig(
            warmup_runs=2,
            measurement_runs=3,
            use_cuda_events=self.device == 'cuda'
        )
        
        latency_results = {}
        for name, model in self.models.items():
            latency_results[name] = benchmark_model_latency(model, self.inputs, latency_config)
            print(f"✓ {name} latency: {latency_results[name].mean_latency_ms:.2f} ms")
        
        # Step 3: Hardware profiling (mocked)
        with patch.object(NsightComputeProfiler, '_check_availability', return_value=True):
            with patch.object(NsightComputeProfiler, '_run_ncu_profiling') as mock_ncu:
                from src.profiler.ncu import NCUResults
                mock_ncu.return_value = NCUResults(
                    l2_cache_hit_rate=85.0,
                    kernels_profiled=2
                )
                
                ncu_config = NCUConfig(
                    metrics=['lts__t_sector_hit_rate.pct'],
                    timeout_seconds=30
                )
                
                hardware_results = {}
                for name, model in self.models.items():
                    hardware_results[name] = collect_hardware_metrics(model, self.inputs, ncu_config)
                    if hardware_results[name].error_message is None:
                        print(f"✓ {name} hardware profiling completed")
        
        # Step 4: Analysis
        improvement_analysis = analyze_latency_improvements(
            latency_results['baseline'],
            latency_results['optimized']
        )
        
        print(f"✓ Performance analysis: {improvement_analysis['improvement_pct']:.1f}% improvement")
        
        # Verify all steps completed successfully
        assert calibration_results.is_calibrated
        assert all(r.mean_latency_ms > 0 for r in latency_results.values())
        assert 'improvement_pct' in improvement_analysis
    
    def test_profiling_with_temporary_files(self):
        """Test profiling with temporary file management."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create latency profiler
            latency_config = LatencyConfig(
                warmup_runs=1,
                measurement_runs=2
            )
            
            # Profile models and save results
            for name, model in self.models.items():
                results = benchmark_model_latency(model, self.inputs, latency_config)
                
                # Save results to temporary file
                results_file = temp_path / f"{name}_latency.json"
                import json
                with open(results_file, 'w') as f:
                    json.dump(results.to_dict(), f, indent=2)
                
                assert results_file.exists()
                print(f"✓ Saved {name} results to {results_file}")
            
            # Verify files were created
            assert len(list(temp_path.glob("*_latency.json"))) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])