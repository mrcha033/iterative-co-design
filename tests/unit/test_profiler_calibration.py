"""
Unit tests for system calibration module.

These tests verify calibration functionality and baseline validation.
"""
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest
import torch
import torch.nn as nn

from src.profiler.calibration import (
    CalibrationConfig, CalibrationResults, SystemCalibrator,
    validate_system_performance, abort_if_system_invalid
)


class TestCalibrationConfig:
    """Test CalibrationConfig validation."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = CalibrationConfig()
        
        assert config.baseline_tolerance_pct == 15.0
        assert config.min_runs == 5
        assert config.max_runs == 10
        assert config.warmup_runs == 10
        assert config.target_cv_pct == 5.0
        assert config.save_baseline is True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = CalibrationConfig(
            baseline_tolerance_pct=20.0,
            min_runs=3,
            max_runs=8,
            warmup_runs=5,
            save_baseline=False
        )
        
        assert config.baseline_tolerance_pct == 20.0
        assert config.min_runs == 3
        assert config.max_runs == 8
        assert config.warmup_runs == 5
        assert config.save_baseline is False
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        valid_config = CalibrationConfig(baseline_tolerance_pct=10.0)
        valid_config.validate()  # Should not raise
        
        # Invalid configs
        with pytest.raises(ValueError, match="baseline_tolerance_pct must be positive"):
            CalibrationConfig(baseline_tolerance_pct=-1.0).validate()
        
        with pytest.raises(ValueError, match="min_runs must be positive"):
            CalibrationConfig(min_runs=0).validate()
        
        with pytest.raises(ValueError, match="max_runs must be >= min_runs"):
            CalibrationConfig(min_runs=10, max_runs=5).validate()
        
        with pytest.raises(ValueError, match="target_cv_pct must be positive"):
            CalibrationConfig(target_cv_pct=-1.0).validate()


class TestCalibrationResults:
    """Test CalibrationResults functionality."""
    
    def test_results_creation(self):
        """Test creating CalibrationResults."""
        results = CalibrationResults(
            is_calibrated=True,
            baseline_latency_ms=10.5,
            measured_latency_ms=11.0,
            deviation_pct=4.8,
            cv_pct=3.2,
            runs_completed=5
        )
        
        assert results.is_calibrated is True
        assert results.baseline_latency_ms == 10.5
        assert results.measured_latency_ms == 11.0
        assert results.deviation_pct == 4.8
        assert results.warnings == []  # Auto-initialized
    
    def test_to_dict(self):
        """Test converting results to dictionary."""
        results = CalibrationResults(
            is_calibrated=False,
            error_message="System overloaded",
            warnings=["High variability"]
        )
        
        result_dict = results.to_dict()
        
        assert result_dict['is_calibrated'] is False
        assert result_dict['error_message'] == "System overloaded"
        assert result_dict['warnings'] == ["High variability"]
        assert 'baseline_latency_ms' in result_dict


class TestSystemCalibrator:
    """Test SystemCalibrator functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = CalibrationConfig(
            baseline_tolerance_pct=20.0,
            min_runs=2,
            max_runs=3,
            warmup_runs=1,
            save_baseline=False  # Don't save during tests
        )
    
    def test_calibrator_initialization(self):
        """Test calibrator initialization."""
        calibrator = SystemCalibrator(self.config)
        
        assert calibrator.config.baseline_tolerance_pct == 20.0
        assert calibrator.device in ['cuda', 'cpu']
        assert calibrator.baseline_data is None
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_device_detection_cpu(self, mock_cuda):
        """Test device detection when CUDA is unavailable."""
        calibrator = SystemCalibrator(self.config)
        assert calibrator.device == 'cpu'
    
    @patch('torch.cuda.is_available', return_value=True)
    def test_device_detection_gpu(self, mock_cuda):
        """Test device detection when CUDA is available."""
        calibrator = SystemCalibrator(self.config)
        assert calibrator.device == 'cuda'
    
    def test_calibration_model_creation(self):
        """Test creation of calibration model."""
        calibrator = SystemCalibrator(self.config)
        model = calibrator._create_calibration_model()
        
        assert isinstance(model, nn.Module)
        assert hasattr(model, 'linear1')
        assert hasattr(model, 'linear2')
        assert hasattr(model, 'linear3')
        assert hasattr(model, 'linear4')
        
        # Test forward pass
        dummy_input = torch.randn(2, 512)
        output = model(dummy_input)
        assert output.shape == (2, 32)
    
    def test_system_info_collection(self):
        """Test system information collection."""
        calibrator = SystemCalibrator(self.config)
        system_info = calibrator._get_system_info()
        
        assert 'platform' in system_info
        assert 'python_version' in system_info
        assert 'torch_version' in system_info
        assert 'cpu_count' in system_info
        assert 'memory_gb' in system_info
        assert 'hostname' in system_info
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_gpu_info_unavailable(self, mock_cuda):
        """Test GPU info when CUDA is unavailable."""
        calibrator = SystemCalibrator(self.config)
        gpu_info = calibrator._get_gpu_info()
        
        assert gpu_info['available'] is False
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=1)
    @patch('torch.cuda.current_device', return_value=0)
    @patch('torch.cuda.get_device_name', return_value="Tesla V100")
    @patch('torch.cuda.get_device_capability', return_value=(7, 0))
    def test_gpu_info_available(self, mock_capability, mock_name, mock_device, mock_count, mock_cuda):
        """Test GPU info when CUDA is available."""
        calibrator = SystemCalibrator(self.config)
        gpu_info = calibrator._get_gpu_info()
        
        assert gpu_info['available'] is True
        assert gpu_info['device_count'] == 1
        assert gpu_info['current_device'] == 0
        assert gpu_info['device_name'] == "Tesla V100"
        assert gpu_info['device_capability'] == (7, 0)
    
    def test_baseline_load_nonexistent(self):
        """Test baseline loading when file doesn't exist."""
        config = CalibrationConfig(baseline_file="nonexistent.json", save_baseline=False)
        calibrator = SystemCalibrator(config)
        
        assert calibrator.baseline_data is None
    
    def test_baseline_save_and_load(self):
        """Test baseline saving and loading."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            baseline_file = f.name
        
        try:
            config = CalibrationConfig(baseline_file=baseline_file, save_baseline=True)
            calibrator = SystemCalibrator(config)
            
            # Save baseline
            baseline_data = {
                'latency_ms': 15.5,
                'timestamp': 1234567890,
                'system_info': {'platform': 'test'}
            }
            calibrator._save_baseline(baseline_data)
            
            # Create new calibrator to test loading
            new_calibrator = SystemCalibrator(config)
            assert new_calibrator.baseline_data is not None
            assert new_calibrator.baseline_data['latency_ms'] == 15.5
            
        finally:
            Path(baseline_file).unlink(missing_ok=True)
    
    @patch.object(SystemCalibrator, '_run_calibration_benchmark')
    @patch.object(SystemCalibrator, '_get_system_info')
    @patch.object(SystemCalibrator, '_get_gpu_info')
    def test_calibrate_system_new_baseline(self, mock_gpu_info, mock_system_info, mock_benchmark):
        """Test system calibration with new baseline creation."""
        # Mock returns
        mock_system_info.return_value = {'platform': 'test'}
        mock_gpu_info.return_value = {'available': False}
        
        from src.profiler.latency import LatencyResults
        mock_benchmark.return_value = LatencyResults(
            mean_latency_ms=15.0,
            std_latency_ms=0.5,
            min_latency_ms=14.5,
            max_latency_ms=15.5,
            median_latency_ms=15.0,
            p95_latency_ms=15.4,
            p99_latency_ms=15.5,
            cv_percent=3.3,
            raw_times=[14.5, 15.0, 15.5],
            warmup_times=[16.0, 15.5],
            outliers_removed=0,
            measurement_runs=3,
            device='cpu',
            timing_method='perf_counter'
        )
        
        calibrator = SystemCalibrator(self.config)
        results = calibrator.calibrate_system(force_recalibrate=True)
        
        assert results.is_calibrated is True
        assert results.baseline_latency_ms == 15.0
        assert results.measured_latency_ms == 15.0
        assert results.deviation_pct == 0.0
        assert results.cv_pct == 3.3
    
    @patch.object(SystemCalibrator, '_run_calibration_benchmark')
    @patch.object(SystemCalibrator, '_get_system_info')
    @patch.object(SystemCalibrator, '_get_gpu_info')
    def test_calibrate_system_validate_existing(self, mock_gpu_info, mock_system_info, mock_benchmark):
        """Test system calibration validation against existing baseline."""
        # Set up existing baseline
        calibrator = SystemCalibrator(self.config)
        calibrator.baseline_data = {
            'latency_ms': 15.0,
            'system_info': {'platform': 'test'},
            'gpu_info': {'available': False}
        }
        
        # Mock returns
        mock_system_info.return_value = {'platform': 'test'}
        mock_gpu_info.return_value = {'available': False}
        
        from src.profiler.latency import LatencyResults
        mock_benchmark.return_value = LatencyResults(
            mean_latency_ms=16.0,  # 6.7% deviation
            std_latency_ms=0.5,
            min_latency_ms=15.5,
            max_latency_ms=16.5,
            median_latency_ms=16.0,
            p95_latency_ms=16.4,
            p99_latency_ms=16.5,
            cv_percent=3.1,
            raw_times=[15.5, 16.0, 16.5],
            warmup_times=[17.0, 16.5],
            outliers_removed=0,
            measurement_runs=3,
            device='cpu',
            timing_method='perf_counter'
        )
        
        results = calibrator.calibrate_system(force_recalibrate=False)
        
        assert results.is_calibrated is True  # Within 20% tolerance
        assert results.baseline_latency_ms == 15.0
        assert results.measured_latency_ms == 16.0
        assert abs(results.deviation_pct - 6.67) < 0.1
    
    @patch.object(SystemCalibrator, '_run_calibration_benchmark')
    def test_calibrate_system_failure(self, mock_benchmark):
        """Test system calibration failure handling."""
        mock_benchmark.side_effect = RuntimeError("Benchmark failed")
        
        calibrator = SystemCalibrator(self.config)
        results = calibrator.calibrate_system()
        
        assert results.is_calibrated is False
        assert "Benchmark failed" in results.error_message
    
    def test_system_compatibility_check(self):
        """Test system compatibility checking."""
        calibrator = SystemCalibrator(self.config)
        calibrator.baseline_data = {
            'system_info': {
                'platform': 'Linux-5.4.0',
                'torch_version': '1.12.0',
                'cuda_version': '11.6'
            },
            'gpu_info': {
                'device_name': 'Tesla V100'
            }
        }
        
        # Compatible system
        current_system = {
            'platform': 'Linux-5.4.0',
            'torch_version': '1.12.0', 
            'cuda_version': '11.6'
        }
        current_gpu = {'device_name': 'Tesla V100'}
        
        warnings = calibrator._check_system_compatibility(current_system, current_gpu)
        assert len(warnings) == 0
        
        # Incompatible system
        incompatible_system = {
            'platform': 'Windows-10',
            'torch_version': '1.13.0',
            'cuda_version': '11.7'
        }
        incompatible_gpu = {'device_name': 'Tesla A100'}
        
        warnings = calibrator._check_system_compatibility(incompatible_system, incompatible_gpu)
        assert len(warnings) > 0
        assert any('platform' in w.lower() for w in warnings)
        assert any('gpu model' in w.lower() for w in warnings)
    
    def test_reset_baseline(self):
        """Test baseline reset functionality."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            baseline_file = f.name
            json.dump({'test': 'data'}, f)
        
        try:
            config = CalibrationConfig(baseline_file=baseline_file)
            calibrator = SystemCalibrator(config)
            
            # Verify baseline was loaded
            assert calibrator.baseline_data is not None
            
            # Reset baseline
            calibrator.reset_baseline()
            assert calibrator.baseline_data is None
            assert not Path(baseline_file).exists()
            
        except FileNotFoundError:
            # File was successfully deleted
            pass


class TestUtilityFunctions:
    """Test utility functions."""
    
    @patch.object(SystemCalibrator, 'calibrate_system')
    def test_validate_system_performance(self, mock_calibrate):
        """Test validate_system_performance utility function."""
        mock_calibrate.return_value = CalibrationResults(
            is_calibrated=True,
            baseline_latency_ms=15.0,
            measured_latency_ms=15.5,
            deviation_pct=3.3
        )
        
        config = CalibrationConfig()
        results = validate_system_performance(config)
        
        assert results.is_calibrated is True
        mock_calibrate.assert_called_once_with(False)
    
    @patch.object(SystemCalibrator, 'calibrate_system')
    def test_abort_if_system_invalid_success(self, mock_calibrate):
        """Test abort_if_system_invalid with valid system."""
        mock_calibrate.return_value = CalibrationResults(
            is_calibrated=True,
            warnings=["Minor warning"]
        )
        
        # Should not raise
        abort_if_system_invalid()
    
    @patch.object(SystemCalibrator, 'calibrate_system')
    def test_abort_if_system_invalid_failure(self, mock_calibrate):
        """Test abort_if_system_invalid with invalid system."""
        mock_calibrate.return_value = CalibrationResults(
            is_calibrated=False,
            error_message="System overloaded",
            deviation_pct=25.0
        )
        
        with pytest.raises(RuntimeError, match="System calibration failed"):
            abort_if_system_invalid()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])