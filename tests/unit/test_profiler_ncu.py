"""
Unit tests for NVIDIA Nsight Compute profiler integration.

These tests verify NCU integration and metric parsing functionality.
"""
import csv
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest
import torch
import torch.nn as nn

from src.profiler.ncu import (
    NCUConfig, NCUResults, NsightComputeProfiler,
    collect_hardware_metrics, compare_hardware_metrics
)


class TestNCUConfig:
    """Test NCUConfig validation."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = NCUConfig()
        
        assert 'lts__t_sector_hit_rate.pct' in config.metrics
        assert 'dram__bytes_read.sum' in config.metrics
        assert config.target_processes == 'all'
        assert config.timeout_seconds == 300
        assert config.output_format == 'csv'
    
    def test_custom_config(self):
        """Test custom configuration values."""
        custom_metrics = ['lts__t_sector_hit_rate.pct', 'gpu__time_duration.sum']
        config = NCUConfig(
            metrics=custom_metrics,
            timeout_seconds=600,
            output_format='json'
        )
        
        assert config.metrics == custom_metrics
        assert config.timeout_seconds == 600
        assert config.output_format == 'json'
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        valid_config = NCUConfig(metrics=['test_metric'])
        valid_config.validate()
        
        # Invalid configs
        with pytest.raises(ValueError):
            invalid_config = NCUConfig(metrics=[])
            invalid_config.validate()
        
        with pytest.raises(ValueError):
            invalid_config = NCUConfig(timeout_seconds=-1)
            invalid_config.validate()
        
        with pytest.raises(ValueError):
            invalid_config = NCUConfig(output_format='invalid')
            invalid_config.validate()


class TestNCUResults:
    """Test NCUResults functionality."""
    
    def test_results_creation(self):
        """Test creating NCUResults."""
        results = NCUResults(
            l2_cache_hit_rate=89.5,
            dram_read_bytes=1000000,
            dram_write_bytes=500000,
            kernels_profiled=5,
            profiling_overhead_ms=1500.0
        )
        
        assert results.l2_cache_hit_rate == 89.5
        assert results.dram_read_bytes == 1000000
        assert results.kernels_profiled == 5
    
    def test_to_dict(self):
        """Test converting results to dictionary."""
        results = NCUResults(
            l2_cache_hit_rate=89.5,
            dram_bandwidth_gb_s=500.0,
            error_message=None
        )
        
        result_dict = results.to_dict()
        
        assert result_dict['l2_cache_hit_rate'] == 89.5
        assert result_dict['dram_bandwidth_gb_s'] == 500.0
        assert result_dict['error_message'] is None
        assert 'raw_metrics' in result_dict


class TestNsightComputeProfiler:
    """Test NsightComputeProfiler functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = NCUConfig(
            metrics=['lts__t_sector_hit_rate.pct', 'dram__bytes_read.sum'],
            timeout_seconds=60
        )
    
    def test_profiler_initialization(self):
        """Test profiler initialization."""
        profiler = NsightComputeProfiler(self.config)
        
        assert profiler.config.metrics == self.config.metrics
        assert profiler.config.timeout_seconds == 60
    
    @patch('subprocess.run')
    def test_availability_check_success(self, mock_run):
        """Test successful availability check."""
        mock_run.return_value = Mock(returncode=0, stdout="ncu version 2023.1.0")
        
        profiler = NsightComputeProfiler(self.config)
        assert profiler.available is True
    
    @patch('subprocess.run')
    def test_availability_check_failure(self, mock_run):
        """Test failed availability check."""
        mock_run.side_effect = FileNotFoundError("ncu not found")
        
        profiler = NsightComputeProfiler(self.config)
        assert profiler.available is False
    
    def test_create_profile_script(self):
        """Test profile script creation."""
        # Create mock model and inputs
        model = nn.Linear(4, 2)
        inputs = torch.randn(2, 4)
        
        profiler = NsightComputeProfiler(self.config)
        script = profiler._create_profile_script(model, inputs)
        
        assert 'import torch' in script
        assert 'def profile_target' in script
        assert 'torch.cuda.synchronize' in script
    
    def test_parse_csv_output_empty_file(self):
        """Test parsing empty CSV file."""
        profiler = NsightComputeProfiler(self.config)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            csv_path = Path(f.name)
        
        try:
            results = profiler._parse_csv_output(csv_path)
            assert results.kernels_profiled == 0
            assert results.error_message is not None
        finally:
            csv_path.unlink()
    
    def test_parse_csv_output_with_data(self):
        """Test parsing CSV file with sample data."""
        profiler = NsightComputeProfiler(self.config)
        
        # Create sample CSV data
        csv_data = [
            ['ID', 'Kernel Name', 'lts__t_sector_hit_rate.pct', 'dram__bytes_read.sum'],
            ['1', 'kernel1', '85.5', '1000000'],
            ['2', 'kernel2', '90.2', '1500000'],
            ['3', 'kernel3', '88.1', '1200000']
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            writer = csv.writer(f)
            writer.writerows(csv_data)
            csv_path = Path(f.name)
        
        try:
            results = profiler._parse_csv_output(csv_path)
            
            assert results.kernels_profiled == 3
            assert results.error_message is None
            assert len(results.raw_metrics) > 0
            
            # Check if metrics were extracted
            if 'lts__t_sector_hit_rate.pct' in results.raw_metrics:
                metric_data = results.raw_metrics['lts__t_sector_hit_rate.pct']
                assert 'mean' in metric_data
                assert 'values' in metric_data
                assert len(metric_data['values']) == 3
                
        finally:
            csv_path.unlink()
    
    def test_extract_metrics_from_raw(self):
        """Test metric extraction from raw data."""
        profiler = NsightComputeProfiler(self.config)
        results = NCUResults()
        
        raw_metrics = {
            'lts__t_sector_hit_rate.pct': {'mean': 87.5, 'sum': 262.5, 'values': [85.5, 90.2, 88.1]},
            'dram__bytes_read.sum': {'mean': 1233333, 'sum': 3700000, 'values': [1000000, 1500000, 1200000]},
            'dram__bytes_write.sum': {'mean': 800000, 'sum': 2400000, 'values': [700000, 900000, 800000]}
        }
        
        profiler._extract_metrics_from_raw(results, raw_metrics)
        
        assert results.l2_cache_hit_rate == 87.5
        assert results.dram_read_bytes == 3700000  # Sum value
        assert results.dram_write_bytes == 2400000  # Sum value
    
    @patch('subprocess.run')
    def test_run_ncu_profiling_success(self, mock_run):
        """Test successful NCU profiling execution."""
        # Mock successful subprocess execution
        mock_run.return_value = Mock(
            returncode=0,
            stdout="Profiling completed",
            stderr=""
        )
        
        # Create temporary script and output files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as script_f:
            script_path = Path(script_f.name)
            script_f.write("print('test script')")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as output_f:
            output_path = Path(output_f.name)
            # Write minimal CSV data
            output_f.write('ID,Kernel Name,lts__t_sector_hit_rate.pct\n')
            output_f.write('1,test_kernel,85.5\n')
        
        try:
            profiler = NsightComputeProfiler(self.config)
            # Mock the availability check
            profiler.available = True
            
            results = profiler._run_ncu_profiling(script_path, output_path)
            
            assert results.error_message is None
            assert results.profiling_overhead_ms > 0
            
        finally:
            script_path.unlink()
            output_path.unlink()
    
    @patch('subprocess.run')
    def test_run_ncu_profiling_failure(self, mock_run):
        """Test failed NCU profiling execution."""
        # Mock failed subprocess execution
        mock_run.return_value = Mock(
            returncode=1,
            stdout="",
            stderr="ncu: command not found"
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as script_f:
            script_path = Path(script_f.name)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as output_f:
            output_path = Path(output_f.name)
        
        try:
            profiler = NsightComputeProfiler(self.config)
            profiler.available = True
            
            results = profiler._run_ncu_profiling(script_path, output_path)
            
            assert results.error_message is not None
            assert "ncu failed" in results.error_message
            
        finally:
            script_path.unlink()
            output_path.unlink()
    
    @patch('subprocess.run')
    def test_run_ncu_profiling_timeout(self, mock_run):
        """Test NCU profiling timeout."""
        from subprocess import TimeoutExpired
        
        mock_run.side_effect = TimeoutExpired('ncu', 60)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as script_f:
            script_path = Path(script_f.name)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as output_f:
            output_path = Path(output_f.name)
        
        try:
            profiler = NsightComputeProfiler(self.config)
            profiler.available = True
            
            results = profiler._run_ncu_profiling(script_path, output_path)
            
            assert results.error_message is not None
            assert "timed out" in results.error_message
            
        finally:
            script_path.unlink()
            output_path.unlink()
    
    def test_profile_model_unavailable(self):
        """Test profiling when NCU is unavailable."""
        profiler = NsightComputeProfiler(self.config)
        profiler.available = False
        
        model = nn.Linear(4, 2)
        inputs = torch.randn(2, 4)
        
        results = profiler.profile_model(model, inputs)
        
        assert results.error_message == "ncu not available"


class TestUtilityFunctions:
    """Test utility functions."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = NCUConfig(
            metrics=['lts__t_sector_hit_rate.pct'],
            timeout_seconds=60
        )
    
    @patch.object(NsightComputeProfiler, 'profile_model')
    def test_collect_hardware_metrics(self, mock_profile):
        """Test hardware metrics collection."""
        # Mock successful profiling
        mock_results = NCUResults(l2_cache_hit_rate=87.5, kernels_profiled=3)
        mock_profile.return_value = mock_results
        
        model = nn.Linear(4, 2)
        inputs = torch.randn(2, 4)
        
        results = collect_hardware_metrics(model, inputs, self.config)
        
        assert results.l2_cache_hit_rate == 87.5
        assert results.kernels_profiled == 3
        mock_profile.assert_called_once()
    
    @patch.object(NsightComputeProfiler, 'profile_model')
    def test_compare_hardware_metrics(self, mock_profile):
        """Test comparing hardware metrics across models."""
        # Mock different results for different models
        def side_effect(model, inputs, output_file=None):
            if hasattr(model, 'in_features') and model.in_features == 4:
                return NCUResults(l2_cache_hit_rate=85.0, kernels_profiled=2)
            else:
                return NCUResults(l2_cache_hit_rate=90.0, kernels_profiled=4)
        
        mock_profile.side_effect = side_effect
        
        models = {
            'small': nn.Linear(4, 2),
            'large': nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2))
        }
        inputs = torch.randn(2, 8)  # Will be adjusted by profiler
        
        results = compare_hardware_metrics(models, inputs, self.config)
        
        assert 'small' in results
        assert 'large' in results
        assert results['small'].l2_cache_hit_rate == 85.0
        assert results['large'].l2_cache_hit_rate == 90.0


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_missing_output_file(self):
        """Test handling of missing output file."""
        profiler = NsightComputeProfiler(NCUConfig())
        
        # Try to parse non-existent file
        non_existent_path = Path('/tmp/non_existent_file.csv')
        results = profiler._parse_ncu_output(non_existent_path)
        
        assert results.error_message is not None
        assert "not found" in results.error_message
    
    def test_malformed_csv_file(self):
        """Test handling of malformed CSV file."""
        profiler = NsightComputeProfiler(NCUConfig())
        
        # Create malformed CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("This is not valid CSV data\n")
            f.write("Missing proper headers and structure\n")
            csv_path = Path(f.name)
        
        try:
            results = profiler._parse_csv_output(csv_path)
            # Should handle gracefully, either with empty results or error message
            assert results.kernels_profiled == 0 or results.error_message is not None
            
        finally:
            csv_path.unlink()
    
    def test_json_parsing_not_implemented(self):
        """Test JSON parsing placeholder."""
        config = NCUConfig(output_format='json')
        profiler = NsightComputeProfiler(config)
        
        # Create dummy JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"test": "data"}, f)
            json_path = Path(f.name)
        
        try:
            results = profiler._parse_json_output(json_path)
            # Should indicate JSON parsing not implemented
            assert "not implemented" in results.error_message
            
        finally:
            json_path.unlink()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])