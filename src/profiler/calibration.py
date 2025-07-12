"""
System calibration and baseline performance validation.

This module implements calibration checks to ensure system performance
is within expected baselines before running experiments, as required by the PRD.
"""
import json
import logging
import time
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import numpy as np

from .latency import LatencyProfiler, LatencyConfig, LatencyResults


logger = logging.getLogger(__name__)


@dataclass
class CalibrationConfig:
    """Configuration for system calibration."""
    baseline_tolerance_pct: float = 15.0  # Allow 15% deviation from baseline
    min_runs: int = 5
    max_runs: int = 10
    warmup_runs: int = 10
    target_cv_pct: float = 5.0  # Target coefficient of variation
    calibration_timeout_seconds: int = 120
    save_baseline: bool = True
    baseline_file: Optional[str] = "system_baseline.json"
    
    def validate(self) -> None:
        """Validate configuration."""
        if self.baseline_tolerance_pct <= 0:
            raise ValueError("baseline_tolerance_pct must be positive")
        if self.min_runs <= 0:
            raise ValueError("min_runs must be positive") 
        if self.max_runs < self.min_runs:
            raise ValueError("max_runs must be >= min_runs")
        if self.target_cv_pct <= 0:
            raise ValueError("target_cv_pct must be positive")


@dataclass 
class CalibrationResults:
    """Results from system calibration."""
    is_calibrated: bool
    baseline_latency_ms: Optional[float] = None
    measured_latency_ms: Optional[float] = None
    deviation_pct: Optional[float] = None
    cv_pct: Optional[float] = None
    runs_completed: int = 0
    gpu_info: Dict[str, Any] = None
    system_info: Dict[str, Any] = None
    error_message: Optional[str] = None
    warnings: list = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'is_calibrated': self.is_calibrated,
            'baseline_latency_ms': self.baseline_latency_ms,
            'measured_latency_ms': self.measured_latency_ms,
            'deviation_pct': self.deviation_pct,
            'cv_pct': self.cv_pct,
            'runs_completed': self.runs_completed,
            'gpu_info': self.gpu_info,
            'system_info': self.system_info,
            'error_message': self.error_message,
            'warnings': self.warnings
        }


class SystemCalibrator:
    """
    System calibrator for baseline performance validation.
    
    This class implements the calibration methodology described in the PRD
    to ensure reproducible and reliable benchmarking results.
    """
    
    def __init__(self, config: Optional[CalibrationConfig] = None):
        """
        Initialize system calibrator.
        
        Args:
            config: Calibration configuration
        """
        self.config = config if config is not None else CalibrationConfig()
        self.config.validate()
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.baseline_data = None
        
        # Load existing baseline if available
        if self.config.baseline_file:
            self._load_baseline()
        
        logger.info(f"SystemCalibrator initialized for device: {self.device}")
    
    def _load_baseline(self) -> None:
        """Load existing baseline data."""
        if not self.config.baseline_file:
            return
        
        baseline_path = Path(self.config.baseline_file)
        if baseline_path.exists():
            try:
                with open(baseline_path, 'r') as f:
                    self.baseline_data = json.load(f)
                logger.info(f"Loaded baseline from {baseline_path}")
            except Exception as e:
                logger.warning(f"Failed to load baseline from {baseline_path}: {e}")
    
    def _save_baseline(self, baseline_data: Dict[str, Any]) -> None:
        """Save baseline data."""
        if not self.config.save_baseline or not self.config.baseline_file:
            return
        
        try:
            baseline_path = Path(self.config.baseline_file)
            with open(baseline_path, 'w') as f:
                json.dump(baseline_data, f, indent=2)
            logger.info(f"Saved baseline to {baseline_path}")
        except Exception as e:
            logger.error(f"Failed to save baseline: {e}")
    
    def calibrate_system(self, force_recalibrate: bool = False) -> CalibrationResults:
        """
        Calibrate system and validate performance baseline.
        
        Args:
            force_recalibrate: Force recalibration even if baseline exists
            
        Returns:
            CalibrationResults with calibration status and metrics
        """
        logger.info("Starting system calibration...")
        
        try:
            # Get system information
            system_info = self._get_system_info()
            gpu_info = self._get_gpu_info()
            
            # Create calibration model
            calibration_model = self._create_calibration_model()
            
            # Run calibration benchmark
            latency_results = self._run_calibration_benchmark(calibration_model)
            
            # Check against baseline
            calibration_results = self._validate_against_baseline(
                latency_results, system_info, gpu_info, force_recalibrate
            )
            
            return calibration_results
            
        except Exception as e:
            logger.error(f"System calibration failed: {e}")
            return CalibrationResults(
                is_calibrated=False,
                error_message=str(e)
            )
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for calibration."""
        import platform
        import psutil
        
        return {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'torch_version': torch.__version__,
            'cuda_version': torch.version.cuda,
            'cpu_count': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'hostname': platform.node()
        }
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information for calibration."""
        if not torch.cuda.is_available():
            return {'available': False}
        
        gpu_info = {
            'available': True,
            'device_count': torch.cuda.device_count(),
            'current_device': torch.cuda.current_device(),
        }
        
        try:
            device = torch.cuda.current_device()
            gpu_info.update({
                'device_name': torch.cuda.get_device_name(device),
                'device_capability': torch.cuda.get_device_capability(device),
                'memory_total_gb': torch.cuda.get_device_properties(device).total_memory / (1024**3),
                'memory_reserved_gb': torch.cuda.memory_reserved(device) / (1024**3),
                'memory_allocated_gb': torch.cuda.memory_allocated(device) / (1024**3)
            })
        except Exception as e:
            logger.warning(f"Could not get detailed GPU info: {e}")
        
        return gpu_info
    
    def _create_calibration_model(self) -> nn.Module:
        """Create a standardized calibration model."""
        class CalibrationModel(nn.Module):
            """Standardized model for system calibration."""
            
            def __init__(self):
                super().__init__()
                # Standard layers that should be representative
                self.linear1 = nn.Linear(512, 256)
                self.linear2 = nn.Linear(256, 128)
                self.linear3 = nn.Linear(128, 64)
                self.linear4 = nn.Linear(64, 32)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.1)
            
            def forward(self, x):
                x = self.relu(self.linear1(x))
                x = self.dropout(x)
                x = self.relu(self.linear2(x))
                x = self.dropout(x)
                x = self.relu(self.linear3(x))
                x = self.linear4(x)
                return x
        
        model = CalibrationModel()
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def _run_calibration_benchmark(self, model: nn.Module) -> LatencyResults:
        """Run calibration benchmark."""
        logger.info("Running calibration benchmark...")
        
        # Create standardized input
        batch_size = 4
        input_size = 512
        dummy_input = torch.randn(batch_size, input_size, device=self.device)
        
        # Configure latency profiler for calibration
        latency_config = LatencyConfig(
            warmup_runs=self.config.warmup_runs,
            measurement_runs=self.config.max_runs,
            use_cuda_events=self.device == 'cuda',
            cuda_sync=True,
            enable_autograd=False,
            memory_cleanup=True,
            statistical_validation=True,
            outlier_threshold=2.0
        )
        
        profiler = LatencyProfiler(latency_config)
        
        # Profile the model
        def model_forward():
            return model(dummy_input)
        
        start_time = time.time()
        results = profiler.profile_function(model_forward)
        calibration_time = time.time() - start_time
        
        logger.info(f"Calibration benchmark completed in {calibration_time:.1f}s")
        logger.info(f"Measured latency: {results.mean_latency_ms:.2f}±{results.std_latency_ms:.2f} ms")
        
        return results
    
    def _validate_against_baseline(
        self,
        latency_results: LatencyResults,
        system_info: Dict[str, Any],
        gpu_info: Dict[str, Any],
        force_recalibrate: bool
    ) -> CalibrationResults:
        """Validate measured performance against baseline."""
        warnings_list = []
        
        # Check if we have a baseline
        if self.baseline_data is None or force_recalibrate:
            # Create new baseline
            baseline_data = {
                'latency_ms': latency_results.mean_latency_ms,
                'std_latency_ms': latency_results.std_latency_ms,
                'cv_pct': latency_results.cv_percent,
                'system_info': system_info,
                'gpu_info': gpu_info,
                'timestamp': time.time(),
                'calibration_config': {
                    'warmup_runs': self.config.warmup_runs,
                    'measurement_runs': latency_results.measurement_runs,
                    'device': self.device
                }
            }
            
            self._save_baseline(baseline_data)
            self.baseline_data = baseline_data
            
            logger.info(f"Created new baseline: {latency_results.mean_latency_ms:.2f} ms")
            
            return CalibrationResults(
                is_calibrated=True,
                baseline_latency_ms=latency_results.mean_latency_ms,
                measured_latency_ms=latency_results.mean_latency_ms,
                deviation_pct=0.0,
                cv_pct=latency_results.cv_percent,
                runs_completed=latency_results.measurement_runs,
                gpu_info=gpu_info,
                system_info=system_info,
                warnings=warnings_list
            )
        
        # Validate against existing baseline
        baseline_latency = self.baseline_data['latency_ms']
        measured_latency = latency_results.mean_latency_ms
        
        deviation_pct = abs(measured_latency - baseline_latency) / baseline_latency * 100
        
        # Check system compatibility
        compatibility_warnings = self._check_system_compatibility(system_info, gpu_info)
        warnings_list.extend(compatibility_warnings)
        
        # Check performance deviation
        is_within_tolerance = deviation_pct <= self.config.baseline_tolerance_pct
        
        if not is_within_tolerance:
            warnings_list.append(
                f"Performance deviation ({deviation_pct:.1f}%) exceeds tolerance "
                f"({self.config.baseline_tolerance_pct}%)"
            )
        
        # Check measurement stability
        if latency_results.cv_percent > self.config.target_cv_pct:
            warnings_list.append(
                f"High measurement variability (CV: {latency_results.cv_percent:.1f}%) "
                f"exceeds target ({self.config.target_cv_pct}%)"
            )
        
        # Log results
        if is_within_tolerance:
            logger.info(f"System calibration PASSED: {deviation_pct:.1f}% deviation from baseline")
        else:
            logger.warning(f"System calibration FAILED: {deviation_pct:.1f}% deviation from baseline")
        
        return CalibrationResults(
            is_calibrated=is_within_tolerance,
            baseline_latency_ms=baseline_latency,
            measured_latency_ms=measured_latency,
            deviation_pct=deviation_pct,
            cv_pct=latency_results.cv_percent,
            runs_completed=latency_results.measurement_runs,
            gpu_info=gpu_info,
            system_info=system_info,
            warnings=warnings_list
        )
    
    def _check_system_compatibility(
        self,
        current_system: Dict[str, Any],
        current_gpu: Dict[str, Any]
    ) -> list:
        """Check system compatibility with baseline."""
        warnings_list = []
        
        if not self.baseline_data:
            return warnings_list
        
        baseline_system = self.baseline_data.get('system_info', {})
        baseline_gpu = self.baseline_data.get('gpu_info', {})
        
        # Check critical system components
        if baseline_system.get('platform') != current_system.get('platform'):
            warnings_list.append("Platform mismatch with baseline")
        
        if baseline_gpu.get('device_name') != current_gpu.get('device_name'):
            warnings_list.append("GPU model mismatch with baseline")
        
        # Check PyTorch version
        baseline_torch = baseline_system.get('torch_version')
        current_torch = current_system.get('torch_version')
        if baseline_torch and current_torch and baseline_torch != current_torch:
            warnings_list.append(f"PyTorch version mismatch: {current_torch} vs {baseline_torch}")
        
        # Check CUDA version
        baseline_cuda = baseline_system.get('cuda_version')
        current_cuda = current_system.get('cuda_version')
        if baseline_cuda and current_cuda and baseline_cuda != current_cuda:
            warnings_list.append(f"CUDA version mismatch: {current_cuda} vs {baseline_cuda}")
        
        return warnings_list
    
    def reset_baseline(self) -> None:
        """Reset the baseline data."""
        self.baseline_data = None
        if self.config.baseline_file:
            baseline_path = Path(self.config.baseline_file)
            if baseline_path.exists():
                try:
                    baseline_path.unlink()
                    logger.info("Baseline file removed")
                except Exception as e:
                    logger.error(f"Failed to remove baseline file: {e}")


def validate_system_performance(
    config: Optional[CalibrationConfig] = None,
    force_recalibrate: bool = False
) -> CalibrationResults:
    """
    Validate system performance against baseline.
    
    This is a convenient wrapper function for system calibration.
    
    Args:
        config: Calibration configuration
        force_recalibrate: Force recalibration even if baseline exists
        
    Returns:
        CalibrationResults with validation status
    """
    calibrator = SystemCalibrator(config)
    return calibrator.calibrate_system(force_recalibrate)


def abort_if_system_invalid(
    config: Optional[CalibrationConfig] = None,
    force_recalibrate: bool = False
) -> None:
    """
    Abort experiment if system performance is invalid.
    
    Args:
        config: Calibration configuration
        force_recalibrate: Force recalibration even if baseline exists
        
    Raises:
        RuntimeError: If system calibration fails
    """
    results = validate_system_performance(config, force_recalibrate)
    
    if not results.is_calibrated:
        error_msg = "System calibration failed - aborting experiment"
        if results.error_message:
            error_msg += f": {results.error_message}"
        if results.deviation_pct is not None:
            error_msg += f" (deviation: {results.deviation_pct:.1f}%)"
        
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    # Log warnings
    for warning in results.warnings:
        logger.warning(f"Calibration warning: {warning}")
    
    logger.info("System calibration passed - proceeding with experiment")