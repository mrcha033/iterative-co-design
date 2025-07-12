"""
Experiment runner for the iterative co-design framework.

This module implements the main experiment orchestration logic, including
strategy dispatch, result collection, and experiment provenance tracking.
"""
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import warnings

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from src.models.manager import ModelManager
from src.utils.dataset_manager import DatasetManager
from src.co_design.iasp import IASPPermutationOptimizer
from src.co_design.hds import HDSOptimizer, HDSConfig, apply_hds_to_model
from src.co_design.ptq import PostTrainingQuantizer, PTQConfig, quantize_model
from src.utils.exceptions import IterativeCoDesignError
from src.profiler.latency import LatencyProfiler, LatencyConfig, benchmark_model_latency
from src.profiler.ncu import NsightComputeProfiler, NCUConfig, collect_hardware_metrics
from src.profiler.calibration import SystemCalibrator, CalibrationConfig, validate_system_performance


class ExperimentRunner:
    """Main experiment runner class."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize experiment runner.
        
        Args:
            config: Experiment configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Set up deterministic execution
        self._setup_reproducibility()
        
        # Initialize managers
        self.model_manager = ModelManager()
        self.dataset_manager = DatasetManager()
        
        # Initialize profiling components
        self._setup_profiling()
        
        # Initialize components
        self.model = None
        self.dataloader = None
        self.baseline_latency = None
        
        # Results tracking
        self.results = {
            'config': config,
            'timestamp': datetime.now().isoformat(),
            'experiment_id': self._generate_experiment_id(),
            'reproducibility_info': self._get_reproducibility_info()
        }
        
        # Setup output directory
        self._setup_output_directory()
    
    def _setup_reproducibility(self) -> None:
        """Setup deterministic execution environment."""
        seed = self.config['experiment']['seed']
        
        # Set Python random seed
        import random
        random.seed(seed)
        
        # Set NumPy random seed  
        np.random.seed(seed)
        
        # Set PyTorch random seeds
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Configure deterministic algorithms
        if self.config['reproducibility']['deterministic']:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
            # Enable deterministic algorithms in PyTorch
            torch.use_deterministic_algorithms(True, warn_only=True)
        
        if self.config['reproducibility']['cuda_deterministic']:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        
        self.logger.info(f"Reproducibility setup complete with seed: {seed}")
    
    def _setup_profiling(self) -> None:
        """Setup profiling components."""
        # Initialize latency profiler
        latency_config = LatencyConfig(
            warmup_runs=self.config['benchmark']['warmup_runs'],
            measurement_runs=self.config['benchmark']['num_runs'],
            use_cuda_events=self.config['benchmark']['use_cuda_events'],
            cuda_sync=self.config['benchmark']['cuda_sync'],
            enable_autograd=False,
            memory_cleanup=True,
            statistical_validation=True
        )
        self.latency_profiler = LatencyProfiler(latency_config)
        
        # Initialize hardware profiler if enabled
        self.hardware_profiler = None
        if self.config['profiling']['enabled']:
            ncu_config = NCUConfig(
                metrics=self.config['profiling']['metrics'],
                timeout_seconds=300,
                output_format='csv'
            )
            self.hardware_profiler = NsightComputeProfiler(ncu_config)
        
        # Initialize system calibrator
        calibration_config = CalibrationConfig(
            baseline_tolerance_pct=15.0,
            min_runs=3,
            max_runs=self.config['benchmark']['num_runs'],
            warmup_runs=self.config['benchmark']['warmup_runs']
        )
        self.system_calibrator = SystemCalibrator(calibration_config)
        
        self.logger.info("Profiling components initialized")
    
    def _generate_experiment_id(self) -> str:
        """Generate unique experiment ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        strategy = self.config['experiment']['strategy']
        model = self.config['model']['name']
        return f"{strategy}_{model}_{timestamp}"
    
    def _get_reproducibility_info(self) -> Dict[str, Any]:
        """Collect reproducibility information."""
        return {
            'seed': self.config['experiment']['seed'],
            'torch_version': torch.__version__,
            'cuda_version': torch.version.cuda,
            'python_version': os.sys.version,
            'hostname': os.uname().nodename if hasattr(os, 'uname') else 'unknown',
            'gpu_info': self._get_gpu_info()
        }
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information for reproducibility."""
        if not torch.cuda.is_available():
            return {'available': False}
        
        gpu_id = self.config['hardware']['gpu_id']
        return {
            'available': True,
            'device_count': torch.cuda.device_count(),
            'current_device': gpu_id,
            'device_name': torch.cuda.get_device_name(gpu_id),
            'device_capability': torch.cuda.get_device_capability(gpu_id),
            'memory_total': torch.cuda.get_device_properties(gpu_id).total_memory,
        }
    
    def _setup_output_directory(self) -> None:
        """Setup output directory for experiment results."""
        base_dir = Path(self.config['experiment']['output_dir'])
        experiment_dir = base_dir / self.results['experiment_id']
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.output_dir = experiment_dir
        self.results['output_dir'] = str(experiment_dir)
        
        self.logger.info(f"Output directory: {experiment_dir}")
        
        # Save initial config
        config_path = experiment_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2, default=str)
    
    def run(self) -> Dict[str, Any]:
        """
        Run the experiment according to the configured strategy.
        
        Returns:
            Dictionary containing all experiment results
        """
        strategy = self.config['experiment']['strategy']
        
        self.logger.info(f"Running experiment with strategy: {strategy}")
        start_time = time.time()
        
        try:
            # System calibration check
            self._validate_system_calibration()
            
            # Load model and data
            self._load_model_and_data()
            
            # Run baseline benchmark
            self._run_baseline_benchmark()
            
            # Execute strategy
            if strategy == 'baseline':
                self._run_baseline_strategy()
            elif strategy == 'permute_only':
                self._run_permute_only_strategy()
            elif strategy == 'sparsity_only':
                self._run_sparsity_only_strategy()
            elif strategy == 'linear_sparsity':
                self._run_linear_sparsity_strategy()
            elif strategy == 'iterative_sparsity':
                self._run_iterative_sparsity_strategy()
            elif strategy == 'linear_quant_permute_first':
                self._run_linear_quant_permute_first_strategy()
            elif strategy == 'linear_quant_quant_first':
                self._run_linear_quant_quant_first_strategy()
            elif strategy == 'iterative_quant':
                self._run_iterative_quant_strategy()
            else:
                raise IterativeCoDesignError(f"Unknown strategy: {strategy}")
            
            # Final benchmark
            self._run_final_benchmark()
            
            # Calculate improvements
            self._calculate_improvements()
            
            self.results['status'] = 'completed'
            self.results['duration_seconds'] = time.time() - start_time
            
        except Exception as e:
            self.results['status'] = 'failed'
            self.results['error'] = str(e)
            self.results['duration_seconds'] = time.time() - start_time
            raise
        
        finally:
            # Save final results
            self._save_results()
        
        return self.results
    
    def _validate_system_calibration(self) -> None:
        """Validate system performance calibration."""
        self.logger.info("Validating system calibration...")
        
        try:
            calibration_results = self.system_calibrator.calibrate_system()
            
            self.results['system_calibration'] = calibration_results.to_dict()
            
            if not calibration_results.is_calibrated:
                error_msg = "System calibration failed"
                if calibration_results.error_message:
                    error_msg += f": {calibration_results.error_message}"
                if calibration_results.deviation_pct is not None:
                    error_msg += f" (deviation: {calibration_results.deviation_pct:.1f}%)"
                
                self.logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # Log warnings
            for warning in calibration_results.warnings:
                self.logger.warning(f"Calibration warning: {warning}")
            
            self.logger.info("System calibration passed")
            
        except Exception as e:
            self.logger.error(f"System calibration failed: {e}")
            self.results['system_calibration'] = {'error': str(e)}
            raise
    
    def _load_model_and_data(self) -> None:
        """Load model and dataset."""
        self.logger.info("Loading model and dataset...")
        
        # Load model
        model_config = self.config['model']
        self.model = self.model_manager.load_model(
            model_name=model_config['name'],
            model_path=model_config.get('pretrained_path'),
            precision=model_config.get('precision', 'float16'),
            device=self.config['hardware']['device']
        )
        
        # Load dataset
        dataset_config = self.config['dataset']
        self.dataloader = self.dataset_manager.get_dataloader(
            dataset_name=dataset_config['name'],
            batch_size=dataset_config['batch_size'],
            sequence_length=dataset_config.get('sequence_length', 512),
            num_samples=dataset_config['num_samples']
        )
        
        self.logger.info(f"Model loaded: {model_config['name']}")
        self.logger.info(f"Dataset loaded: {dataset_config['name']}")
    
    def _run_baseline_benchmark(self) -> None:
        """Run baseline performance benchmark."""
        self.logger.info("Running baseline benchmark...")
        
        latency_results = self._benchmark_model(self.model, "baseline")
        self.baseline_latency = latency_results['mean_latency_ms']
        
        self.results['baseline_benchmark'] = latency_results
        self.logger.info(f"Baseline latency: {self.baseline_latency:.2f} ± {latency_results['std_latency_ms']:.2f} ms")
    
    def _benchmark_model(self, model: nn.Module, stage: str) -> Dict[str, Any]:
        """Benchmark model performance with advanced profiling."""
        device = self.config['hardware']['device']
        
        model.eval()
        model.to(device)
        
        # Get first batch for benchmarking
        inputs = None
        for batch in self.dataloader:
            if isinstance(batch, (list, tuple)):
                inputs = batch[0].to(device)
            else:
                inputs = batch.to(device)
            break
        
        if inputs is None:
            raise RuntimeError("No input data available for benchmarking")
        
        # Run latency profiling
        latency_results = benchmark_model_latency(model, inputs, self.latency_profiler.config)
        
        # Collect hardware metrics if enabled
        hardware_results = None
        if self.hardware_profiler is not None:
            try:
                hardware_output_file = self.output_dir / f'ncu_profile_{stage}.csv'
                hardware_results = self.hardware_profiler.profile_model(
                    model, inputs, hardware_output_file
                )
                self.logger.info(f"Hardware profiling saved to {hardware_output_file}")
            except Exception as e:
                self.logger.warning(f"Hardware profiling failed for {stage}: {e}")
                hardware_results = None
        
        # Combine results
        results = {
            'latency': latency_results.to_dict(),
            'stage': stage
        }
        
        if hardware_results is not None:
            results['hardware'] = hardware_results.to_dict()
        
        # Legacy compatibility fields
        results.update({
            'mean_latency_ms': latency_results.mean_latency_ms,
            'std_latency_ms': latency_results.std_latency_ms,
            'min_latency_ms': latency_results.min_latency_ms,
            'max_latency_ms': latency_results.max_latency_ms,
            'num_runs': latency_results.measurement_runs
        })
        
        return results
    
    def _apply_iasp_permutation(self, model: nn.Module, stage: str) -> Dict[str, Any]:
        """Apply IASP permutation to model."""
        self.logger.info(f"Applying IASP permutation ({stage})...")
        
        iasp_config = self.config['iasp']
        
        # Check for precomputed correlation matrix
        correlation_matrix = None
        if self.config.get('precomputed_correlation'):
            correlation_path = Path(self.config['precomputed_correlation'])
            if correlation_path.exists():
                self.logger.info(f"Loading precomputed correlation matrix from {correlation_path}")
                correlation_matrix = torch.load(correlation_path)
        
        # Initialize IASP optimizer
        iasp = IASPPermutationOptimizer(
            layer_name=iasp_config['layer_name'],
            num_clusters=iasp_config['num_clusters'],
            method=iasp_config.get('method', 'spectral'),
            correlation_threshold=iasp_config.get('correlation_threshold', 0.1)
        )
        
        # Compute permutation
        permutation, iasp_results = iasp.compute_permutation(
            model=model,
            dataloader=self.dataloader,
            correlation_matrix=correlation_matrix
        )
        
        # Apply permutation
        iasp.apply_permutation(model, permutation)
        
        # Save permutation if requested
        if self.config['experiment']['save_intermediate']:
            perm_path = self.output_dir / f'permutation_{stage}.pt'
            torch.save(permutation, perm_path)
            self.logger.info(f"Permutation saved to {perm_path}")
        
        return {
            'permutation': permutation.tolist(),
            'modularity': iasp_results.get('modularity', 0.0),
            'num_clusters': iasp_config['num_clusters'],
            'stage': stage
        }
    
    def _apply_hds_sparsity(self, model: nn.Module, stage: str) -> Dict[str, Any]:
        """Apply HDS sparsity to model."""
        self.logger.info(f"Applying HDS sparsity ({stage})...")
        
        hds_config_dict = self.config['hds']
        
        # Create HDS configuration
        hds_config = HDSConfig(
            sparsity_ratio=hds_config_dict['pattern'],
            target_sparsity=hds_config_dict.get('sparsity_ratio', 0.5),
            learning_rate=hds_config_dict['learning_rate'],
            num_epochs=hds_config_dict['num_epochs'],
            gumbel_temperature=hds_config_dict.get('gumbel_temperature', 1.0)
        )
        
        # Apply HDS
        sparse_model, hds_results = apply_hds_to_model(
            model, self.dataloader, hds_config
        )
        
        return {
            'sparsity_pattern': hds_config_dict['pattern'],
            'target_sparsity': hds_config.target_sparsity,
            'final_sparsity': hds_results.get('final_sparsity', {}),
            'training_epochs': hds_config.num_epochs,
            'stage': stage
        }
    
    def _apply_ptq_quantization(self, model: nn.Module, stage: str) -> Dict[str, Any]:
        """Apply PTQ quantization to model."""
        self.logger.info(f"Applying PTQ quantization ({stage})...")
        
        ptq_config_dict = self.config['ptq']
        
        # Create PTQ configuration
        ptq_config = PTQConfig(
            weight_bits=ptq_config_dict['bits'],
            activation_bits=ptq_config_dict['bits'],
            calibration_samples=ptq_config_dict['calibration_samples'],
            symmetric=ptq_config_dict['scheme'] == 'symmetric'
        )
        
        # Apply quantization
        quantized_model, ptq_results = quantize_model(
            model, self.dataloader, ptq_config
        )
        
        return {
            'quantization_bits': ptq_config_dict['bits'],
            'quantization_scheme': ptq_config_dict['scheme'],
            'calibration_samples': ptq_config.calibration_samples,
            'quantization_stats': ptq_results,
            'stage': stage
        }
    
    # Strategy implementations
    def _run_baseline_strategy(self) -> None:
        """Run baseline strategy (no optimization)."""
        self.logger.info("Running baseline strategy")
        self.results['strategy_results'] = {'type': 'baseline'}
    
    def _run_permute_only_strategy(self) -> None:
        """Run permute-only strategy."""
        self.logger.info("Running permute-only strategy")
        
        iasp_results = self._apply_iasp_permutation(self.model, "permute_only")
        
        self.results['strategy_results'] = {
            'type': 'permute_only',
            'iasp_results': iasp_results
        }
    
    def _run_sparsity_only_strategy(self) -> None:
        """Run sparsity-only strategy."""
        self.logger.info("Running sparsity-only strategy")
        
        hds_results = self._apply_hds_sparsity(self.model, "sparsity_only")
        
        self.results['strategy_results'] = {
            'type': 'sparsity_only',
            'hds_results': hds_results
        }
    
    def _run_linear_sparsity_strategy(self) -> None:
        """Run linear sparsity strategy (HDS -> IASP)."""
        self.logger.info("Running linear sparsity strategy")
        
        # Step 1: Apply HDS
        hds_results = self._apply_hds_sparsity(self.model, "linear_hds")
        
        # Step 2: Apply IASP
        iasp_results = self._apply_iasp_permutation(self.model, "linear_iasp")
        
        self.results['strategy_results'] = {
            'type': 'linear_sparsity',
            'hds_results': hds_results,
            'iasp_results': iasp_results
        }
    
    def _run_iterative_sparsity_strategy(self) -> None:
        """Run iterative sparsity strategy (IASP -> HDS -> IASP)."""
        self.logger.info("Running iterative sparsity strategy")
        
        num_iterations = self.config['experiment']['num_iterations']
        iteration_results = []
        
        for iteration in range(num_iterations):
            self.logger.info(f"Starting iteration {iteration + 1}/{num_iterations}")
            
            # Step 1: Apply IASP
            iasp_results = self._apply_iasp_permutation(self.model, f"iter{iteration}_iasp")
            
            # Step 2: Apply HDS 
            hds_results = self._apply_hds_sparsity(self.model, f"iter{iteration}_hds")
            
            # Step 3: Apply IASP again
            iasp2_results = self._apply_iasp_permutation(self.model, f"iter{iteration}_iasp2")
            
            iteration_results.append({
                'iteration': iteration,
                'initial_iasp': iasp_results,
                'hds': hds_results,
                'final_iasp': iasp2_results
            })
        
        self.results['strategy_results'] = {
            'type': 'iterative_sparsity',
            'num_iterations': num_iterations,
            'iterations': iteration_results
        }
    
    def _run_linear_quant_permute_first_strategy(self) -> None:
        """Run linear quantization strategy (IASP -> PTQ)."""
        self.logger.info("Running linear quantization strategy (permute first)")
        
        # Step 1: Apply IASP
        iasp_results = self._apply_iasp_permutation(self.model, "linear_iasp")
        
        # Step 2: Apply PTQ
        ptq_results = self._apply_ptq_quantization(self.model, "linear_ptq")
        
        self.results['strategy_results'] = {
            'type': 'linear_quant_permute_first',
            'iasp_results': iasp_results,
            'ptq_results': ptq_results
        }
    
    def _run_linear_quant_quant_first_strategy(self) -> None:
        """Run linear quantization strategy (PTQ -> IASP)."""
        self.logger.info("Running linear quantization strategy (quantize first)")
        
        # Step 1: Apply PTQ
        ptq_results = self._apply_ptq_quantization(self.model, "linear_ptq")
        
        # Step 2: Apply IASP
        iasp_results = self._apply_iasp_permutation(self.model, "linear_iasp")
        
        self.results['strategy_results'] = {
            'type': 'linear_quant_quant_first',
            'ptq_results': ptq_results,
            'iasp_results': iasp_results
        }
    
    def _run_iterative_quant_strategy(self) -> None:
        """Run iterative quantization strategy (IASP -> PTQ -> IASP)."""
        self.logger.info("Running iterative quantization strategy")
        
        # Step 1: Apply IASP
        iasp_results = self._apply_iasp_permutation(self.model, "iter_iasp")
        
        # Step 2: Apply PTQ
        ptq_results = self._apply_ptq_quantization(self.model, "iter_ptq")
        
        # Step 3: Apply IASP again
        iasp2_results = self._apply_iasp_permutation(self.model, "iter_iasp2")
        
        self.results['strategy_results'] = {
            'type': 'iterative_quant',
            'initial_iasp': iasp_results,
            'ptq': ptq_results,
            'final_iasp': iasp2_results
        }
    
    def _run_final_benchmark(self) -> None:
        """Run final benchmark after optimization."""
        self.logger.info("Running final benchmark...")
        
        final_results = self._benchmark_model(self.model, "final")
        self.results['final_benchmark'] = final_results
        
        self.logger.info(f"Final latency: {final_results['mean_latency_ms']:.2f} ± {final_results['std_latency_ms']:.2f} ms")
    
    def _calculate_improvements(self) -> None:
        """Calculate performance improvements."""
        if 'final_benchmark' not in self.results:
            return
        
        baseline_latency = self.results['baseline_benchmark']['mean_latency_ms']
        final_latency = self.results['final_benchmark']['mean_latency_ms']
        
        improvement_ms = baseline_latency - final_latency
        improvement_pct = (improvement_ms / baseline_latency) * 100
        
        self.results['improvements'] = {
            'baseline_latency_ms': baseline_latency,
            'final_latency_ms': final_latency,
            'improvement_ms': improvement_ms,
            'improvement_pct': improvement_pct,
            'speedup_factor': baseline_latency / final_latency
        }
        
        self.logger.info(f"Performance improvement: {improvement_pct:.1f}% ({improvement_ms:.2f} ms)")
    
    def _save_results(self) -> None:
        """Save experiment results to files."""
        # Save JSON results
        results_path = self.output_dir / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {results_path}")
        
        # Save summary
        summary_path = self.output_dir / 'summary.txt'
        with open(summary_path, 'w') as f:
            f.write(f"Experiment Summary\n")
            f.write(f"==================\n\n")
            f.write(f"Experiment ID: {self.results['experiment_id']}\n")
            f.write(f"Timestamp: {self.results['timestamp']}\n")
            f.write(f"Strategy: {self.config['experiment']['strategy']}\n")
            f.write(f"Model: {self.config['model']['name']}\n")
            f.write(f"Dataset: {self.config['dataset']['name']}\n")
            f.write(f"Status: {self.results.get('status', 'unknown')}\n")
            f.write(f"Duration: {self.results.get('duration_seconds', 0):.1f} seconds\n\n")
            
            if 'improvements' in self.results:
                imp = self.results['improvements']
                f.write(f"Performance Results:\n")
                f.write(f"  Baseline latency: {imp['baseline_latency_ms']:.2f} ms\n")
                f.write(f"  Final latency: {imp['final_latency_ms']:.2f} ms\n")
                f.write(f"  Improvement: {imp['improvement_pct']:.1f}% ({imp['improvement_ms']:.2f} ms)\n")
                f.write(f"  Speedup factor: {imp['speedup_factor']:.2f}x\n")
        
        self.logger.info(f"Summary saved to {summary_path}")