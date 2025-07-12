"""
Performance benchmarking suite for T-004 validation.

This module implements comprehensive performance benchmarks to validate
spectral clustering and permutation application performance characteristics
with quantitative metrics and regression detection.
"""
import time
import tracemalloc
import pytest
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
import psutil
import gc
from unittest.mock import patch

from src.co_design.spectral import spectral_clustering
from src.co_design.iasp import IASPPermutationOptimizer, IASPConfig
from src.co_design.apply import apply_permutation_to_layer
from src.co_design.correlation import compute_activation_correlation


class PerformanceProfiler:
    """High-precision performance profiling utility."""
    
    def __init__(self, enable_memory_profiling: bool = True):
        self.enable_memory = enable_memory_profiling
        self.results = {}
    
    def profile_execution(self, func: Callable, *args, **kwargs) -> Dict:
        """
        Profile function execution with timing and memory measurements.
        
        Args:
            func: Function to profile
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Dict containing timing and memory metrics
        """
        # Memory profiling setup
        if self.enable_memory:
            tracemalloc.start()
            gc.collect()
            initial_memory = psutil.Process().memory_info().rss
        
        # High-precision timing
        start_time = time.perf_counter()
        
        # Execute function
        result = func(*args, **kwargs)
        
        # End timing
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        
        # Memory profiling results
        memory_stats = {}
        if self.enable_memory:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            final_memory = psutil.Process().memory_info().rss
            
            memory_stats = {
                'peak_memory_mb': peak / 1024 / 1024,
                'current_memory_mb': current / 1024 / 1024,
                'rss_increase_mb': (final_memory - initial_memory) / 1024 / 1024
            }
        
        return {
            'result': result,
            'execution_time_s': execution_time,
            'execution_time_ms': execution_time * 1000,
            **memory_stats
        }
    
    def benchmark_multiple_runs(self, func: Callable, num_runs: int = 5, 
                               warmup_runs: int = 2, *args, **kwargs) -> Dict:
        """
        Benchmark function with multiple runs for statistical analysis.
        
        Args:
            func: Function to benchmark
            num_runs: Number of measurement runs
            warmup_runs: Number of warmup runs
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Dict containing statistical timing metrics
        """
        # Warmup runs
        for _ in range(warmup_runs):
            func(*args, **kwargs)
        
        # Measurement runs
        times = []
        memory_peaks = []
        
        for _ in range(num_runs):
            profile_result = self.profile_execution(func, *args, **kwargs)
            times.append(profile_result['execution_time_ms'])
            if 'peak_memory_mb' in profile_result:
                memory_peaks.append(profile_result['peak_memory_mb'])
        
        times = np.array(times)
        results = {
            'mean_time_ms': np.mean(times),
            'std_time_ms': np.std(times),
            'min_time_ms': np.min(times),
            'max_time_ms': np.max(times),
            'median_time_ms': np.median(times),
            'cv_time': np.std(times) / np.mean(times),  # Coefficient of variation
            'num_runs': num_runs
        }
        
        if memory_peaks:
            memory_peaks = np.array(memory_peaks)
            results.update({
                'mean_memory_mb': np.mean(memory_peaks),
                'std_memory_mb': np.std(memory_peaks),
                'max_memory_mb': np.max(memory_peaks)
            })
        
        return results


class AlgorithmicBenchmarks:
    """Algorithmic performance benchmarks for spectral clustering and IASP."""
    
    def __init__(self):
        self.profiler = PerformanceProfiler()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def benchmark_spectral_clustering_scaling(self, matrix_sizes: List[int]) -> Dict:
        """
        Benchmark spectral clustering runtime vs matrix size.
        
        Args:
            matrix_sizes: List of matrix dimensions to test
            
        Returns:
            Dict containing scaling results
        """
        results = {}
        
        for size in matrix_sizes:
            print(f"Benchmarking spectral clustering for size {size}x{size}...")
            
            # Create test correlation matrix
            correlation_matrix = self._create_test_correlation_matrix(size)
            
            # Benchmark spectral clustering
            def run_spectral():
                return spectral_clustering(
                    correlation_matrix, 
                    num_clusters=min(8, size // 4),
                    method='spectral'
                )
            
            benchmark_result = self.profiler.benchmark_multiple_runs(
                run_spectral, num_runs=3, warmup_runs=1
            )
            
            results[size] = benchmark_result
            
            # Analyze scaling characteristics
            if len(results) > 1:
                self._analyze_scaling_trend(results)
        
        return results
    
    def benchmark_eigenvalue_computation(self, matrix_sizes: List[int]) -> Dict:
        """
        Benchmark eigenvalue computation time vs matrix size.
        
        Args:
            matrix_sizes: List of matrix dimensions to test
            
        Returns:
            Dict containing eigenvalue computation results
        """
        results = {}
        
        for size in matrix_sizes:
            correlation_matrix = self._create_test_correlation_matrix(size)
            
            def compute_eigenvalues():
                return torch.linalg.eigh(correlation_matrix)
            
            benchmark_result = self.profiler.benchmark_multiple_runs(
                compute_eigenvalues, num_runs=5, warmup_runs=2
            )
            
            results[size] = benchmark_result
        
        return results
    
    def benchmark_memory_scaling(self, matrix_sizes: List[int]) -> Dict:
        """
        Benchmark memory usage scaling with matrix size.
        
        Args:
            matrix_sizes: List of matrix dimensions to test
            
        Returns:
            Dict containing memory scaling results
        """
        results = {}
        
        for size in matrix_sizes:
            correlation_matrix = self._create_test_correlation_matrix(size)
            
            # Benchmark memory usage during spectral clustering
            def memory_intensive_operation():
                # Simulate memory-intensive operations
                eigenvalues, eigenvectors = torch.linalg.eigh(correlation_matrix)
                similarity_matrix = torch.exp(-0.5 * correlation_matrix)
                return eigenvalues, eigenvectors, similarity_matrix
            
            profile_result = self.profiler.profile_execution(memory_intensive_operation)
            
            results[size] = {
                'matrix_size': size,
                'theoretical_memory_mb': (size * size * 4) / 1024 / 1024,  # float32
                'actual_peak_memory_mb': profile_result.get('peak_memory_mb', 0),
                'memory_efficiency': profile_result.get('peak_memory_mb', 0) / ((size * size * 4) / 1024 / 1024)
            }
        
        return results
    
    def _create_test_correlation_matrix(self, size: int) -> torch.Tensor:
        """Create test correlation matrix with realistic structure."""
        # Create block-diagonal structure for realistic clustering
        block_size = size // 4
        matrix = torch.zeros(size, size, device=self.device)
        
        for i in range(0, size, block_size):
            end_idx = min(i + block_size, size)
            block_data = torch.randn(end_idx - i, end_idx - i, device=self.device)
            block_corr = torch.corrcoef(block_data)
            matrix[i:end_idx, i:end_idx] = block_corr
        
        # Add some cross-block correlations
        noise = torch.randn(size, size, device=self.device) * 0.1
        matrix = matrix + noise
        matrix = (matrix + matrix.T) / 2  # Ensure symmetry
        
        return matrix
    
    def _analyze_scaling_trend(self, results: Dict) -> None:
        """Analyze scaling trend and detect performance regressions."""
        sizes = sorted(results.keys())
        times = [results[size]['mean_time_ms'] for size in sizes]
        
        if len(sizes) >= 3:
            # Fit polynomial to detect scaling pattern
            log_sizes = np.log(sizes)
            log_times = np.log(times)
            
            # Linear fit in log space: log(time) = a * log(size) + b
            # Slope 'a' indicates scaling: 1=linear, 2=quadratic, 3=cubic
            coeffs = np.polyfit(log_sizes, log_times, 1)
            scaling_exponent = coeffs[0]
            
            print(f"Detected scaling exponent: {scaling_exponent:.2f}")
            if scaling_exponent > 2.5:
                print("⚠️  Warning: Potentially cubic or worse scaling detected")
            elif scaling_exponent > 1.8:
                print("⚠️  Warning: Super-linear scaling detected")
            else:
                print("✅ Good scaling characteristics")


class ScalabilityTests:
    """Tests for scalability and performance validation."""
    
    def __init__(self):
        self.profiler = PerformanceProfiler()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def test_block_approximation_benefits(self, large_sizes: List[int]) -> Dict:
        """
        Test performance benefits of block-wise approximation for large matrices.
        
        Args:
            large_sizes: List of large matrix dimensions to test
            
        Returns:
            Dict containing block approximation results
        """
        results = {}
        
        for size in large_sizes:
            if size < 1000:
                continue  # Only test block approximation for large matrices
            
            print(f"Testing block approximation for size {size}...")
            
            # Create large correlation matrix
            correlation_matrix = self._create_large_test_matrix(size)
            
            # Benchmark full spectral clustering (if feasible)
            full_time = None
            try:
                def full_spectral():
                    return spectral_clustering(correlation_matrix, num_clusters=16)
                
                full_result = self.profiler.profile_execution(full_spectral)
                full_time = full_result['execution_time_ms']
                full_memory = full_result.get('peak_memory_mb', 0)
            except (RuntimeError, torch.cuda.OutOfMemoryError):
                full_time = None
                full_memory = float('inf')
            
            # Benchmark block-wise approximation
            def block_spectral():
                return self._block_wise_spectral_clustering(correlation_matrix, block_size=512)
            
            block_result = self.profiler.profile_execution(block_spectral)
            block_time = block_result['execution_time_ms']
            block_memory = block_result.get('peak_memory_mb', 0)
            
            results[size] = {
                'size': size,
                'full_time_ms': full_time,
                'block_time_ms': block_time,
                'full_memory_mb': full_memory,
                'block_memory_mb': block_memory,
                'speedup_factor': full_time / block_time if full_time else None,
                'memory_reduction': (full_memory - block_memory) / full_memory if full_memory < float('inf') else None
            }
        
        return results
    
    def test_linear_vs_cubic_scaling(self, size_range: range) -> Dict:
        """
        Validate linear vs cubic scaling characteristics.
        
        Args:
            size_range: Range of sizes to test
            
        Returns:
            Dict containing scaling validation results
        """
        sizes = list(size_range)
        linear_times = []
        cubic_times = []
        
        for size in sizes:
            # Test linear operation (matrix-vector multiply)
            matrix = torch.randn(size, size, device=self.device)
            vector = torch.randn(size, device=self.device)
            
            def linear_op():
                return torch.matmul(matrix, vector)
            
            linear_result = self.profiler.benchmark_multiple_runs(linear_op, num_runs=10)
            linear_times.append(linear_result['mean_time_ms'])
            
            # Test cubic operation (matrix-matrix-matrix multiply)
            def cubic_op():
                temp = torch.matmul(matrix, matrix)
                return torch.matmul(temp, matrix)
            
            cubic_result = self.profiler.benchmark_multiple_runs(cubic_op, num_runs=5)
            cubic_times.append(cubic_result['mean_time_ms'])
        
        # Analyze scaling
        return self._analyze_scaling_validation(sizes, linear_times, cubic_times)
    
    def _create_large_test_matrix(self, size: int) -> torch.Tensor:
        """Create large test matrix with block structure."""
        # Use efficient block construction for large matrices
        num_blocks = 8
        block_size = size // num_blocks
        
        matrix = torch.zeros(size, size, device=self.device)
        
        for i in range(num_blocks):
            start_idx = i * block_size
            end_idx = min((i + 1) * block_size, size)
            
            # Create smaller blocks to avoid memory issues
            block = torch.randn(end_idx - start_idx, end_idx - start_idx, device=self.device)
            block = torch.corrcoef(block)
            matrix[start_idx:end_idx, start_idx:end_idx] = block
        
        return matrix
    
    def _block_wise_spectral_clustering(self, matrix: torch.Tensor, block_size: int = 512) -> torch.Tensor:
        """Implement block-wise approximation for large matrices."""
        size = matrix.shape[0]
        num_blocks = (size + block_size - 1) // block_size
        
        block_results = []
        
        for i in range(num_blocks):
            start_idx = i * block_size
            end_idx = min((i + 1) * block_size, size)
            
            block = matrix[start_idx:end_idx, start_idx:end_idx]
            block_clustering = spectral_clustering(block, num_clusters=4)
            block_results.append(block_clustering + start_idx)  # Offset indices
        
        # Combine block results
        full_clustering = torch.cat(block_results)
        return full_clustering
    
    def _analyze_scaling_validation(self, sizes: List[int], linear_times: List[float], 
                                  cubic_times: List[float]) -> Dict:
        """Analyze and validate scaling characteristics."""
        sizes_np = np.array(sizes)
        linear_times_np = np.array(linear_times)
        cubic_times_np = np.array(cubic_times)
        
        # Fit scaling models
        log_sizes = np.log(sizes_np)
        
        # Linear scaling fit
        linear_coeffs = np.polyfit(log_sizes, np.log(linear_times_np), 1)
        linear_scaling = linear_coeffs[0]
        
        # Cubic scaling fit
        cubic_coeffs = np.polyfit(log_sizes, np.log(cubic_times_np), 1)
        cubic_scaling = cubic_coeffs[0]
        
        return {
            'sizes': sizes,
            'linear_times': linear_times,
            'cubic_times': cubic_times,
            'linear_scaling_exponent': linear_scaling,
            'cubic_scaling_exponent': cubic_scaling,
            'linear_scaling_valid': abs(linear_scaling - 2.0) < 0.3,  # O(n²) for matrix-vector
            'cubic_scaling_valid': abs(cubic_scaling - 3.0) < 0.5,   # O(n³) for matrix-matrix-matrix
        }


class ComparativeAnalysis:
    """Comparative performance analysis between different approaches."""
    
    def __init__(self):
        self.profiler = PerformanceProfiler()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def compare_spectral_vs_random_permutation(self, matrix_sizes: List[int]) -> Dict:
        """
        Compare spectral clustering vs random permutation performance.
        
        Args:
            matrix_sizes: List of matrix dimensions to test
            
        Returns:
            Dict containing comparative results
        """
        results = {}
        
        for size in matrix_sizes:
            correlation_matrix = self._create_test_matrix(size)
            
            # Benchmark spectral clustering
            def spectral_clustering_run():
                return spectral_clustering(correlation_matrix, num_clusters=8)
            
            spectral_result = self.profiler.benchmark_multiple_runs(
                spectral_clustering_run, num_runs=3
            )
            
            # Benchmark random permutation
            def random_permutation_run():
                return torch.randperm(size, device=self.device)
            
            random_result = self.profiler.benchmark_multiple_runs(
                random_permutation_run, num_runs=10
            )
            
            results[size] = {
                'spectral_time_ms': spectral_result['mean_time_ms'],
                'random_time_ms': random_result['mean_time_ms'],
                'overhead_factor': spectral_result['mean_time_ms'] / random_result['mean_time_ms'],
                'spectral_memory_mb': spectral_result.get('mean_memory_mb', 0),
                'random_memory_mb': random_result.get('mean_memory_mb', 0)
            }
        
        return results
    
    def compare_dense_vs_sparse_operations(self, sparsity_levels: List[float]) -> Dict:
        """
        Compare dense vs sparse matrix operations performance.
        
        Args:
            sparsity_levels: List of sparsity ratios to test
            
        Returns:
            Dict containing dense vs sparse comparison
        """
        size = 1024
        results = {}
        
        for sparsity in sparsity_levels:
            # Create dense matrix
            dense_matrix = torch.randn(size, size, device=self.device)
            
            # Create sparse matrix
            mask = torch.rand(size, size, device=self.device) > sparsity
            sparse_values = dense_matrix[mask]
            sparse_indices = mask.nonzero().t()
            sparse_matrix = torch.sparse_coo_tensor(
                sparse_indices, sparse_values, (size, size), device=self.device
            )
            
            # Benchmark dense operations
            vector = torch.randn(size, device=self.device)
            
            def dense_matmul():
                return torch.matmul(dense_matrix, vector)
            
            dense_result = self.profiler.benchmark_multiple_runs(dense_matmul, num_runs=10)
            
            # Benchmark sparse operations
            def sparse_matmul():
                return torch.sparse.mm(sparse_matrix, vector.unsqueeze(1)).squeeze()
            
            sparse_result = self.profiler.benchmark_multiple_runs(sparse_matmul, num_runs=10)
            
            results[sparsity] = {
                'sparsity_ratio': sparsity,
                'dense_time_ms': dense_result['mean_time_ms'],
                'sparse_time_ms': sparse_result['mean_time_ms'],
                'speedup_factor': dense_result['mean_time_ms'] / sparse_result['mean_time_ms'],
                'dense_memory_mb': size * size * 4 / 1024 / 1024,  # float32
                'sparse_memory_mb': len(sparse_values) * 8 / 1024 / 1024  # indices + values
            }
        
        return results
    
    def compare_cpu_vs_gpu_performance(self, matrix_sizes: List[int]) -> Dict:
        """
        Compare CPU vs GPU performance (if available).
        
        Args:
            matrix_sizes: List of matrix dimensions to test
            
        Returns:
            Dict containing CPU vs GPU comparison
        """
        if not torch.cuda.is_available():
            return {'error': 'CUDA not available for GPU comparison'}
        
        results = {}
        
        for size in matrix_sizes:
            # CPU benchmark
            cpu_matrix = torch.randn(size, size, device='cpu')
            
            def cpu_spectral():
                return spectral_clustering(cpu_matrix, num_clusters=8)
            
            cpu_result = self.profiler.benchmark_multiple_runs(cpu_spectral, num_runs=3)
            
            # GPU benchmark
            gpu_matrix = cpu_matrix.cuda()
            
            def gpu_spectral():
                return spectral_clustering(gpu_matrix, num_clusters=8)
            
            gpu_result = self.profiler.benchmark_multiple_runs(gpu_spectral, num_runs=3)
            
            results[size] = {
                'cpu_time_ms': cpu_result['mean_time_ms'],
                'gpu_time_ms': gpu_result['mean_time_ms'],
                'gpu_speedup': cpu_result['mean_time_ms'] / gpu_result['mean_time_ms'],
                'cpu_memory_mb': cpu_result.get('mean_memory_mb', 0),
                'gpu_memory_mb': gpu_result.get('mean_memory_mb', 0)
            }
        
        return results
    
    def _create_test_matrix(self, size: int) -> torch.Tensor:
        """Create test correlation matrix."""
        data = torch.randn(size, size, device=self.device)
        return torch.corrcoef(data)


@pytest.mark.benchmark
class TestAlgorithmicPerformance:
    """Test suite for algorithmic performance benchmarks."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.benchmarks = AlgorithmicBenchmarks()
        self.small_sizes = [64, 128, 256]
        self.medium_sizes = [512, 1024, 2048]
        self.large_sizes = [4096, 8192] if torch.cuda.is_available() else [2048, 4096]
    
    def test_spectral_clustering_scaling(self):
        """Test spectral clustering runtime scaling."""
        results = self.benchmarks.benchmark_spectral_clustering_scaling(self.small_sizes)
        
        # Verify results structure
        for size in self.small_sizes:
            assert size in results
            assert 'mean_time_ms' in results[size]
            assert results[size]['mean_time_ms'] > 0
            assert results[size]['cv_time'] < 0.2  # Low variance
        
        # Check scaling is reasonable (not worse than O(n³))
        times = [results[size]['mean_time_ms'] for size in self.small_sizes]
        ratios = [times[i+1] / times[i] for i in range(len(times)-1)]
        
        # Should not have extreme scaling (factor > 16 suggests > O(n³))
        for ratio in ratios:
            assert ratio < 16, f"Scaling ratio {ratio} suggests poor algorithmic complexity"
    
    def test_eigenvalue_computation_scaling(self):
        """Test eigenvalue computation scaling."""
        results = self.benchmarks.benchmark_eigenvalue_computation(self.small_sizes)
        
        for size in self.small_sizes:
            assert size in results
            assert results[size]['mean_time_ms'] > 0
            
        # Eigenvalue computation should be roughly O(n³)
        if len(self.small_sizes) >= 2:
            size_ratio = self.small_sizes[1] / self.small_sizes[0]
            time_ratio = results[self.small_sizes[1]]['mean_time_ms'] / results[self.small_sizes[0]]['mean_time_ms']
            
            # Allow for some variance in timing, but check general scaling
            expected_ratio = size_ratio ** 3  # O(n³)
            assert time_ratio < expected_ratio * 3, "Eigenvalue computation scaling worse than expected"
    
    def test_memory_scaling_efficiency(self):
        """Test memory usage scaling efficiency."""
        results = self.benchmarks.benchmark_memory_scaling(self.small_sizes)
        
        for size in self.small_sizes:
            assert size in results
            efficiency = results[size]['memory_efficiency']
            
            # Memory efficiency should be reasonable (between 1x and 10x theoretical)
            assert 0.5 < efficiency < 10, f"Memory efficiency {efficiency} outside reasonable range"
            
            # Larger matrices should not have drastically worse efficiency
            if size > self.small_sizes[0]:
                prev_size = self.small_sizes[self.small_sizes.index(size) - 1]
                prev_efficiency = results[prev_size]['memory_efficiency']
                
                # Efficiency shouldn't degrade by more than 3x
                assert efficiency < prev_efficiency * 3, "Memory efficiency degrading too rapidly"


@pytest.mark.benchmark
class TestScalabilityPerformance:
    """Test suite for scalability performance validation."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.scalability = ScalabilityTests()
        self.size_range = range(100, 500, 100)
        self.large_sizes = [2048, 4096] if torch.cuda.is_available() else [1024, 2048]
    
    def test_linear_vs_cubic_scaling_validation(self):
        """Test validation of linear vs cubic scaling."""
        results = self.scalability.test_linear_vs_cubic_scaling(self.size_range)
        
        # Verify scaling analysis
        assert 'linear_scaling_exponent' in results
        assert 'cubic_scaling_exponent' in results
        assert 'linear_scaling_valid' in results
        assert 'cubic_scaling_valid' in results
        
        # Linear operations should have quadratic scaling (matrix-vector)
        linear_exp = results['linear_scaling_exponent']
        assert 1.5 < linear_exp < 2.5, f"Linear scaling exponent {linear_exp} not in expected range"
        
        # Cubic operations should have cubic scaling
        cubic_exp = results['cubic_scaling_exponent']
        assert 2.5 < cubic_exp < 3.5, f"Cubic scaling exponent {cubic_exp} not in expected range"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_block_approximation_benefits(self):
        """Test performance benefits of block-wise approximation."""
        results = self.scalability.test_block_approximation_benefits([1024, 2048])
        
        for size, result in results.items():
            if result['speedup_factor'] is not None:
                # Block approximation should provide speedup for large matrices
                assert result['speedup_factor'] > 1, f"No speedup from block approximation at size {size}"
                
            # Block approximation should use less memory
            if result['memory_reduction'] is not None:
                assert result['memory_reduction'] > 0, f"No memory reduction from block approximation at size {size}"


@pytest.mark.benchmark  
class TestComparativePerformance:
    """Test suite for comparative performance analysis."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.analysis = ComparativeAnalysis()
        self.test_sizes = [256, 512, 1024]
        self.sparsity_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    def test_spectral_vs_random_comparison(self):
        """Test spectral clustering vs random permutation comparison."""
        results = self.analysis.compare_spectral_vs_random_permutation(self.test_sizes)
        
        for size in self.test_sizes:
            assert size in results
            result = results[size]
            
            # Spectral clustering should be slower than random (but provide value)
            assert result['overhead_factor'] > 1, "Spectral clustering not showing expected overhead"
            
            # Overhead should be reasonable (not more than 1000x)
            assert result['overhead_factor'] < 1000, f"Spectral overhead {result['overhead_factor']} too high"
            
            # Both should complete successfully
            assert result['spectral_time_ms'] > 0
            assert result['random_time_ms'] > 0
    
    def test_dense_vs_sparse_comparison(self):
        """Test dense vs sparse operations comparison."""
        results = self.analysis.compare_dense_vs_sparse_operations(self.sparsity_levels)
        
        for sparsity in self.sparsity_levels:
            assert sparsity in results
            result = results[sparsity]
            
            # Higher sparsity should generally provide better speedup
            if sparsity > 0.5:
                assert result['speedup_factor'] > 1, f"No speedup at sparsity {sparsity}"
            
            # Sparse memory usage should be lower for high sparsity
            if sparsity > 0.7:
                assert result['sparse_memory_mb'] < result['dense_memory_mb']
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_cpu_vs_gpu_comparison(self):
        """Test CPU vs GPU performance comparison."""
        results = self.analysis.compare_cpu_vs_gpu_performance([512, 1024])
        
        if 'error' not in results:
            for size in [512, 1024]:
                assert size in results
                result = results[size]
                
                # GPU should generally be faster for larger matrices
                if size >= 1024:
                    assert result['gpu_speedup'] > 1, f"No GPU speedup at size {size}"
                
                # Both should complete successfully
                assert result['cpu_time_ms'] > 0
                assert result['gpu_time_ms'] > 0


class PerformanceValidator:
    """Validates performance requirements and detects regressions."""
    
    def __init__(self, baseline_file: Optional[str] = None):
        self.baseline_file = baseline_file
        self.baseline_data = self._load_baseline() if baseline_file else {}
    
    def validate_timing_requirements(self, results: Dict, max_time_ms: float = 5000) -> bool:
        """
        Validate that operations complete within timing requirements.
        
        Args:
            results: Benchmark results
            max_time_ms: Maximum allowed execution time
            
        Returns:
            bool: True if timing requirements are met
        """
        violations = []
        
        def check_timing(data, path=""):
            if isinstance(data, dict):
                if 'mean_time_ms' in data:
                    if data['mean_time_ms'] > max_time_ms:
                        violations.append(f"{path}: {data['mean_time_ms']:.2f}ms > {max_time_ms}ms")
                else:
                    for key, value in data.items():
                        check_timing(value, f"{path}.{key}" if path else key)
        
        check_timing(results)
        
        if violations:
            print("Timing requirement violations:")
            for violation in violations:
                print(f"  ❌ {violation}")
            return False
        
        print("✅ All timing requirements met")
        return True
    
    def validate_memory_constraints(self, results: Dict, max_memory_mb: float = 2048) -> bool:
        """
        Validate that operations stay within memory constraints.
        
        Args:
            results: Benchmark results
            max_memory_mb: Maximum allowed memory usage in MB
            
        Returns:
            bool: True if memory constraints are met
        """
        violations = []
        
        def check_memory(data, path=""):
            if isinstance(data, dict):
                if 'peak_memory_mb' in data:
                    if data['peak_memory_mb'] > max_memory_mb:
                        violations.append(f"{path}: {data['peak_memory_mb']:.2f}MB > {max_memory_mb}MB")
                elif 'mean_memory_mb' in data:
                    if data['mean_memory_mb'] > max_memory_mb:
                        violations.append(f"{path}: {data['mean_memory_mb']:.2f}MB > {max_memory_mb}MB")
                else:
                    for key, value in data.items():
                        check_memory(value, f"{path}.{key}" if path else key)
        
        check_memory(results)
        
        if violations:
            print("Memory constraint violations:")
            for violation in violations:
                print(f"  ❌ {violation}")
            return False
        
        print("✅ All memory constraints met")
        return True
    
    def detect_performance_regression(self, current_results: Dict, tolerance: float = 0.2) -> bool:
        """
        Detect performance regressions compared to baseline.
        
        Args:
            current_results: Current benchmark results
            tolerance: Allowed performance degradation (0.2 = 20%)
            
        Returns:
            bool: True if no significant regressions detected
        """
        if not self.baseline_data:
            print("No baseline data available for regression detection")
            return True
        
        regressions = []
        
        def compare_metrics(current, baseline, path=""):
            if isinstance(current, dict) and isinstance(baseline, dict):
                for key in current:
                    if key in baseline:
                        if key.endswith('_time_ms'):
                            current_val = current[key]
                            baseline_val = baseline[key]
                            
                            if current_val > baseline_val * (1 + tolerance):
                                regression_pct = (current_val - baseline_val) / baseline_val * 100
                                regressions.append(f"{path}.{key}: {regression_pct:.1f}% slower")
                        else:
                            compare_metrics(current[key], baseline[key], f"{path}.{key}" if path else key)
        
        compare_metrics(current_results, self.baseline_data)
        
        if regressions:
            print("Performance regressions detected:")
            for regression in regressions:
                print(f"  ⚠️  {regression}")
            return False
        
        print("✅ No significant performance regressions detected")
        return True
    
    def _load_baseline(self) -> Dict:
        """Load baseline performance data."""
        try:
            import json
            with open(self.baseline_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}


if __name__ == '__main__':
    # Quick performance validation
    print("Running performance benchmark suite...")
    
    # Run basic benchmarks
    benchmarks = AlgorithmicBenchmarks()
    results = benchmarks.benchmark_spectral_clustering_scaling([128, 256, 512])
    
    # Validate performance
    validator = PerformanceValidator()
    
    timing_ok = validator.validate_timing_requirements(results, max_time_ms=10000)
    memory_ok = validator.validate_memory_constraints(results, max_memory_mb=1024)
    
    if timing_ok and memory_ok:
        print("✅ Performance validation passed")
    else:
        print("❌ Performance validation failed")