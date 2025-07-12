"""
Performance testing module for the iterative co-design framework.

This module provides comprehensive performance benchmarking and validation
utilities for the spectral clustering and permutation optimization algorithms.
"""

from .test_benchmarks import (
    PerformanceProfiler,
    AlgorithmicBenchmarks,
    ScalabilityTests,
    ComparativeAnalysis,
    PerformanceValidator
)

__all__ = [
    'PerformanceProfiler',
    'AlgorithmicBenchmarks', 
    'ScalabilityTests',
    'ComparativeAnalysis',
    'PerformanceValidator'
]