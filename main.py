#!/usr/bin/env python3
"""
Main CLI entry point for the iterative co-design framework.

This script provides a command-line interface to run experiments with different
optimization strategies (baseline, sparsity, permutation, quantization, iterative).
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any

from scripts.run_experiment import ExperimentRunner
from src.utils.config import load_config, merge_config_with_args
from src.utils.exceptions import IterativeCoDesignError


def setup_logging(level: str = "INFO", rich_formatting: bool = True) -> None:
    """Setup logging configuration."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatter
    if rich_formatting:
        try:
            from rich.logging import RichHandler
            handler = RichHandler(rich_tracebacks=True)
            logging.basicConfig(
                level=log_level,
                format="%(message)s",
                handlers=[handler]
            )
        except ImportError:
            # Fallback to standard logging
            logging.basicConfig(
                level=log_level,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
    else:
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Iterative Co-Design Framework for Efficient AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run baseline experiment
  python main.py --strategy baseline --model mamba-3b
  
  # Run iterative sparsity experiment
  python main.py --strategy iterative_sparsity --model bert-large --num-iterations 2
  
  # Run quantization experiment with custom config
  python main.py --config configs/quantization.yaml --strategy iterative_quant
  
  # Enable profiling
  python main.py --strategy iterative_sparsity --profile
        """
    )
    
    # Configuration
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/default.yaml',
        help='Path to configuration file (default: configs/default.yaml)'
    )
    
    # Experiment strategy
    parser.add_argument(
        '--strategy', '-s',
        type=str,
        choices=[
            'baseline',
            'permute_only', 
            'sparsity_only',
            'linear_sparsity',
            'iterative_sparsity',
            'linear_quant_permute_first',
            'linear_quant_quant_first', 
            'iterative_quant'
        ],
        help='Optimization strategy to run'
    )
    
    # Model configuration
    parser.add_argument(
        '--model', '-m',
        type=str,
        choices=['mamba-3b', 'bert-large', 'resnet-50', 'gcn'],
        help='Model architecture to use'
    )
    
    # Dataset configuration  
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        choices=['wikitext-103', 'imagenet', 'ogbn-arxiv'],
        help='Dataset to use for correlation computation and benchmarking'
    )
    
    # Experiment parameters
    parser.add_argument(
        '--num-iterations', '-n',
        type=int,
        help='Number of co-design iterations (default: 1)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducibility (default: 42)'
    )
    
    # IASP parameters
    parser.add_argument(
        '--layer-name',
        type=str,
        help='Target layer for permutation (e.g., layers.0.mixer)'
    )
    
    parser.add_argument(
        '--num-clusters',
        type=int,
        help='Number of clusters for modularity-based permutation (default: 64)'
    )
    
    # HDS parameters
    parser.add_argument(
        '--sparsity-pattern',
        type=str,
        help='Sparsity pattern for HDS (e.g., 2:4, 4:8, 1:2)'
    )
    
    parser.add_argument(
        '--hds-epochs',
        type=int,
        help='Number of fine-tuning epochs for HDS (default: 5)'
    )
    
    # PTQ parameters
    parser.add_argument(
        '--quantization-bits',
        type=int,
        choices=[4, 8],
        help='Number of bits for quantization (default: 8)'
    )
    
    # Output and logging
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        help='Output directory for results (default: ./results/)'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    
    # Profiling and benchmarking
    parser.add_argument(
        '--profile', '-p',
        action='store_true',
        help='Enable hardware profiling with nsight-compute'
    )
    
    parser.add_argument(
        '--benchmark-runs',
        type=int,
        help='Number of benchmark runs (default: 5)'
    )
    
    # Hardware configuration
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        help='Device to use (default: cuda)'
    )
    
    parser.add_argument(
        '--gpu-id',
        type=int,
        help='GPU device ID if multiple GPUs available (default: 0)'
    )
    
    # Misc options
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print configuration and exit without running experiment'
    )
    
    parser.add_argument(
        '--save-intermediate',
        action='store_true',
        help='Save intermediate results during experiment'
    )
    
    parser.add_argument(
        '--precomputed-correlation',
        type=str,
        help='Path to precomputed correlation matrix file'
    )
    
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate command line arguments."""
    # Check if config file exists
    if not Path(args.config).exists():
        raise FileNotFoundError(f"Configuration file not found: {args.config}")
    
    # Strategy-specific validations
    if args.strategy in ['linear_sparsity', 'iterative_sparsity'] and args.sparsity_pattern:
        if not args.sparsity_pattern.count(':') == 1:
            raise ValueError(f"Invalid sparsity pattern: {args.sparsity_pattern}. Expected format: 'N:M'")
    
    # Model-dataset compatibility
    model_dataset_mapping = {
        'mamba-3b': ['wikitext-103'],
        'bert-large': ['wikitext-103'],
        'resnet-50': ['imagenet'],
        'gcn': ['ogbn-arxiv']
    }
    
    if args.model and args.dataset:
        if args.dataset not in model_dataset_mapping.get(args.model, []):
            raise ValueError(f"Model {args.model} is not compatible with dataset {args.dataset}")


def main() -> int:
    """Main entry point."""
    try:
        # Parse arguments
        args = parse_args()
        
        # Load base configuration
        config = load_config(args.config)
        
        # Merge with command line arguments
        config = merge_config_with_args(config, args)
        
        # Setup logging
        setup_logging(
            level=config.get('logging', {}).get('level', 'INFO'),
            rich_formatting=config.get('logging', {}).get('rich_formatting', True)
        )
        
        logger = logging.getLogger(__name__)
        logger.info("Starting Iterative Co-Design Framework")
        logger.info(f"Configuration loaded from: {args.config}")
        logger.info(f"Strategy: {config['experiment']['strategy']}")
        logger.info(f"Model: {config['model']['name']}")
        logger.info(f"Dataset: {config['dataset']['name']}")
        
        # Validate arguments
        validate_arguments(args)
        
        # Print configuration if dry run
        if args.dry_run:
            logger.info("Dry run mode - printing configuration and exiting")
            print("\n" + "="*60)
            print("EXPERIMENT CONFIGURATION")
            print("="*60)
            for section, values in config.items():
                print(f"\n[{section.upper()}]")
                if isinstance(values, dict):
                    for key, value in values.items():
                        print(f"  {key}: {value}")
                else:
                    print(f"  {values}")
            print("="*60)
            return 0
        
        # Initialize and run experiment
        runner = ExperimentRunner(config)
        results = runner.run()
        
        logger.info("Experiment completed successfully")
        logger.info(f"Results saved to: {results.get('output_dir', 'Unknown')}")
        
        # Print summary statistics
        if 'benchmark_results' in results:
            benchmark = results['benchmark_results']
            logger.info(f"Final latency: {benchmark.get('mean_latency_ms', 'N/A'):.2f} ± {benchmark.get('std_latency_ms', 'N/A'):.2f} ms")
            if 'improvement_pct' in benchmark:
                logger.info(f"Improvement over baseline: {benchmark['improvement_pct']:.1f}%")
        
        return 0
        
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
        return 130
    except IterativeCoDesignError as e:
        logging.error(f"Framework error: {e}")
        return 1
    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())