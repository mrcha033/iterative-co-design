#!/usr/bin/env python3
"""
Script to generate correlation matrices for IASP optimization.
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.manager import ModelManager
from src.utils.dataset_manager import DatasetManager
from src.utils.config import load_config, create_default_config
from src.co_design.correlation import CorrelationMatrixComputer
from src.utils.exceptions import create_helpful_error_message


def main():
    """Main function for correlation matrix generation."""
    parser = argparse.ArgumentParser(
        description='Generate correlation matrices for IASP optimization'
    )
    
    # Required arguments
    parser.add_argument('--model', required=True, help='Model name (e.g., mamba-3b)')
    parser.add_argument('--layer', required=True, help='Layer name (e.g., layers.0.mixer)')
    parser.add_argument('--dataset', required=True, help='Dataset name (e.g., wikitext-103)')
    
    # Optional arguments
    parser.add_argument('--num-samples', type=int, default=1000, help='Number of samples')
    parser.add_argument('--sequence-length', type=int, default=4096, help='Sequence length')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--device', default='cuda', help='Device to use')
    parser.add_argument('--output-dir', default='./data/correlation_matrices', help='Output directory')
    parser.add_argument('--config', help='Path to config file')
    parser.add_argument('--force', action='store_true', help='Force recomputation')
    parser.add_argument('--check-memory', action='store_true', default=True, help='Check memory usage')
    parser.add_argument('--cache-info', action='store_true', help='Show cache information')
    parser.add_argument('--clear-cache', action='store_true', help='Clear correlation cache')
    parser.add_argument('--validate-only', action='store_true', help='Only validate existing matrices')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        if args.config:
            config = load_config(args.config)
        else:
            config = create_default_config()
            # Override with command line arguments
            config.model.name = args.model
            config.dataset.name = args.dataset
            config.dataset.sequence_length = args.sequence_length
            config.dataset.batch_size = args.batch_size
            config.dataset.num_samples = args.num_samples
            config.iasp.layer_name = args.layer
            config.hardware.device = args.device
        
        # Initialize components
        model_manager = ModelManager()
        dataset_manager = DatasetManager(data_dir='./data')
        correlation_computer = CorrelationMatrixComputer(
            cache_dir=args.output_dir,
            device=args.device
        )
        
        # Handle cache operations
        if args.cache_info:
            cache_info = correlation_computer.get_cache_info()
            print("Correlation Matrix Cache Information:")
            print(f"  Cache directory: {cache_info['cache_dir']}")
            print(f"  Cached matrices: {cache_info['cached_matrices']}")
            print(f"  Total size: {cache_info['total_size_mb']:.2f} MB")
            if cache_info['files']:
                print("  Files:")
                for filename in cache_info['files']:
                    print(f"    {filename}")
            return 0
        
        if args.clear_cache:
            correlation_computer.clear_cache()
            print("Correlation matrix cache cleared.")
            return 0
        
        # Validate configuration
        if not model_manager.is_model_supported(args.model):
            supported_models = model_manager.list_supported_models()
            print(f"Error: Model '{args.model}' not supported.")
            print(f"Supported models: {', '.join(supported_models)}")
            return 1
        
        if not dataset_manager.is_dataset_supported(args.dataset):
            supported_datasets = dataset_manager.list_supported_datasets()
            print(f"Error: Dataset '{args.dataset}' not supported.")
            print(f"Supported datasets: {', '.join(supported_datasets)}")
            return 1
        
        # Load model
        print(f"Loading model '{args.model}'...")
        model = model_manager.load_model(
            args.model,
            device=args.device,
            precision=config.model.precision
        )
        
        # Validate layer name
        if not model_manager.validate_layer_name(args.model, args.layer):
            available_layers = model.get_layer_names()
            print(f"Error: Layer '{args.layer}' not found in model '{args.model}'.")
            print(f"Available layers: {', '.join(available_layers[:10])}...")
            return 1
        
        # Load dataset
        print(f"Loading dataset '{args.dataset}'...")
        dataloader = dataset_manager.load_dataset(
            args.dataset,
            split='train',
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            num_samples=args.num_samples,
            shuffle=False
        )
        
        if args.validate_only:
            # Only validate existing matrices
            cache_info = correlation_computer.get_cache_info()
            if cache_info['cached_matrices'] == 0:
                print("No cached correlation matrices found.")
                return 1
            
            print(f"Validating {cache_info['cached_matrices']} cached matrices...")
            valid_count = 0
            
            for filename in cache_info['files']:
                matrix_path = Path(args.output_dir) / filename
                try:
                    matrix = correlation_computer.load_precomputed_matrix(str(matrix_path))
                    if correlation_computer.validate_correlation_matrix(matrix):
                        valid_count += 1
                        print(f"  ✅ {filename}: Valid ({matrix.shape[0]}x{matrix.shape[1]})")
                    else:
                        print(f"  ❌ {filename}: Invalid matrix")
                except Exception as e:
                    print(f"  ❌ {filename}: Error loading - {e}")
            
            print(f"Validation complete: {valid_count}/{cache_info['cached_matrices']} matrices valid")
            return 0 if valid_count == cache_info['cached_matrices'] else 1
        
        # Generate correlation matrix
        print(f"Generating correlation matrix for layer '{args.layer}'...")
        correlation_matrix = correlation_computer.compute_correlation_matrix(
            model=model,
            dataloader=dataloader,
            layer_name=args.layer,
            num_samples=args.num_samples,
            force_recompute=args.force,
            check_memory=args.check_memory
        )
        
        # Validate the generated matrix
        if not correlation_computer.validate_correlation_matrix(correlation_matrix):
            print("Warning: Generated correlation matrix failed validation!")
            return 1
        
        print(f"✅ Correlation matrix generated successfully!")
        print(f"   Shape: {correlation_matrix.shape}")
        print(f"   Range: [{correlation_matrix.min():.4f}, {correlation_matrix.max():.4f}]")
        print(f"   Cached in: {args.output_dir}")
        
        # Show cache information
        cache_info = correlation_computer.get_cache_info()
        print(f"   Cache size: {cache_info['total_size_mb']:.2f} MB")
        print(f"   Total matrices: {cache_info['cached_matrices']}")
        
        return 0
        
    except Exception as e:
        error_msg = create_helpful_error_message(
            e, "Correlation matrix generation failed"
        )
        print(f"\n{error_msg}")
        return 1


if __name__ == '__main__':
    sys.exit(main())