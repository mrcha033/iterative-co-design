#!/usr/bin/env python3
"""
Quick start demonstration of the experiment pipeline.

This script shows how to run different experiment strategies without
requiring full model downloads or CUDA setup.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from main import parse_args, validate_arguments
from src.utils.config import load_config, merge_config_with_args


def create_demo_config():
    """Create a minimal demo configuration."""
    return {
        'model': {
            'name': 'mamba-3b',
            'hf_model_id': 'state-spaces/mamba-3b',
            'pretrained_path': None,
            'precision': 'float16'
        },
        'dataset': {
            'name': 'wikitext-103',
            'path': './data/',
            'sequence_length': 512,
            'batch_size': 1,
            'num_samples': 10,
            'num_benchmark_samples': 5
        },
        'iasp': {
            'layer_name': 'layers.0.mixer',
            'num_clusters': 8,
            'correlation_threshold': 0.1,
            'method': 'spectral',
            'precomputed_path': './data/correlation_matrices/'
        },
        'hds': {
            'pattern': '2:4',
            'learning_rate': 1e-5,
            'num_epochs': 2,
            'gumbel_temperature': 1.0,
            'sparsity_ratio': 0.5
        },
        'ptq': {
            'bits': 8,
            'scheme': 'symmetric',
            'calibration_samples': 10
        },
        'experiment': {
            'strategy': 'baseline',
            'num_iterations': 1,
            'output_dir': './demo_results/',
            'seed': 42,
            'save_intermediate': True
        },
        'benchmark': {
            'warmup_runs': 3,
            'num_runs': 5,
            'use_cuda_events': False,
            'cuda_sync': True
        },
        'profiling': {
            'enabled': False,
            'tool': 'pytorch_profiler',
            'metrics': ['lts__t_sector_hit_rate.pct']
        },
        'hardware': {
            'device': 'cpu',
            'gpu_id': 0,
            'mixed_precision': False
        },
        'logging': {
            'level': 'INFO',
            'file': './demo_results/demo.log',
            'console': True,
            'rich_formatting': True
        },
        'reproducibility': {
            'deterministic': True,
            'cuda_deterministic': False,
            'warn_non_deterministic': True
        }
    }


def demo_cli_parsing():
    """Demonstrate CLI argument parsing."""
    print("🔍 CLI Argument Parsing Demo")
    print("-" * 40)
    
    # Simulate different CLI commands
    test_commands = [
        ['--strategy', 'baseline', '--model', 'mamba-3b'],
        ['--strategy', 'iterative_sparsity', '--model', 'bert-large', '--num-iterations', '2'],
        ['--strategy', 'iterative_quant', '--quantization-bits', '8', '--profile'],
        ['--config', 'configs/default.yaml', '--strategy', 'linear_sparsity', '--dry-run']
    ]
    
    import sys
    original_argv = sys.argv
    
    try:
        for i, cmd in enumerate(test_commands, 1):
            print(f"\nTest {i}: python main.py {' '.join(cmd)}")
            
            sys.argv = ['main.py'] + cmd
            try:
                args = parse_args()
                print(f"   ✓ Strategy: {args.strategy}")
                print(f"   ✓ Model: {args.model}")
                if hasattr(args, 'num_iterations') and args.num_iterations:
                    print(f"   ✓ Iterations: {args.num_iterations}")
                if hasattr(args, 'profile') and args.profile:
                    print(f"   ✓ Profiling: enabled")
                if hasattr(args, 'dry_run') and args.dry_run:
                    print(f"   ✓ Dry run: enabled")
            except SystemExit:
                # argparse calls sys.exit on error
                print(f"   ✗ Failed to parse arguments")
    
    finally:
        sys.argv = original_argv


def demo_config_system():
    """Demonstrate configuration system."""
    print("\n⚙️  Configuration System Demo")
    print("-" * 40)
    
    # Create demo config
    demo_config = create_demo_config()
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        import yaml
        yaml.dump(demo_config, f)
        config_path = f.name
    
    try:
        # Load config
        loaded_config = load_config(config_path)
        print(f"✓ Loaded config with {len(loaded_config)} sections")
        
        # Mock CLI args
        class MockArgs:
            strategy = 'iterative_sparsity'
            model = 'bert-large'
            num_iterations = 3
            seed = 999
            profile = True
            
            def __getattr__(self, name):
                return None
        
        # Merge with CLI args
        merged_config = merge_config_with_args(loaded_config, MockArgs())
        
        print(f"✓ Merged CLI args:")
        print(f"   - Strategy: {merged_config['experiment']['strategy']}")
        print(f"   - Model: {merged_config['model']['name']}")
        print(f"   - Iterations: {merged_config['experiment']['num_iterations']}")
        print(f"   - Seed: {merged_config['experiment']['seed']}")
        print(f"   - Profiling: {merged_config['profiling']['enabled']}")
        
    finally:
        # Cleanup
        Path(config_path).unlink()


def demo_strategy_overview():
    """Show overview of all supported strategies."""
    print("\n🚀 Supported Experiment Strategies")
    print("-" * 40)
    
    strategies = {
        'baseline': 'No optimization - measure baseline performance',
        'permute_only': 'Apply IASP permutation only',
        'sparsity_only': 'Apply HDS sparsity only',
        'linear_sparsity': 'Linear pipeline: HDS → IASP',
        'iterative_sparsity': 'Iterative pipeline: IASP → HDS → IASP',
        'linear_quant_permute_first': 'Linear quantization: IASP → PTQ',
        'linear_quant_quant_first': 'Linear quantization: PTQ → IASP',
        'iterative_quant': 'Iterative quantization: IASP → PTQ → IASP'
    }
    
    for strategy, description in strategies.items():
        print(f"  {strategy:25} - {description}")


def demo_expected_outputs():
    """Show expected output structure."""
    print("\n📁 Expected Output Structure")
    print("-" * 40)
    
    output_structure = [
        'results/',
        '├── <strategy>_<model>_<timestamp>/',
        '│   ├── config.json              # Experiment configuration',
        '│   ├── results.json             # Complete experiment results',  
        '│   ├── summary.txt              # Human-readable summary',
        '│   ├── permutation_*.pt         # Saved permutations (if enabled)',
        '│   ├── ncu_profile.csv          # Hardware profiling (if enabled)',
        '│   └── pytorch_trace.json       # PyTorch profiling (if enabled)',
        '',
        'Key Result Fields:',
        '  - baseline_benchmark: Initial performance metrics',
        '  - final_benchmark: Final performance metrics',
        '  - improvements: Performance improvement statistics',
        '  - strategy_results: Strategy-specific results',
        '  - reproducibility_info: Environment and seed info'
    ]
    
    for line in output_structure:
        print(f"  {line}")


def demo_usage_examples():
    """Show practical usage examples."""
    print("\n💡 Usage Examples")
    print("-" * 40)
    
    examples = [
        {
            'title': 'Run baseline experiment',
            'command': 'python main.py --strategy baseline --model mamba-3b'
        },
        {
            'title': 'Run iterative sparsity with 2 iterations',
            'command': 'python main.py --strategy iterative_sparsity --model bert-large --num-iterations 2'
        },
        {
            'title': 'Run quantization experiment with profiling',
            'command': 'python main.py --strategy iterative_quant --model mamba-3b --profile'
        },
        {
            'title': 'Use custom config file',
            'command': 'python main.py --config configs/my_config.yaml --strategy linear_sparsity'
        },
        {
            'title': 'Dry run to see configuration',
            'command': 'python main.py --strategy iterative_sparsity --model mamba-3b --dry-run'
        }
    ]
    
    for example in examples:
        print(f"\n  {example['title']}:")
        print(f"    {example['command']}")


def main():
    """Main demo function."""
    print("🎯 Iterative Co-Design Framework - Quick Start Demo")
    print("=" * 60)
    
    try:
        demo_cli_parsing()
        demo_config_system()
        demo_strategy_overview()
        demo_expected_outputs()
        demo_usage_examples()
        
        print("\n" + "=" * 60)
        print("✅ Demo completed successfully!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Configure your model and dataset paths")
        print("3. Run an experiment: python main.py --strategy baseline --model mamba-3b")
        print("4. Check results in the generated output directory")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())