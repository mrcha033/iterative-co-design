#!/usr/bin/env python3
"""
Setup script for iterative-co-design project.
Provides convenient installation with CPU or GPU PyTorch options.
"""

import argparse
import subprocess
import sys
import platform
from pathlib import Path


def run_command(cmd, description=""):
    """Run a command and handle errors."""
    print(f"Running: {description or cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False


def install_pytorch(device_type="cpu"):
    """Install PyTorch with CPU or GPU support."""
    if device_type == "cpu":
        # Use more compatible PyTorch version for broader Python support
        cmd = "pip install 'torch>=2.0.0,<2.5.0' 'torchvision>=0.15.0' 'torchaudio>=2.0.0' --index-url https://download.pytorch.org/whl/cpu"
        desc = "Installing PyTorch (CPU-only, compatible version)"
    elif device_type == "cuda":
        cmd = "pip install 'torch>=2.0.0' 'torchvision>=0.15.0' 'torchaudio>=2.0.0'"
        desc = "Installing PyTorch (with CUDA support)"
    else:
        print(f"Unknown device type: {device_type}")
        return False
    
    return run_command(cmd, desc)


def main():
    parser = argparse.ArgumentParser(
        description="Setup script for iterative-co-design project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/setup.py --device cpu      # Install with CPU-only PyTorch (faster)
  python scripts/setup.py --device cuda     # Install with CUDA PyTorch
  python scripts/setup.py --dev             # Install development dependencies
  python scripts/setup.py --test            # Run tests after installation
        """
    )
    
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="PyTorch device support (default: cpu)"
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Install development dependencies (ruff, docs)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run tests after installation"
    )
    parser.add_argument(
        "--skip-pytorch",
        action="store_true",
        help="Skip PyTorch installation (useful if already installed)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Iterative Co-Design Setup Script")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        print("Error: pyproject.toml not found. Please run this script from the project root.")
        return 1
    
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"PyTorch device: {args.device}")
    
    # Check for common issues
    python_version = sys.version_info
    if python_version < (3, 8):
        print("WARNING: Python 3.8+ is recommended for best compatibility")
    
    print()
    
    success = True
    
    # Step 1: Upgrade pip and setuptools
    if not run_command("python -m pip install --upgrade pip setuptools wheel", "Upgrading pip and build tools"):
        success = False
    
    # Step 2: Install PyTorch
    if not args.skip_pytorch:
        if not install_pytorch(args.device):
            success = False
    else:
        print("Skipping PyTorch installation")
    
    # Step 3: Install main dependencies
    if not run_command("pip install -r requirements.txt", "Installing main dependencies"):
        print("Failed to install from requirements.txt. Trying with --no-deps to diagnose...")
        print("If this fails, you may need to:")
        print("1. Update your Python version (3.9+ recommended)")
        print("2. Update pip: python -m pip install --upgrade pip")
        print("3. Try installing individual packages")
        success = False
    
    # Step 4: Install test dependencies
    if not run_command("pip install -r tests/requirements.txt", "Installing test dependencies"):
        success = False
    
    # Step 5: Install package in editable mode
    if not run_command("pip install -e .", "Installing package in editable mode"):
        success = False
    
    # Step 6: Install development dependencies
    if args.dev:
        if not run_command("pip install -e .[dev]", "Installing development dependencies"):
            success = False
    
    if success:
        print("\n" + "=" * 60)
        print("Installation completed successfully!")
        print("=" * 60)
        
        # Verify installation
        print("\nVerifying installation...")
        verification_commands = [
            ('python -c "import torch; print(f\'PyTorch version: {torch.__version__}\')"', "PyTorch"),
            ('python -c "import numpy; print(f\'NumPy version: {numpy.__version__}\')"', "NumPy"),
            ('python -c "import yaml; print(\'PyYAML available\')"', "PyYAML"),
            ('python -c "from utils.config import load_yaml_config; print(\'Config import successful\')"', "Config module"),
            ('python -c "from co_design.modularity import calculate_modularity; print(\'Modularity module successful\')"', "Modularity module"),
            ('python -c "from models.wrapper import ModelWrapper; print(\'Model wrapper successful\')"', "Model wrapper"),
        ]
        
        for cmd, desc in verification_commands:
            if not run_command(cmd, f"Verifying {desc}"):
                success = False
        
        if success:
            print("\nAll components verified successfully!")
            
            # Step 7: Run tests if requested
            if args.test:
                print("\nRunning tests...")
                if run_command("pytest tests/ -v", "Running test suite"):
                    print("All tests passed!")
                else:
                    print("Some tests failed")
                    success = False
        else:
            print("\nVerification failed")
            return 1
    else:
        print("\n" + "=" * 60)
        print("Installation failed!")
        print("=" * 60)
        return 1
    
    print("\nNext steps:")
    print("1. Run experiments: python scripts/run_experiment.py model=mamba_3b dataset=wikitext103 method=dense")
    print("2. Run tests: pytest tests/")
    print("3. Generate figures: python scripts/generate_all_figures.py")
    print("\nFor more information, see README.md")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 