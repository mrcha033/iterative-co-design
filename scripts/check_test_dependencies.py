#!/usr/bin/env python3
"""
Test dependency checker for iterative-co-design.

This script automatically checks if all dependencies required for testing
are available, and provides helpful installation instructions if not.

Usage:
    python scripts/check_test_dependencies.py
    python scripts/check_test_dependencies.py --install  # Auto-install missing deps
"""

import sys
import subprocess
import importlib
from pathlib import Path


def check_import(package_name: str, import_name: str = None) -> tuple[bool, str]:
    """Check if a package can be imported. Returns (success, error_message)."""
    try:
        if import_name:
            importlib.import_module(import_name)
        else:
            importlib.import_module(package_name)
        return True, ""
    except ImportError as e:
        return False, str(e)


def check_dependencies():
    """Check all required test dependencies."""
    print("🔍 Checking test dependencies...")
    print("=" * 50)
    
    # Core dependencies required for testing
    required_packages = [
        ("yaml", "PyYAML", "pyyaml"),
        ("torch", "PyTorch", "torch"),
        ("numpy", "NumPy", "numpy"),
        ("sklearn", "scikit-learn", "scikit-learn"),
        ("transformers", "Transformers", "transformers"),
        ("datasets", "Datasets", "datasets"),
        ("hydra", "Hydra", "hydra-core"),
        ("omegaconf", "OmegaConf", "omegaconf"),
        ("pytest", "pytest", "pytest"),
    ]
    
    missing_packages = []
    working_packages = []
    
    for import_name, display_name, pip_name in required_packages:
        success, error = check_import(import_name)
        if success:
            working_packages.append(display_name)
            print(f"✅ {display_name}: Available")
        else:
            missing_packages.append((display_name, pip_name, error))
            print(f"❌ {display_name}: Missing - {error}")
    
    print(f"\n📊 Summary: {len(working_packages)}/{len(required_packages)} packages available")
    
    if missing_packages:
        print(f"\n⚠️ Missing {len(missing_packages)} required packages:")
        pip_install_cmd = "pip install " + " ".join([pkg[1] for pkg in missing_packages])
        
        print("\n💡 Quick fix - run this command:")
        print(f"   {pip_install_cmd}")
        
        print("\n📋 Or install from requirements files:")
        print("   pip install -r requirements.txt -r tests/requirements.txt")
        
        print("\n🐳 Alternative - use Docker (all deps included):")
        print("   docker-compose run --rm trainer bash")
        
        return False
    else:
        print("\n✅ All test dependencies are available!")
        print("\n🚀 You can now run tests with:")
        print("   pytest tests/")
        print("   bash scripts/run_tests.sh")
        return True


def install_missing_dependencies():
    """Attempt to install missing dependencies automatically."""
    print("🔧 Installing missing test dependencies...")
    
    try:
        # Install from both requirements files
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "-r", "requirements.txt", 
            "-r", "tests/requirements.txt"
        ], check=True)
        
        print("✅ Dependencies installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        print("\n💡 Try manual installation:")
        print("   pip install -r requirements.txt -r tests/requirements.txt")
        return False


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Check test dependencies")
    parser.add_argument(
        "--install", 
        action="store_true", 
        help="Automatically install missing dependencies"
    )
    
    args = parser.parse_args()
    
    print("🧪 Test Dependency Checker")
    print("=" * 50)
    
    # Check current status
    all_available = check_dependencies()
    
    if not all_available and args.install:
        print("\n" + "=" * 50)
        if install_missing_dependencies():
            print("\n" + "=" * 50)
            print("🔍 Re-checking dependencies after installation...")
            check_dependencies()
    
    if not all_available and not args.install:
        print("\n💡 Tip: Use --install flag to auto-install missing packages")
        sys.exit(1)


if __name__ == "__main__":
    main()
