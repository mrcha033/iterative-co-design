#!/usr/bin/env python3
"""
Check if all required dependencies for running tests are available.
Run this script to verify your environment is set up correctly for testing.
"""

import sys
from importlib import import_module

# Required packages for tests
REQUIRED_PACKAGES = [
    ("pytest", "pytest"),
    ("torch", "torch"),
    ("numpy", "numpy"),
    ("sklearn", "scikit-learn"),
    ("transformers", "transformers"),
    ("datasets", "datasets"),
    ("yaml", "pyyaml"),
    ("hydra", "hydra-core"),
    ("omegaconf", "omegaconf"),
]


def check_package(import_name, package_name):
    """Check if a package can be imported."""
    try:
        import_module(import_name)
        return True, None
    except ImportError as e:
        return False, str(e)


def main():
    """Check all required packages."""
    print("🔍 Checking test dependencies...")
    print("=" * 50)

    missing_packages = []

    for import_name, package_name in REQUIRED_PACKAGES:
        success, error = check_package(import_name, package_name)

        if success:
            print(f"✅ {package_name}")
        else:
            print(f"❌ {package_name} - {error}")
            missing_packages.append(package_name)

    print("=" * 50)

    if not missing_packages:
        print("🎉 All test dependencies are available!")
        print("\nYou can now run tests with:")
        print("  python -m pytest tests/")
        return 0
    else:
        print(f"❌ {len(missing_packages)} package(s) missing:")
        for pkg in missing_packages:
            print(f"  - {pkg}")

        print("\nTo install missing packages:")
        print("  pip install -r requirements.txt")
        print("  # or install individually:")
        print(f"  pip install {' '.join(missing_packages)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
