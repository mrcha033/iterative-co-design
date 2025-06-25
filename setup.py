#!/usr/bin/env python
"""
Setup script for Iterative Co-Design project.

This setup.py provides full package configuration for the iterative-co-design project.
While most configuration is in pyproject.toml, this file ensures compatibility with
older setuptools versions and provides explicit package discovery.
"""

from setuptools import setup, find_packages
from pathlib import Path
import sys

# Read the long description from README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Check Python version
if sys.version_info < (3, 8):
    print("ERROR: Python 3.8 or newer is required")
    sys.exit(1)

# Core dependencies that are always required
INSTALL_REQUIRES = [
    "numpy>=1.21.0",
    "torch>=2.0.0,<=2.3.1",  # Compatible with CUDA 12.1
    "transformers>=4.42.4",  # Unified version with Mamba support
    "datasets>=2.14.0",
    "pyyaml>=6.0",
    "scikit-learn>=1.3.0",
    "pandas>=1.5.0",
    "tqdm>=4.64.0",
    "hydra-core>=1.3.0",
    "omegaconf>=2.3.0",
    "wandb>=0.15.0",
    "matplotlib>=3.5.0",
    "seaborn>=0.11.0",
]

# Optional dependencies
EXTRAS_REQUIRE = {
    "test": [
        "pytest>=8.0.2",
        "pytest-cov",
        # Core dependencies needed for testing
        "torch>=2.0.0",
        "numpy>=1.21.0,<2.0.0",
        "scikit-learn>=1.3.0",
        "transformers>=4.42.4",
        "datasets>=2.14.0",
        "pyyaml>=6.0",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
    ],
    "docs": [
        "mkdocs>=1.5.3",
        "mkdocs-material>=9.5.13",
        "mkdocstrings[python]>=0.24.1",
    ],
    "dev": [
        "ruff>=0.12.0",
    ],
}

# Combine all extras for convenience
EXTRAS_REQUIRE["all"] = list(set(sum(EXTRAS_REQUIRE.values(), [])))
EXTRAS_REQUIRE["dev"] = EXTRAS_REQUIRE["dev"] + EXTRAS_REQUIRE["test"] + EXTRAS_REQUIRE["docs"]

setup(
    name="iterative-co-design",
    version="0.1.0",
    author="Yunmin Cha",
    author_email="mrcha033@yonsei.ac.kr",
    description="Iterative Co-Design of Sparsity and Permutation Structures for Hardware-Accelerated Inference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mrcha033/iterative-co-design",
    project_urls={
        "Bug Reports": "https://github.com/mrcha033/iterative-co-design/issues",
        "Source": "https://github.com/mrcha033/iterative-co-design",
        "Documentation": "https://mrcha033.github.io/iterative-co-design",
    },
    
    # Package discovery
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    # Include additional files
    package_data={
        "": ["*.yaml", "*.yml"],  # Include YAML configuration files
    },
    include_package_data=True,
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Dependencies
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    
    # Entry points (if any command-line scripts are needed)
    entry_points={
        "console_scripts": [
            # Example: "iterative-co-design=co_design.cli:main",
        ],
    },
    
    # PyPI classifiers
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    
    # Keywords for PyPI search
    keywords="machine-learning deep-learning sparsity hardware-acceleration optimization",
    
    # License
    license="MIT",
    
    # Prevent zip_safe to ensure proper module loading
    zip_safe=False,
) 