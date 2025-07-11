from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="iterative-co-design",
    version="1.0.0",
    author="Research Team",
    author_email="research@example.com",
    description="Iterative Co-Design Framework for Efficient AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/research-team/iterative-co-design",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.2",
        "torchvision>=0.16.2",
        "scikit-learn>=1.5.0",
        "numpy>=1.24.3",
        "pandas>=2.0.3",
        "PyYAML>=6.0.1",
        "hydra-core>=1.3.2",
        "omegaconf>=2.3.0",
        "datasets>=2.14.5",
        "transformers>=4.35.0",
        "torch-geometric>=2.4.0",
        "matplotlib>=3.7.2",
        "tqdm>=4.66.1",
        "click>=8.1.7",
        "rich>=13.6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.2",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.11.1",
            "black>=23.9.1",
            "ruff>=0.0.292",
            "isort>=5.12.0",
            "mypy>=1.5.1",
        ],
        "profiling": [
            "nvidia-ml-py>=12.535.108",
            "psutil>=5.9.6",
        ],
    },
    entry_points={
        "console_scripts": [
            "iterative-co-design=src.cli:main",
        ],
    },
)