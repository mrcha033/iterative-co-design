# Iterative Co-Design Documentation

Welcome to the documentation for the **Iterative Co-Design** framework - a novel approach for joint optimization of algorithmic and hardware-level efficiencies in neural networks.

## Overview

This framework challenges the 'Orthogonality Fallacy' - the assumption that algorithmic optimizations (like sparsity) and hardware-level optimizations (like memory layout) are separable concerns. Our iterative approach demonstrates significant performance gains by creating a feedback loop between these two domains.

## Key Components

- **Hardware-Native Differentiable Sparsity (HDS)**: Learns structured sparsity patterns that are hardware-friendly
- **IO-Aware Scan Permutation (IASP)**: Optimizes memory layout for improved cache locality
- **Modularity Metric**: Quantifies the quality of memory layout clustering

## Quick Start

```bash
# Install the package
pip install -e .[test]

# Run experiments
python scripts/run_experiment.py model=mamba_370m dataset=wikitext103 method=iterative

# Generate figures
python scripts/generate_all_figures.py
```

## API Reference

- [Co-Design Algorithms](api/co_design.md)
- [Model Utilities](api/models.md) 
- [Utility Functions](api/utils.md)

## Results

Our framework achieves:
- **17.8%** latency reduction over linear pipeline approaches
- **89.5%** L2 cache hit rate (vs 78.2% for baselines)
- State-of-the-art efficiency on the Pareto frontier

## Citation

If you use this framework in your research, please cite our paper:

```bibtex
@inproceedings{cha2025orthogonality,
  title={The Orthogonality Fallacy: Iterative Co-Design as a First-Class Principle for Efficient AI},
  author={Cha, Yunmin},
  booktitle={Conference on Machine Learning and Systems (MLSys)},
  year={2025}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](../LICENSE) file for details. 