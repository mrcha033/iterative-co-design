# Iterative Co-Design Framework

A comprehensive framework for reproducing and extending the experiments from "The Orthogonality Fallacy: Iterative Co-Design as a First-Class Principle for Efficient AI".

## 🚀 Quick Start

### Docker (Recommended)
```bash
# Build and run with Docker
make build-docker
make run-docker
```

### Local Installation
```bash
# Install dependencies
make install-deps
make install

# Run basic test
python scripts/test_framework.py
```

## 📊 Reproducing Results

### Table 1: Main Results
```bash
make replicate-table-1
```

### Table 2: Causal Mechanism
```bash
make replicate-table-2
```

### Generate Figures
```bash
make generate-figures
```

## 🔧 Configuration

All experiments are configured via `configs/default.yaml`. Key parameters:

- **Model**: Choose from `mamba-3b`, `bert-large`, `resnet-50`, `gcn`
- **Strategy**: Select optimization strategy (baseline, iterative, etc.)
- **Hardware**: Configure GPU settings and profiling options

## 📁 Project Structure

```
iterative-co-design/
├── src/                    # Core framework code
│   ├── co_design/         # IASP and HDS implementations
│   ├── models/            # Model wrappers and utilities
│   └── utils/             # Profiling and benchmarking
├── scripts/               # Experiment scripts
├── tests/                 # Test suite
├── configs/               # Configuration files
├── docs/                  # Documentation
└── results/               # Experiment outputs
```

## 🎯 Key Features

- **IASP**: IO-Aware Scan Permutation for memory layout optimization
- **HDS**: Hardware-Native Differentiable Sparsity
- **Profiling**: Hardware performance analysis with NVIDIA Nsight
- **Reproducibility**: Exact dependency pinning and Docker support

## 📋 Requirements

- **GPU**: NVIDIA GPU with compute capability ≥ 7.0
- **VRAM**: 16GB+ for Mamba-3B experiments
- **CUDA**: Version 11.8 or 12.x
- **Python**: 3.10+

## 🐛 Troubleshooting

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for detailed setup instructions and troubleshooting guide.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Please read our contributing guidelines before submitting pull requests.

## 📞 Support

For issues and questions:
- Check the [troubleshooting guide](REPRODUCIBILITY.md)
- Search existing GitHub issues
- Create a new issue with detailed information