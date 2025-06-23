# Tests

This directory contains unit tests for the iterative co-design framework.

## Prerequisites

Before running tests, ensure all dependencies are installed:

### Installing Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt

# Or install with development dependencies
pip install -e ".[test]"

# Or install minimal test dependencies
pip install pytest torch numpy scikit-learn transformers datasets pyyaml hydra-core omegaconf
```

### Checking Your Setup

Use the dependency checker to verify your environment:

```bash
python scripts/check_test_dependencies.py
```

This script will check if all required packages are available and provide installation instructions for any missing dependencies.

### Required Dependencies for Tests
- `pytest` - Test framework
- `torch` - PyTorch for neural network functionality
- `numpy` - Numerical operations
- `scikit-learn` - Machine learning utilities
- `transformers` - Hugging Face transformers
- `datasets` - Hugging Face datasets
- `pyyaml` - YAML configuration loading
- `hydra-core` - Configuration management
- `omegaconf` - Configuration objects

## Running Tests

The tests use `pytest` and are configured to automatically find the source modules via the `pythonpath` setting in `pyproject.toml`.

### Using pytest (recommended)
```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_hds.py -v

# Run with coverage
python -m pytest tests/ --cov=src
```

### Import Setup

Tests import modules using the `src.` prefix (e.g., `from src.co_design.hds import ...`). This works because:

1. **pytest configuration**: The `pyproject.toml` file configures pytest with `pythonpath = ["src"]`
2. **Source layout**: The project uses a `src/` layout with packages under `src/`

If you need to run individual test files outside of pytest, ensure `PYTHONPATH` includes the `src` directory:

```bash
# On Unix/Linux/macOS
export PYTHONPATH=src:$PYTHONPATH
python tests/test_hds.py

# On Windows
set PYTHONPATH=src;%PYTHONPATH%
python tests/test_hds.py
```

## Test Structure

- `test_hds.py` - Tests for Hardware-Native Differentiable Sparsity
- `test_config.py` - Tests for configuration loading utilities  
- `test_wrapper.py` - Tests for model wrapper functionality
- `test_*.py` - Additional test modules

## Adding New Tests

When adding new test files:

1. Use the `test_*.py` naming convention
2. Import source modules with the `src.` prefix
3. Add proper docstrings explaining the test purpose
4. Use appropriate pytest markers for slow/GPU tests if needed

## Troubleshooting

### Common Issues

**"ModuleNotFoundError" when running tests:**
- Ensure dependencies are installed: `pip install -r requirements.txt`
- Check that you're running from the project root directory

**"No module named 'src'" errors:**
- Use pytest instead of running Python files directly
- If running outside pytest, set PYTHONPATH as described above

**Import errors for specific packages:**
- Install missing dependencies individually
- Check `requirements.txt` for the exact versions needed

**Tests fail with tensor/model errors:**
- Ensure PyTorch is properly installed
- For GPU tests, check CUDA availability 