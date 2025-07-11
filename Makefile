# Makefile for iterative-co-design project

# Variables
DOCKER_IMAGE = iterative-co-design
DOCKER_TAG = latest
PYTHON = python3
PIP = pip3

# Help target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  build-docker      Build Docker image"
	@echo "  run-docker        Run Docker container interactively"
	@echo "  test              Run all tests"
	@echo "  test-unit         Run unit tests only"
	@echo "  test-integration  Run integration tests only"
	@echo "  format            Format code with black and isort"
	@echo "  lint              Run linting with ruff and mypy"
	@echo "  install           Install package in development mode"
	@echo "  install-deps      Install dependencies"
	@echo "  clean             Clean build artifacts"
	@echo "  replicate-table-1 Replicate Table 1 from paper"
	@echo "  replicate-table-2 Replicate Table 2 from paper"
	@echo "  generate-figures  Generate all figures from paper"

# Docker targets
.PHONY: build-docker
build-docker:
	@echo "Building Docker image..."
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

.PHONY: run-docker
run-docker:
	@echo "Running Docker container..."
	docker run -it --rm --gpus all \
		-v $(PWD):/workspace \
		-v $(PWD)/data:/workspace/data \
		-v $(PWD)/results:/workspace/results \
		$(DOCKER_IMAGE):$(DOCKER_TAG)

# Development targets
.PHONY: install
install:
	@echo "Installing package in development mode..."
	$(PIP) install -e .

.PHONY: install-deps
install-deps:
	@echo "Installing dependencies..."
	$(PIP) install -r requirements.txt

# Testing targets
.PHONY: test
test:
	@echo "Running all tests..."
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

.PHONY: test-unit
test-unit:
	@echo "Running unit tests..."
	pytest tests/unit/ -v

.PHONY: test-integration
test-integration:
	@echo "Running integration tests..."
	pytest tests/integration/ -v

# Code quality targets
.PHONY: format
format:
	@echo "Formatting code..."
	black src/ tests/ scripts/
	isort src/ tests/ scripts/

.PHONY: lint
lint:
	@echo "Running linting..."
	ruff check src/ tests/ scripts/
	mypy src/ --ignore-missing-imports

# Replication targets
.PHONY: replicate-table-1
replicate-table-1:
	@echo "Replicating Table 1 from paper..."
	$(PYTHON) scripts/replicate_table_1.py

.PHONY: replicate-table-2
replicate-table-2:
	@echo "Replicating Table 2 from paper..."
	$(PYTHON) scripts/replicate_table_2.py

.PHONY: generate-figures
generate-figures:
	@echo "Generating figures..."
	$(PYTHON) scripts/generate_figures.py

# Cleanup targets
.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/

# Environment setup targets
.PHONY: setup-conda
setup-conda:
	@echo "Setting up conda environment..."
	conda env create -f environment.yml

.PHONY: setup-venv
setup-venv:
	@echo "Setting up virtual environment..."
	python3 -m venv venv
	./venv/bin/pip install -r requirements.txt
	./venv/bin/pip install -e .

# Default target
.DEFAULT_GOAL := help