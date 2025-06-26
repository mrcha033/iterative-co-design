#!/usr/bin/env bash
# Install dependencies for running tests and execute pytest
set -e

# Prefer pyproject optional group; fall back to requirements files for legacy use
if [ -f requirements.txt ]; then
  pip install -r requirements.txt -r tests/requirements.txt
else
  # Install package with [test] extras declared in pyproject.toml
  pip install -e .[test]
fi

export PYTHONPATH="$(pwd)"
pytest -q tests "$@"
