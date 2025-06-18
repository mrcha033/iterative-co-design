#!/usr/bin/env bash
# Install dependencies for running tests and execute pytest
set -e
pip install -r requirements.txt -r tests/requirements.txt
export PYTHONPATH="$(pwd)"
pytest -q tests "$@"
