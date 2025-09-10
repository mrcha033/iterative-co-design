"""
Ensure project root is on sys.path for test imports.

Pytest sometimes runs with a working directory that doesn't implicitly include
the repository root on sys.path, which can cause `ModuleNotFoundError: icd`.
This small shim makes the package import stable across CI and local runs.
"""

import os
import sys
from pathlib import Path

# tests/ -> repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
repo_str = str(REPO_ROOT)
if repo_str not in sys.path:
    sys.path.insert(0, repo_str)

