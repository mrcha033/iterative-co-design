# Contributing to ICD

Thanks for your interest in improving Iterative HW–SW Co-Design (ICD)!

## Getting Started

- Python 3.10+
- Recommended: create a virtual environment

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
# No extra deps required to run tests
```

## Development Workflow

- Run tests locally

```bash
pytest -q tests/unit
pytest -q tests/integration
pytest -q tests/ir
```

- Lint (non-blocking in CI, but please keep clean)

```bash
ruff check .
black --check .
```

- Useful CLI flags

```bash
# Validate config without running
python -m icd.cli.main run -c configs/mock.json --out /tmp/ignore --dry-run
# Print config schema
python -m icd.cli.main run --print-schema -c configs/mock.json --out /tmp/ignore
# Solver only
python -m icd.cli.main run -c configs/mock.json --no-measure --out runs/solver_only
```

## Commit & PR Guidelines

- Keep changes focused and tested. Update docs when behavior changes.
- Reference the spec files when implementing new features (ICD/PRD/SAS/SOP).
- Include a short PR description with:
  - Problem statement
  - Approach
  - Tests added/updated
  - Docs updated
  - Risks/rollout plan

## Filing Issues

- Provide reproducible steps, expected vs actual behavior, and environment details.
- For performance/measurement issues, attach `metrics.json`, `run.log`, and relevant artifacts if possible.

## Code of Conduct

Be respectful and constructive. We welcome diverse perspectives and encourage thoughtful discussion.

