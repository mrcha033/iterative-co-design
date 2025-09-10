**SBOM & Contributing (V1)**

- **Scope:** License policy, vulnerability scanning, coding/review conventions, and ADR template to govern changes. Tailored for a Python project with CI.

**License & Third‑Party Policy**
- **License:** MIT/Apache‑2.0 recommended (choose and add LICENSE file).
- **Dependencies:** Prefer permissive licenses; avoid GPL unless isolated.
- **SBOM:** Generate CycloneDX or SPDX for Python deps via `pip-audit`/`cyclonedx-bom` in CI.

**Vulnerability Scanning**
- Enable `pip-audit` job in CI; fail on known critical CVEs.
- Periodic `safety` checks optional.

**Coding Conventions**
- **Style:** black (88 cols), ruff, isort; add pre‑commit hooks.
- **Typing:** Python 3.10+, type hints in public APIs; mypy (strict in core modules).
- **Testing:** pytest, `-q` in CI; unit/integration/ir suites must pass.
- **Docs:** Keep `docs/` current; when changing behavior, update corresponding spec and tests.

**Review Process**
- **PR Template:** include problem statement, approach, tests added, docs updated, risk & rollout.
- **Required Reviews:** ≥1 reviewer for code, ≥1 for performance‑sensitive changes.
- **Quality Gates:** must pass CI unit/integration/ir; perf gates run in nightly.

**ADR Template**
```
# ADR NNN: Title
Date: YYYY-MM-DD
Status: Proposed|Accepted|Superseded

Context
- Problem, constraints, non-goals

Decision
- Chosen option and rationale

Consequences
- Positive, negative, mitigations

Alternatives Considered
- Briefly list discarded options and why
```

**Self‑Review**
- Practical for this repo’s size; CI workflow already exists and can be extended with lint/audit jobs without impacting tests.

