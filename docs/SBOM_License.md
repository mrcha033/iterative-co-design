# SBOM & License Policy

One-line: Generate SBOM, scan licenses/vulns in CI, and lock builds.

## SBOM
- Tooling: CycloneDX (python) or license files enumerator.
- Artifacts: `sbom.json` per release.

## License
- Allow: Apache-2.0, MIT, BSD.
- Review: GPL/AGPL copyleft dependencies → approval required.

## Build Lock
- `requirements.txt`/`conda env export --from-history` recorded; `config.lock.json` saved per run.

