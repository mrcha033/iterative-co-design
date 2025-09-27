.PHONY: repro-smoke repro-codesign repro-ablation pair test schema clean-runs

repro-smoke:
	bash scripts/repro_smoke.sh

repro-codesign:
        bash scripts/repro_codesign.sh

repro-ablation:
        bash scripts/repro_ablation.sh

pair:
	python3 -m icd.cli.main pair -c configs/mock.json --out runs/pair01

test:
	pytest -q tests/unit && pytest -q tests/integration && pytest -q tests/ir

schema:
	python3 -m icd.cli.main run --print-schema -c configs/mock.json --out /tmp/ignore

clean-runs:
	rm -rf runs .icd_cache || true
