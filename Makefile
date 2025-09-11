.PHONY: repro-smoke pair test schema clean-runs

repro-smoke:
	bash scripts/repro_smoke.sh

pair:
	python3 -m icd.cli.main pair -c configs/mock.json --out runs/pair01

test:
	pytest -q tests/unit && pytest -q tests/integration && pytest -q tests/ir

schema:
	python3 -m icd.cli.main run --print-schema -c configs/mock.json --out /tmp/ignore

clean-runs:
	rm -rf runs .icd_cache || true

