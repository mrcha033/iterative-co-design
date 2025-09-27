from pathlib import Path

import yaml

from icd.hds.training import MaskTrainingConfig, validate_mask_training_config


def test_default_config_passes_guardrail(tmp_path) -> None:
    cfg_doc = yaml.safe_load(Path("configs/hds_default.yaml").read_text())
    cfg = MaskTrainingConfig.from_dict(cfg_doc["mask_training"])
    validate_mask_training_config(cfg)


def test_guardrail_rejects_increasing_temperature() -> None:
    cfg = MaskTrainingConfig(steps=10, temperature_init=0.5, temperature_final=1.0)
    try:
        validate_mask_training_config(cfg)
    except ValueError as exc:
        assert "temperature_final" in str(exc)
    else:
        raise AssertionError("guardrail should reject increasing temperature")

