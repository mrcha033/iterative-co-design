from icd.runtime.nvml_logger import NVMLPowerLogger


def test_nvml_logger_handles_missing_nvml():
    logger = NVMLPowerLogger()
    with logger:
        logger.tick()
    assert logger.energy_j() == 0.0
