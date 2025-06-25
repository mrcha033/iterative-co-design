from unittest.mock import patch
from omegaconf import OmegaConf
from utils.logging import initialize_wandb, setup_logging

@patch('wandb.init')
def test_initialize_wandb(mock_wandb_init):
    cfg = OmegaConf.create({
        "project_name": "test_project",
        "method": "test_method",
        "model": {"name": "test_model"},
        "dataset": {"name": "test_dataset"},
        "wandb": {"mode": "online"}
    })
    initialize_wandb(cfg)
    mock_wandb_init.assert_called_once()

@patch('wandb.init')
def test_initialize_wandb_disabled(mock_wandb_init):
    cfg = OmegaConf.create({
        "wandb": {"mode": "disabled"}
    })
    initialize_wandb(cfg)
    mock_wandb_init.assert_called_once_with(mode="disabled")

@patch('logging.basicConfig')
def test_setup_logging(mock_basic_config):
    cfg = OmegaConf.create({
        "wandb": {"mode": "disabled"}
    })
    setup_logging(cfg)
    mock_basic_config.assert_called_once()
