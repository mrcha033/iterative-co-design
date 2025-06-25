import unittest
from unittest.mock import patch, MagicMock
from omegaconf import OmegaConf
from utils.logging import initialize_wandb, setup_logging

class TestLogging(unittest.TestCase):

    @patch('wandb.init')
    def test_initialize_wandb(self, mock_wandb_init):
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
    def test_initialize_wandb_disabled(self, mock_wandb_init):
        cfg = OmegaConf.create({
            "wandb": {"mode": "disabled"}
        })
        initialize_wandb(cfg)
        mock_wandb_init.assert_called_once_with(mode="disabled")

    @patch('logging.basicConfig')
    def test_setup_logging(self, mock_basic_config):
        cfg = OmegaConf.create({
            "wandb": {"mode": "disabled"}
        })
        setup_logging(cfg)
        mock_basic_config.assert_called_once()

if __name__ == '__main__':
    unittest.main()