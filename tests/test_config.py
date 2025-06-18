import tempfile
import yaml
from pathlib import Path
from src.utils.config import load_yaml_config, load_omegaconf, config_to_namespace


class TestConfigLoader:
    def test_load_yaml_config_basic(self):
        """Test basic YAML config loading."""
        # Create a temporary config file
        test_config = {
            "model": {"name": "test-model", "size": "small"},
            "dataset": {"name": "test-data", "batch_size": 32},
            "training": {"epochs": 10, "lr": 0.001},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(test_config, f)
            temp_path = f.name

        try:
            # Load the config
            loaded_config = load_yaml_config(temp_path)

            # Verify the content
            assert loaded_config == test_config
            assert loaded_config["model"]["name"] == "test-model"
            assert loaded_config["dataset"]["batch_size"] == 32
            assert loaded_config["training"]["epochs"] == 10

        finally:
            # Clean up
            Path(temp_path).unlink()

    def test_load_omegaconf(self):
        """Test OmegaConf loading."""
        test_config = {"model": {"name": "test-model"}, "dataset": {"batch_size": 16}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(test_config, f)
            temp_path = f.name

        try:
            # Load with OmegaConf
            config = load_omegaconf(temp_path)

            # Verify it's a DictConfig
            from omegaconf import DictConfig

            assert isinstance(config, DictConfig)

            # Verify content access
            assert config.model.name == "test-model"
            assert config.dataset.batch_size == 16

        finally:
            Path(temp_path).unlink()

    def test_config_to_namespace(self):
        """Test conversion to namespace object."""
        config_dict = {
            "model": {"name": "test-model", "hidden_size": 768},
            "training": {"epochs": 5},
        }

        namespace = config_to_namespace(config_dict)

        # Test attribute access
        assert namespace.model.name == "test-model"
        assert namespace.model.hidden_size == 768
        assert namespace.training.epochs == 5

        # Test string representation
        repr_str = repr(namespace)
        assert "ConfigNamespace" in repr_str

    def test_file_not_found(self):
        """Test handling of missing config file."""
        import pytest

        with pytest.raises(FileNotFoundError):
            load_yaml_config("nonexistent_config.yaml")
