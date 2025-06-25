import wandb
import logging
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
from pathlib import Path
from typing import Optional

# Create a module-level logger
logger = logging.getLogger(__name__)


def setup_logging(cfg: DictConfig, level: str = "INFO") -> None:
    """
    Setup logging configuration and optionally initialize W&B.

    Args:
        cfg: Configuration object
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    # Setup basic logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Initialize W&B if configured
    if hasattr(cfg, "wandb"):
        initialize_wandb(cfg)


def initialize_wandb(cfg: DictConfig):
    """
    Initializes a Weights & Biases run if enabled in configuration.

    Respects the cfg.wandb.mode setting:
    - "online": Normal W&B logging
    - "offline": Local logging only
    - "disabled": No W&B logging
    """
    # Check if wandb is disabled in config
    wandb_mode = OmegaConf.select(cfg, "wandb.mode", default="online")

    if wandb_mode == "disabled":
        print("W&B logging disabled by configuration")
        wandb.init(mode="disabled")
        return

    # Sanitize the config for wandb
    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    print(f"Initializing W&B logging in {wandb_mode} mode")
    wandb.init(
        project=cfg.project_name,
        config=config_dict,
        job_type="experiment",
        name=f"{cfg.method}-{cfg.model.name.split('/')[-1]}-{cfg.dataset.name}",
        mode=wandb_mode,
    )


def setup_experiment_logging(cfg: DictConfig, log_file: Optional[Path] = None) -> None:
    """
    Setup experiment-specific logging with detailed formatting.
    
    Args:
        cfg: Configuration object
        log_file: Optional path to log file
    """
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    root_logger.handlers = []
    
    # Console handler with simple format
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(simple_formatter)
    console_handler.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    
    # File handler with detailed format if log_file provided
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(detailed_formatter)
        file_handler.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")
    
    # Log experiment configuration
    logger.info("=" * 60)
    logger.info(f"Starting experiment: {cfg.method}")
    logger.info(f"Model: {cfg.model.name}")
    logger.info(f"Dataset: {cfg.dataset.name}")
    logger.info(f"Random seed: {cfg.seed}")
    logger.info("=" * 60)


def log_metrics(metrics: dict, prefix: str = "") -> None:
    """
    Log metrics in a formatted way.
    
    Args:
        metrics: Dictionary of metric names and values
        prefix: Optional prefix for the log message
    """
    if prefix:
        logger.info(f"{prefix}:")
    
    for metric_name, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {metric_name}: {value:.4f}")
        else:
            logger.info(f"  {metric_name}: {value}")


def log_time_elapsed(start_time: datetime, operation: str) -> None:
    """
    Log time elapsed for an operation.
    
    Args:
        start_time: Start time of the operation
        operation: Description of the operation
    """
    elapsed = datetime.now() - start_time
    total_seconds = elapsed.total_seconds()
    
    if total_seconds < 60:
        logger.info(f"{operation} completed in {total_seconds:.1f} seconds")
    else:
        minutes = int(total_seconds // 60)
        seconds = total_seconds % 60
        logger.info(f"{operation} completed in {minutes}m {seconds:.1f}s")
