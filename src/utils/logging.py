import wandb
from omegaconf import DictConfig, OmegaConf

def initialize_wandb(cfg: DictConfig):
    """
    Initializes a Weights & Biases run.
    """
    # Sanitize the config for wandb
    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    
    wandb.init(
        project=cfg.project_name,
        config=config_dict,
        job_type="experiment",
        name=f"{cfg.method}-{cfg.model.name.split('/')[-1]}-{cfg.dataset.name}"
    ) 