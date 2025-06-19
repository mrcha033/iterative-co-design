import wandb
from omegaconf import DictConfig, OmegaConf


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
        print("📊 W&B logging disabled by configuration")
        wandb.init(mode="disabled")
        return

    # Sanitize the config for wandb
    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    print(f"📊 Initializing W&B logging in {wandb_mode} mode")
    wandb.init(
        project=cfg.project_name,
        config=config_dict,
        job_type="experiment",
        name=f"{cfg.method}-{cfg.model.name.split('/')[-1]}-{cfg.dataset.name}",
        mode=wandb_mode,
    )
