import logging
from pathlib import Path
import sys
from typing import List, Union
import os

cwd_path = Path(__file__).absolute().parents[0]
parent_path = cwd_path.parents[0]
# This is for using the locally installed repo clone when using slurm
sys.path.insert(0, parent_path.as_posix())

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback, LightningModule, seed_everything, Trainer
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities import rank_zero_only
from t3vip.utils.utils import get_git_commit_hash, print_system_env_info, get_last_checkpoint, get_model_via_name
from t3vip.datasets.utils.load_utils import get_intrinsics

logger = logging.getLogger(__name__)
os.chdir(cwd_path)


@hydra.main(config_path="../config", config_name="config", version_base=None)
def train(cfg: DictConfig) -> None:
    """
    This is called to start a training.
    Args:
        cfg: hydra config
    """
    seed_everything(cfg.seed, workers=True)
    dataset_name = cfg.datamodule.dataset["_target_"].split(".")[-1]
    intrinsics, xygrid = get_intrinsics(cfg.datamodule, dataset_name)
    datamodule = hydra.utils.instantiate(cfg.datamodule, intrinsics=intrinsics, xygrid=xygrid)
    chk = get_last_checkpoint(Path.cwd())

    # Load Model
    if chk is not None:
        model_name = cfg.model["_target_"].split(".")[-1]
        models_m = get_model_via_name(model_name)
        model = getattr(models_m, model_name).load_from_checkpoint(chk.as_posix())
    else:
        model = hydra.utils.instantiate(cfg.model, intrinsics=intrinsics, xygrid=xygrid)

    log_rank_0(f"Training with the following config:\n{OmegaConf.to_yaml(cfg)}")
    # log_rank_0("Repo commit hash: {}".format(get_git_commit_hash(Path(hydra.utils.to_absolute_path(__file__)))))
    log_rank_0(print_system_env_info())
    train_logger = setup_logger(cfg, model, cfg.logger.name)
    callbacks = setup_callbacks(cfg.callbacks)

    trainer_args = {
        **cfg.trainer,
        "logger": train_logger,
        "callbacks": callbacks,
        "benchmark": False,
    }

    trainer = Trainer(**trainer_args)

    # Start training
    trainer.fit(model, datamodule=datamodule, ckpt_path=chk)


def setup_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    callbacks = [hydra.utils.instantiate(cb) for cb in callbacks_cfg.values()]
    return callbacks


def setup_logger(cfg: DictConfig, model: LightningModule, name: str, evaluate: bool = False) -> LightningLoggerBase:
    """
    Set up the logger (tensorboard or wandb) from hydra config.
    Args:
        cfg: Hydra config
        model: LightningModule
    Returns:
        logger
    """
    cwd_path = Path.cwd()
    print("cwd_path", cwd_path)
    if cfg.slurm:
        path_date = cwd_path.parts[5]
        path_time = cwd_path.parts[6]
        cfg.logger.name = path_date + "/" + path_time
    else:
        cfg.logger.name = name

    if evaluate:
        cfg.logger.name += "_eval"

    if cfg.logger["_target_"].split(".")[-1] == "WandbLogger":
        if hasattr(cfg, "ray"):
            cfg.logger.group = cfg.ray.name
            cfg.logger.name = name
        else:
            cfg.logger.group = cwd_path.parts[5]
        cfg.logger.id = cfg.logger.name.replace("/", "_")

    train_logger = hydra.utils.instantiate(cfg.logger)

    return train_logger


@rank_zero_only
def log_rank_0(*args, **kwargs):
    # when using ddp, only log with rank 0 process
    logger.info(*args, **kwargs)


if __name__ == "__main__":
    train()
