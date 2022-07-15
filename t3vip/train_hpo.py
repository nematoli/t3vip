import logging
from pathlib import Path
import sys
import os
import math

cwd_path = Path(__file__).absolute().parents[0]
parent_path = cwd_path.parents[0]
# This is for using the locally installed repo clone when using slurm
sys.path.insert(0, parent_path.as_posix())

import hydra
from omegaconf import DictConfig, OmegaConf
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from pytorch_lightning import seed_everything, Trainer
from t3vip.datasets.utils.load_utils import get_intrinsics
from t3vip.utils.utils import print_system_env_info, get_last_checkpoint, get_model_via_name
from t3vip.train import setup_logger, setup_callbacks, log_rank_0

logger = logging.getLogger(__name__)
os.chdir(cwd_path)


def overwrite_model_cfg(model_cfg, sampled_cfg):

    # Model params
    for key, value in sampled_cfg.items():
        if key in model_cfg:
            model_cfg[key] = value
        
    # Optimizer params
    if "lr" in sampled_cfg:
        model_cfg.optimizer.lr = sampled_cfg["lr"]
    if "weight_decay" in sampled_cfg:
        model_cfg.optimizer.weight_decay = sampled_cfg["weight_decay"]
    if "eps" in sampled_cfg:
        model_cfg.optimizer.eps = sampled_cfg["eps"]

    # Lr scheduler
    if "lr_scheduler" in sampled_cfg and sampled_cfg["lr_scheduler"] is not None:
        lr_scheduler_cfg = {"_target_": sampled_cfg["lr_scheduler"]}
        if "CosineAnnealingLR" in sampled_cfg["lr_scheduler"]:
            lr_scheduler_cfg["T_max"] = max_epochs_ceil
        model_cfg.lr_scheduler = lr_scheduler_cfg

    return model_cfg


def train_t3vip(config: dict = {}, cfg: DictConfig = {}, budget: int = 10, num_gpus: int = 0, checkpoint_dir=None):
    """
    This is called to start a training.
    Args:
        cfg: hydra config
    """
    seed_everything(cfg.seed, workers=True)
    dataset_name = cfg.datamodule.dataset["_target_"].split(".")[-1]
    intrinsics, xygrid = get_intrinsics(cfg.datamodule, dataset_name)
    datamodule = hydra.utils.instantiate(cfg.datamodule, intrinsics=intrinsics, xygrid=xygrid)
    datamodule.setup(stage="fit")

    cfg.model = overwrite_model_cfg(model_cfg=cfg.model, sampled_cfg=config)
    model = hydra.utils.instantiate(cfg.model, intrinsics=intrinsics, xygrid=xygrid)

    cfg.trainer.gpus = math.ceil(num_gpus)
    num_batches_per_epoch = len(datamodule.train_dataloader())
    cfg.trainer.val_check_interval = int(num_batches_per_epoch / cfg.ray.reports_per_epoch)
    cfg.trainer.max_steps = budget * cfg.trainer.val_check_interval + 1

    log_rank_0(f"Training with the following config:\n{OmegaConf.to_yaml(cfg)}")
    log_rank_0(print_system_env_info())
    train_logger = setup_logger(cfg, model, tune.get_trial_id())
    callbacks = setup_callbacks(cfg.callbacks)
    metrics = {"SPSNR": "metrics/val-SPSNR", "IPSNR": "metrics/val-IPSNR", "VGG": "metrics/val-VGG"}
    tc = TuneReportCheckpointCallback(metrics=metrics, filename="checkpoint", on="validation_end")
    callbacks.append(tc)

    trainer_args = {
        **cfg.trainer,
        "logger": train_logger,
        "callbacks": callbacks,
        "benchmark": False,
    }

    trainer = Trainer(**trainer_args)

    # Start training
    trainer.fit(model, datamodule=datamodule)


def get_search_space(search_space):
    config = {}
    merged_ss = [item for sublist in search_space.values() for item in sublist]
    for entry in merged_ss:
        search_function = getattr(tune, entry.function)
        config[entry.name] = search_function(**entry.input)
    return config


def get_progress_reporter(reporter_cfg):
    reporter = CLIReporter(**reporter_cfg)
    return reporter


def ray_optim(cfg):
    redis_pwd = os.getenv("redis_password")
    if redis_pwd is None:
        ray.init()
    else:
        ray.init("auto", _redis_password=redis_pwd)

    max_t = cfg.ray.scheduler.max_t
    time_attr = cfg.ray.scheduler.time_attr

    scheduler = hydra.utils.instantiate(cfg.ray.scheduler)
    search_alg = hydra.utils.instantiate(cfg.ray.search)
    if cfg.ray.use_concurrency_limiter:
        search_alg = tune.suggest.ConcurrencyLimiter(search_alg, max_concurrent=cfg.ray.max_concurrent)

    progress_reporter = get_progress_reporter(cfg.ray.reporter)
    trainable = tune.with_parameters(
        train_t3vip,
        cfg=cfg,
        budget=max_t,
        num_gpus=cfg.ray.gpus_per_trial,
    )

    analysis = tune.run(
        trainable,
        name=cfg.ray.name,
        local_dir=cfg.ray.shared_directory,
        resources_per_trial={"cpu": cfg.ray.cpus_per_trial, "gpu": cfg.ray.gpus_per_trial},
        progress_reporter=progress_reporter,
        config=get_search_space(cfg.ray.search_space),
        scheduler=scheduler,
        search_alg=search_alg,
        num_samples=cfg.ray.num_samples,
        stop={time_attr: max_t},
        metric=cfg.ray.opt_metric,
        mode="max",
        sync_config=tune.SyncConfig(syncer=None),  # Disable syncing
        resume="AUTO",
    )

    print("Best hyperparameters found were: ", analysis.best_config)


@hydra.main(config_path="../config", config_name="optim", version_base=None)
def main(cfg: DictConfig) -> None:
    ray_optim(cfg)


if __name__ == "__main__":
    main()
