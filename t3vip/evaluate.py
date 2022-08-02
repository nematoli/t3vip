import logging
from pathlib import Path
import sys
import os

cwd_path = Path(__file__).absolute().parents[0]
parent_path = cwd_path.parents[0]
# This is for using the locally installed repo clone when using slurm
sys.path.insert(0, parent_path.as_posix())

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything, Trainer
from t3vip.utils.utils import get_git_commit_hash, print_system_env_info, get_last_checkpoint, get_model_via_name
from t3vip.datasets.utils.load_utils import get_intrinsics
from t3vip.train import setup_logger, setup_callbacks, log_rank_0

logger = logging.getLogger(__name__)
os.chdir(cwd_path)


@hydra.main(config_path="../config", config_name="config", version_base=None)
def evaluate(cfg: DictConfig) -> None:
    """
    This is called to start a training.
    Args:
        cfg: hydra config
    """
    seed_everything(cfg.seed, workers=True)
    dataset_name = cfg.datamodule.dataset["_target_"].split(".")[-1]
    intrinsics, xygrid = get_intrinsics(cfg.datamodule, dataset_name)
    seq_len = cfg.eval_seq_len
    cfg.datamodule.batch_size = 1
    datamodule = hydra.utils.instantiate(cfg.datamodule, intrinsics=intrinsics, xygrid=xygrid, seq_len=seq_len)
    chk = get_last_checkpoint(Path.cwd())

    model_name = cfg.model["_target_"].split(".")[-1]
    models_m = get_model_via_name(model_name)
    model = getattr(models_m, model_name).load_from_checkpoint(chk.as_posix())

    log_rank_0(f"Evaluating with the following config:\n{OmegaConf.to_yaml(cfg)}")
    # log_rank_0("Repo commit hash: {}".format(get_git_commit_hash(Path(hydra.utils.to_absolute_path(__file__)))))
    log_rank_0(print_system_env_info())
    train_logger = setup_logger(cfg, model, cfg.logger.name, evaluate=True)
    cfg.callbacks.plot.vis_imgs = True
    callbacks = setup_callbacks(cfg.callbacks)

    trainer_args = {
        **cfg.trainer,
        "logger": train_logger,
        "callbacks": callbacks,
        "benchmark": False,
    }

    trainer = Trainer(**trainer_args)

    # Start training
    trainer.test(model, datamodule=datamodule, ckpt_path=chk)


if __name__ == "__main__":
    evaluate()
