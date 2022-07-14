import logging
from typing import Dict, Optional, Union, Any, List
import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only


logger = logging.getLogger(__name__)


@rank_zero_only
def log_rank_0(*args, **kwargs):
    # when using ddp, only log with rank 0 process
    logger.info(*args, **kwargs)


class VideoModel(pl.LightningModule):
    """
    The lightning module used for training self-supervised t3vip.
    Args:
    """

    def __init__(self):
        super(VideoModel, self).__init__()

    def configure_optimizers(self):
        raise NotImplementedError

    def forward(
        self, dpts: torch.Tensor, rgbs: torch.Tensor, acts: torch.Tensor, stts: torch.Tensor, p: float, inference: bool
    ) -> Dict[str, torch.Tensor]:
        """
        Main forward pass for at each step.
        Args:
            dpts: point cloud of time step t.
            rgbs: point cloud of time step t.
            acts: point cloud of time step t.
            stts: point cloud of time step t.
            p: action executed at time step t.
            inference: action executed at time step t.
        Returns:
            outputs (dict):
                - 'tfmptc_t' (Tensor): predicted transformed point cloud of time step t
                - 'masks_t' (Tensor): predicted masks of time step t
                - 'sflow_t' (Tensor): predicted scene flow of time step t
                - 'oflow_t' (Tensor): predicted optical flow of time step t
        """

        raise NotImplementedError

    def forward_single_frame(
        self,
        rgb_t: torch.Tensor,
        dpt_t: torch.Tensor,
        act_t: torch.Tensor,
        stt_t: torch.Tensor,
        rgb_1: torch.Tensor,
        rgb_complete: torch.Tensor,
        dpt_complete: torch.Tensor,
        latent: torch.Tensor,
        lstm_states: List[torch.Tensor],
    ):

        raise NotImplementedError

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Compute and return the training loss.
        Args:
            batch (dict):
                - 'ptc_obs' (Tensor): Two consecutive point clouds of static camera
                - 'depth_obs' (Tensor): Two consecutive depth images of static camera
                - 'rgb_obs' (Tensor): Two consecutive RGB images of static camera
                - 'action' (Tensor): Ground truth action between two consecutive frames.
            batch_idx (int): Integer displaying index of this batch.
        Returns:
            loss tensor
        """

        raise NotImplementedError

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Compute and return the validation loss.
        Args:
            batch (dict):
                - 'ptc_obs' (Tensor): Two consecutive point clouds of static camera
                - 'depth_obs' (Tensor): Two consecutive depth images of static camera
                - 'rgb_obs' (Tensor): Two consecutive RGB images of static camera
                - 'action' (Tensor): Ground truth action between two consecutive frames.
            batch_idx (int): Integer displaying index of this batch.
        Returns:
            loss tensor
        """
        raise NotImplementedError

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Compute and return the test loss.
        Args:
            batch (dict):
                - 'ptc_obs' (Tensor): Two consecutive point clouds of static camera
                - 'depth_obs' (Tensor): Two consecutive depth images of static camera
                - 'rgb_obs' (Tensor): Two consecutive RGB images of static camera
                - 'action' (Tensor): Ground truth action between two consecutive frames.
            batch_idx (int): Integer displaying index of this batch.
        Returns:
            loss tensor
        """
        raise NotImplementedError

    @rank_zero_only
    def on_train_epoch_start(self) -> None:
        logger.info(f"Start training epoch {self.current_epoch}")

    @rank_zero_only
    def on_train_epoch_end(self, unused: Optional = None) -> None:
        logger.info(f"Finished training epoch {self.current_epoch}")

    @rank_zero_only
    def on_validation_epoch_start(self) -> None:
        log_rank_0(f"Start validation epoch {self.current_epoch}")

    @rank_zero_only
    def on_validation_epoch_end(self) -> None:
        logger.info(f"Finished validation epoch {self.current_epoch}")

    def loss(self, batch: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    @torch.no_grad()
    def metrics(self, batch: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def set_kl_beta(self, alpha_kl):
        """Set alpha_kl from Callback"""
        self.alpha_kl = alpha_kl

    def log_loss(self, loss: Dict[str, torch.Tensor], mode: str):
        for key, val in loss.items():
            if loss[key] != 0:
                info = key.split("_")
                self.log(info[0] + "/{}_".format(mode) + info[1], loss[key])

    def log_metrics(self, metrics: Dict[str, torch.Tensor], mode: str, on_step: bool, on_epoch: bool):
        for key, val in metrics.items():
            if metrics[key] != 0:
                info = key.split("_")
                self.log(info[0] + "/{}-".format(mode) + info[1], metrics[key], on_step=on_step, on_epoch=on_epoch)
