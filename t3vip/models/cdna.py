import logging
from typing import Dict, Optional, Union, Any, List
import hydra
from omegaconf import DictConfig
import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

from t3vip.utils.net_utils import gen_nxtrgb, scheduled_sampling
from t3vip.helpers.losses import calc_2d_loss, calc_kl_loss

logger = logging.getLogger(__name__)

from collections import namedtuple

ContState = namedtuple("ContState", ["mean", "std"])


@rank_zero_only
def log_rank_0(*args, **kwargs):
    # when using ddp, only log with rank 0 process
    logger.info(*args, **kwargs)


class CDNA(pl.LightningModule):
    """
    The lightning module used for training self-supervised t3vip.
    Args:
        obs_encoder: DictConfig for ptc_encoder.
        act_encoder: DictConfig for act_encoder.
        msk_decoder: DictConfig for msk_encoder.
        se3_decoder: DictConfig for se3_decoder.
        rgbd_inpainter: DictConfig for rgbd_inpainter.
        optimizer: DictConfig for optimizer.
        lr_scheduler: DictConfig for learning rate scheduler.
    """

    def __init__(
        self,
        obs_encoder: DictConfig,
        act_encoder: DictConfig,
        msk_decoder: DictConfig,
        inference_net: DictConfig,
        knl_decoder: DictConfig,
        distribution: DictConfig,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        intrinsics: Dict,
        xygrid: torch.Tensor,
        act_cond: bool,
        num_context_frames: int,
        alpha_rcr: float,
        alpha_kl: float,
        alpha_l: int,
        reuse_first_rgb: bool,
        time_invariant: bool,
        stochastic: bool,
        gen_iters: int,
    ):
        super(CDNA, self).__init__()
        self.obs_encoder = hydra.utils.instantiate(obs_encoder)
        self.act_encoder = hydra.utils.instantiate(act_encoder)
        self.msk_decoder = hydra.utils.instantiate(msk_decoder)
        self.knl_decoder = hydra.utils.instantiate(knl_decoder)
        self.inference_net = hydra.utils.instantiate(inference_net)
        self.dist = hydra.utils.instantiate(distribution)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.act_cond = act_cond
        self.num_context_frames = num_context_frames
        self.alpha_rcr = alpha_rcr
        self.alpha_kl = alpha_kl
        self.alpha_l = alpha_l
        self.reuse_first_rgb = reuse_first_rgb
        if self.reuse_first_rgb:
            self.msk_decoder.num_masks += 1
        self.time_invariant = time_invariant
        self.stochastic = stochastic
        self.gen_iters = gen_iters
        self.prior = self.dist.set_unit_dist(self.inference_net.dim_latent)
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.optimizer, params=self.parameters())
        return {"optimizer": optimizer}

    def forward(
        self, rgbs: torch.Tensor, acts: torch.Tensor, stts: torch.Tensor, inference: bool, p: float
    ) -> Dict[str, torch.Tensor]:
        """
        Main forward pass for at each step.
        Args:
            rgbs: point cloud of time step t.
            acts: action executed at time step t.
            stts: action executed at time step t.
            p: action executed at time step t.

        Returns:
            outputs (dict):
                - 'tfmptc_t' (Tensor): predicted transformed point cloud of time step t
                - 'masks_t' (Tensor): predicted masks of time step t
                - 'sflow_t' (Tensor): predicted scene flow of time step t
                - 'oflow_t' (Tensor): predicted optical flow of time step t
        """

        B, S, C, H, W = rgbs.size()
        latent = None
        lstm_states = None
        rgb_complete = None

        if inference:
            rgb_complete = rgbs

        rgb_1 = rgbs[:, 0]

        outputs_cell = {}
        outputs = {
            "mu_t": [],
            "std_t": [],
            "emb_t": [],
            "masks_t": [],
            "nxtrgb": [],
        }

        for i in range(S - 1):
            act_t = acts[:, i] if acts is not None else None
            stt_t = stts[:, i] if stts is not None else None
            if i < self.num_context_frames:
                rgb_t = rgbs[:, i]
            elif self.training:
                # scheduled sampling
                num_samples_true = int(B * p)
                [rgb_t] = scheduled_sampling([rgbs[:, i]], [outputs_cell["nxtrgb"]], num_samples_true)
            else:
                rgb_t = outputs_cell["nxtrgb"]

            outputs_cell, latent, lstm_states = self.forward_single_frame(
                rgb_t,
                act_t,
                stt_t,
                rgb_1,
                rgb_complete,
                latent,
                lstm_states,
            )

            for key, val in outputs_cell.items():
                if key not in outputs.keys():
                    outputs[key] = []
                outputs[key].append(val)

            if not self.time_invariant or self.training:
                latent = None

        for key, val in outputs.items():
            outputs[key] = torch.stack(outputs[key], dim=1)

        return outputs

    def forward_single_frame(
        self,
        rgb_t: torch.Tensor,
        act_t: torch.Tensor,
        stt_t: torch.Tensor,
        rgb_1: torch.Tensor,
        rgb_complete: torch.Tensor,
        latent: torch.Tensor,
        lstm_states: List[torch.Tensor],
    ):
        prior = self.dist.repeat_to_device(self.prior, rgb_t.device, rgb_t.size(0))
        dist = self.dist.get_dist(prior)
        state = prior

        if latent is None and self.stochastic:
            # infer posterior distribution q(z|x)
            if rgb_complete is not None:
                posterior = self.inference_net(rgb_complete)
                dist = self.dist.get_dist(posterior)
                state = posterior
            latent = self.dist.sample_latent_code(dist).to(act_t.device)

        if lstm_states is not None:
            obs_lstms = lstm_states[0:4]
            act_lstms = lstm_states[4]
            msk_lstms = lstm_states[5:7]
        else:
            lstm_states = [None] * 7
            obs_lstms, act_lstms, msk_lstms = (None, None, None)

        emb_t, obs_lstms = self.obs_encoder(rgb_t, obs_lstms)
        emb_ta, act_lstms = self.act_encoder(emb_t[-1], act_t, stt_t, latent, act_lstms)
        emb_t[-1] = emb_ta

        masks_t, _, msk_lstms = self.msk_decoder(emb_t, msk_lstms)
        tfmrgb_t = self.knl_decoder(emb_t[-1], rgb_t)

        rgb_extra = rgb_1 if self.reuse_first_rgb else None
        nxt_rgb = gen_nxtrgb(rgb_t, masks_t, tfmrgb_t, rgb_extra)

        outputs = {
            "mu_t": state.mean,
            "std_t": state.std,
            "emb_t": emb_t[-1],
            "masks_t": masks_t,
            "nxtrgb": nxt_rgb,
        }

        return outputs, latent, lstm_states

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

        acts = batch["actions"] if self.act_cond else None
        stts = None
        inference = True if self.stochastic and self.global_step > self.gen_iters else False
        p = 1.0

        out = self(batch["rgb_obs"], acts, stts, inference, p)
        losses = self.loss(batch, out)
        self.log_loss(losses, mode="train")
        return {"loss": losses["loss_total"], "out": out}

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
        acts = batch["actions"] if self.act_cond else None
        stts = None
        inference = False
        p = 0.0
        out = self(batch["rgb_obs"], acts, stts, inference, p)
        losses = self.loss(batch, out)
        self.log_loss(losses, mode="val")
        return {"loss": losses["loss_total"], "out": out}

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
        acts = batch["actions"] if self.act_cond else None
        stts = None
        inference = False
        p = 0.0
        out = self(batch["rgb_obs"], acts, stts, inference, p)
        losses = self.loss(batch, out)
        self.log_loss(losses, mode="test")
        return {"loss": losses["loss_total"], "out": out}

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

        rgb_1, rgb_2 = batch["rgb_obs"][:, :-1], batch["rgb_obs"][:, 1:]

        rcr_loss, _ = calc_2d_loss(self.alpha_rcr, 0, self.alpha_l, rgb_1, rgb_2, outputs["nxtrgb"], None)

        if self.stochastic:
            prior = self.dist.repeat_to_device(
                self.prior, outputs["mu_t"].device, outputs["mu_t"].size(0), outputs["mu_t"].size(1)
            )
            posterior = ContState(outputs["mu_t"], outputs["std_t"])
            kl_loss = calc_kl_loss(self.alpha_kl, self.dist, prior, posterior)
        else:
            kl_loss = torch.tensor(0.0).to(rgb_1.device)

        total_loss = rcr_loss + kl_loss

        losses = {
            "loss_total": total_loss,
            "loss2d_rgbrcs": rcr_loss,
            "loss_kl": kl_loss,
        }

        return losses

    def set_kl_beta(self, alpha_kl):
        """Set alpha_kl from Callback"""
        self.alpha_kl = alpha_kl

    def log_loss(
        self,
        loss: Dict[str, torch.Tensor],
        mode: str,
    ):

        for key, val in loss.items():
            if loss[key] != 0:
                info = key.split("_")
                self.log(
                    info[0] + "/{}_".format(mode) + info[1],
                    loss[key],
                    # batch_size=batch_size,
                )
