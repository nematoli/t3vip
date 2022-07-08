import logging
from typing import Dict, Optional, Union, Any, List
import hydra
from omegaconf import DictConfig
import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

from t3vip.utils.net_utils import transform_ptc, scheduled_sampling, compute_occlusion
from t3vip.utils.cam_utils import get2Dflow, get_prj_mat
from t3vip.datasets.utils.load_utils import get_ptc_from_dpt
from t3vip.helpers import softsplat
from t3vip.utils.transforms import RealDepthTensor, ScaleDepthTensor
from t3vip.helpers.losses import calc_3d_loss, calc_2d_loss

logger = logging.getLogger(__name__)


@rank_zero_only
def log_rank_0(*args, **kwargs):
    # when using ddp, only log with rank 0 process
    logger.info(*args, **kwargs)


class T3VIP(pl.LightningModule):
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
        se3_decoder: DictConfig,
        rgbd_inpainter: DictConfig,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        intrinsics: Dict,
        xygrid: torch.Tensor,
        act_cond: bool,
        num_context_frames: int,
        prediction_horizon: int,
        splat: str,
        alpha_rcr: float,
        alpha_rcd: float,
        alpha_knn: float,
        alpha_sfs: float,
        alpha_ofs: float,
        alpha_kl: float,
        min_dpt: float,
        max_dpt: float,
        time_invariant: bool,
    ):
        super(T3VIP, self).__init__()
        self.obs_encoder = hydra.utils.instantiate(obs_encoder)
        self.act_encoder = hydra.utils.instantiate(act_encoder)
        self.msk_decoder = hydra.utils.instantiate(msk_decoder)
        self.se3_decoder = hydra.utils.instantiate(se3_decoder)
        self.rgbd_inpainter = hydra.utils.instantiate(rgbd_inpainter)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.prj_mat = get_prj_mat(intrinsics)
        self.xygrid = torch.unsqueeze(xygrid, dim=0)
        self.act_cond = act_cond
        self.num_context_frames = num_context_frames
        self.prediction_horizon = prediction_horizon
        self.splat = splat
        self.alpha_rcr = alpha_rcr
        self.alpha_rcd = alpha_rcd
        self.alpha_knn = alpha_knn
        self.alpha_sfs = alpha_sfs
        self.alpha_ofs = alpha_ofs
        self.alpha_kl = alpha_kl
        self.min_dpt = min_dpt
        self.max_dpt = max_dpt
        self.time_invariant = time_invariant
        self.scale_dpt = ScaleDepthTensor(self.min_dpt, self.max_dpt)
        self.real_dpt = RealDepthTensor(self.min_dpt, self.max_dpt)
        self.intrinsics = intrinsics
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.optimizer, params=self.parameters())
        return {"optimizer": optimizer}

    def forward(
        self, dpts: torch.Tensor, rgbs: torch.Tensor, acts: torch.Tensor, stts: torch.Tensor, p: float, inference: bool
    ) -> Dict[str, torch.Tensor]:
        """
        Main forward pass for at each step.
        Args:
            ptc_t: point cloud of time step t.
            act_t: action executed at time step t.
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
        rgbs_complete = None
        dpts_complete = None

        outputs_cell = {}
        outputs = {
            "emb_t": [],
            "masks_t": [],
            "tfmptc_t": [],
            "sflow_t": [],
            "oflow_t": [],
            "occmap_t": [],
            "irgb_t": [],
            "idpt_t": [],
            "nxtrgb": [],
            "nxtdpt": [],
        }

        for i in range(S - 1):
            act_t = acts[:, i] if acts is not None else None
            stt_t = stts[:, i] if stts is not None else None
            if i < self.num_context_frames:
                dpt_t = dpts[:, i]
                rgb_t = rgbs[:, i]
            elif self.training:
                # scheduled sampling
                num_samples_true = int(B * p)
                [dpt_t] = scheduled_sampling([dpts[:, i]], [outputs_cell["nxtdpt"]], num_samples_true)
                [rgb_t] = scheduled_sampling([rgbs[:, i]], [outputs_cell["nxtrgb"]], num_samples_true)
            else:
                dpt_t = outputs_cell["nxtdpt"]
                rgb_t = outputs_cell["nxtrgb"]

            outputs_cell, latent, lstm_states = self.forward_single_frame(
                rgb_t,
                dpt_t,
                act_t,
                stt_t,
                rgbs_complete,
                dpts_complete,
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
        dpt_t: torch.Tensor,
        act_t: torch.Tensor,
        stt_t: torch.Tensor,
        rgb_complete: torch.Tensor,
        dpt_complete: torch.Tensor,
        latent: torch.Tensor,
        lstm_states: List[torch.Tensor],
    ):

        if lstm_states is not None:
            obs_lstms = lstm_states[0:4]
            act_lstms = lstm_states[4]
            msk_lstms = lstm_states[5:7]
            inp_lstms = lstm_states[7:]
        else:
            lstm_states = [None] * 9
            obs_lstms, act_lstms, msk_lstms, inp_lstms = (None, None, None, None)

        rgbd_t = torch.cat([rgb_t, self.scale_dpt(dpt_t)], dim=1)
        emb_t, obs_lstms = self.obs_encoder(rgbd_t, obs_lstms)
        emb_ta, act_lstms = self.act_encoder(emb_t[-1], act_t, stt_t, latent, act_lstms)
        emb_t[-1] = emb_ta

        masks_t, _, msk_lstms = self.msk_decoder(emb_t, msk_lstms)
        se3s_t = self.se3_decoder(emb_t[-1])

        ptc_t = get_ptc_from_dpt(dpt_t, self.xygrid)
        tfmptc_t = transform_ptc(ptc_t, masks_t, se3s_t)

        sflow_t = tfmptc_t - ptc_t
        oflow_t = get2Dflow(tfmptc_t, self.prj_mat.to(tfmptc_t.device))

        fwd_rgb = softsplat.FunctionSoftsplat(rgb_t, oflow_t, None, self.splat)
        fwd_dpt = softsplat.FunctionSoftsplat(tfmptc_t, oflow_t, None, self.splat).narrow(1, 2, 1)

        inp_rgb, inp_dpt, inp_lstms = self.rgbd_inpainter(emb_t, inp_lstms)
        occ_map = compute_occlusion(tfmptc_t, self.intrinsics)

        nxt_rgb = (1 - occ_map) * fwd_rgb + occ_map * inp_rgb
        nxt_dpt = (1 - occ_map) * fwd_dpt + occ_map * self.real_dpt(inp_dpt)
        nxt_dpt = torch.clamp(nxt_dpt, min=self.min_dpt, max=self.max_dpt)

        outputs = {
            "emb_t": emb_t[-1],
            "masks_t": masks_t,
            "tfmptc_t": tfmptc_t,
            "sflow_t": sflow_t,
            "oflow_t": oflow_t,
            "occmap_t": occ_map,
            "irgb_t": inp_rgb,
            "idpt_t": inp_dpt,
            "nxtrgb": nxt_rgb,
            "nxtdpt": nxt_dpt,
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
        p = 1.0
        inference = False

        out = self(batch["depth_obs"], batch["rgb_obs"], acts, stts, p, inference)
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
        p = 0.0
        inference = False
        out = self(batch["depth_obs"], batch["rgb_obs"], acts, stts, p, inference)
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
        p = 0.0
        inference = False
        out = self(batch["depth_obs"], batch["rgb_obs"], acts, stts, p, inference)
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
        (loss_3d, loss_2d, loss_kl, total_loss) = (
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
        )

        ptc_1, ptc_2 = batch["ptc_obs"][:, :-1], batch["ptc_obs"][:, 1:]
        dpt_1, dpt_2 = batch["depth_obs"][:, :-1], batch["depth_obs"][:, 1:]
        rgb_1, rgb_2 = batch["rgb_obs"][:, :-1], batch["rgb_obs"][:, 1:]

        rcd_loss, knn_loss, sfs_loss = calc_3d_loss(
            self.alpha_rcd,
            self.alpha_knn,
            self.alpha_sfs,
            dpt_2,
            ptc_2,
            rgb_1,
            outputs["nxtdpt"],
            outputs["tfmptc_t"],
            outputs["sflow_t"],
        )
        loss_3d = rcd_loss + knn_loss + sfs_loss

        rcr_loss, ofs_loss = calc_2d_loss(
            self.alpha_rcr, self.alpha_ofs, rgb_1, rgb_2, outputs["nxtrgb"], outputs["oflow_t"]
        )
        loss_2d = rcr_loss + ofs_loss

        total_loss = loss_3d + loss_2d + loss_kl

        losses = {
            "loss_total": total_loss,
            "loss_2d": loss_2d,
            "loss_3d": loss_3d,
            "loss_kl": loss_kl,
            "loss3d_dptrcs": rcd_loss,
            "loss3d_knn": knn_loss,
            "loss3d_sflowsmt": sfs_loss,
            "loss2d_rgbrcs": rcr_loss,
            "loss2d_oflowsmt": ofs_loss,
        }

        return losses

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
