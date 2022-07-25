import logging
from typing import Dict, Union, Any, List
import hydra
from omegaconf import DictConfig
import torch

from t3vip.models.video import VideoModel
from t3vip.utils.net_utils import transform_ptc, scheduled_sampling, compute_occlusion
from t3vip.utils.cam_utils import get2Dflow, get_prj_mat
from t3vip.datasets.utils.load_utils import get_ptc_from_dpt
from t3vip.helpers import softsplat
from t3vip.utils.transforms import RealDepthTensor, ScaleDepthTensor
from t3vip.helpers.losses import calc_3d_loss, calc_2d_loss, calc_kl_loss
from t3vip.utils.distributions import ContState
from t3vip.utils.cam_utils import batch_seq_view
from torchmetrics.functional import peak_signal_noise_ratio as PSNR
from torchmetrics.functional import structural_similarity_index_measure as SSIM
from torchmetrics.functional import mean_squared_error as RMSE
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from t3vip.utils.abs_rel_err import mean_absolute_relative_error as MARE

logger = logging.getLogger(__name__)


class T3VIP(VideoModel):
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
        inference_net: DictConfig,
        distribution: DictConfig,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        intrinsics: Dict,
        xygrid: torch.Tensor,
        act_cond: bool,
        num_context_frames: int,
        splat: str,
        alpha_rcr: float,
        alpha_rcd: float,
        alpha_knn: float,
        alpha_sfs: float,
        alpha_ofs: float,
        alpha_kl: float,
        alpha_l: int,
        min_dpt: float,
        max_dpt: float,
        time_invariant: bool,
        stochastic: bool,
        gen_iters: int,
    ):
        super(VideoModel, self).__init__()
        self.obs_encoder = hydra.utils.instantiate(obs_encoder)
        self.act_encoder = hydra.utils.instantiate(act_encoder)
        self.msk_decoder = hydra.utils.instantiate(msk_decoder)
        self.se3_decoder = hydra.utils.instantiate(se3_decoder)
        self.rgbd_inpainter = hydra.utils.instantiate(rgbd_inpainter)
        self.inference_net = hydra.utils.instantiate(inference_net)
        self.dist = hydra.utils.instantiate(distribution)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.prj_mat = get_prj_mat(intrinsics)
        self.xygrid = torch.unsqueeze(xygrid, dim=0)
        self.act_cond = act_cond
        self.num_context_frames = num_context_frames
        self.splat = splat
        self.alpha_rcr = alpha_rcr
        self.alpha_rcd = alpha_rcd
        self.alpha_knn = alpha_knn
        self.alpha_sfs = alpha_sfs
        self.alpha_ofs = alpha_ofs
        self.alpha_kl = alpha_kl
        self.alpha_l = alpha_l
        self.min_dpt = min_dpt
        self.max_dpt = max_dpt
        self.time_invariant = time_invariant
        self.scale_dpt = ScaleDepthTensor(self.min_dpt, self.max_dpt)
        self.real_dpt = RealDepthTensor(self.min_dpt, self.max_dpt)
        self.stochastic = stochastic
        self.gen_iters = gen_iters
        if self.stochastic:
            self.prior = self.dist.set_unit_dist(self.inference_net.dim_latent)
        self.intrinsics = intrinsics
        self.lpips = LPIPS(net_type="vgg").to(self.device)

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
            dpts: point cloud of time step t.
            rgbs: point cloud of time step t.
            acts: action executed at time step t.
            stts: action executed at time step t.
            inference: action executed at time step t.
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
        dpt_complete = None

        if inference:
            rgb_complete = rgbs
            dpt_complete = dpts

        outputs_cell = {}
        outputs = {
            "s_t": [],
            "s_prime_t": [],
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
                None,
                rgb_complete,
                dpt_complete,
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
        rgb_1: torch.Tensor,
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

        if self.stochastic:
            prior = self.dist.repeat_to_device(self.prior, rgb_t.device, rgb_t.size(0))
            latent_dist = self.dist.get_dist(prior)
            latent_state = prior

            if latent is None:
                # infer posterior distribution q(z|x)
                if rgb_complete is not None:
                    posterior = self.inference_net(rgb_complete, dpt_complete)
                    latent_dist = self.dist.get_dist(posterior)
                    latent_state = posterior
                latent = self.dist.sample_latent_code(latent_dist).to(act_t.device)

        rgbd_t = torch.cat([rgb_t, self.scale_dpt(dpt_t)], dim=1)
        emb_t, obs_lstms = self.obs_encoder(rgbd_t, obs_lstms)
        s_t = emb_t[-1].clone()
        s_prime_t, act_lstms = self.act_encoder(emb_t[-1], act_t, stt_t, latent, act_lstms)
        emb_t[-1] = s_prime_t

        masks_t, _, msk_lstms = self.msk_decoder(emb_t, msk_lstms)
        se3s_t = self.se3_decoder(s_prime_t)

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
            "s_t": s_t,
            "s_prime_t": s_prime_t,
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
        if self.stochastic:
            outputs["mu_t"] = latent_state.mean
            outputs["std_t"] = latent_state.std

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

        out = self(batch["depth_obs"], batch["rgb_obs"], acts, stts, p, inference)
        losses = self.loss(batch, out)
        self.log_loss(losses, mode="train")
        metrics = self.metrics(batch, out)
        self.log_metrics(metrics, mode="train", on_step=True, on_epoch=False)
        return {"loss": losses["loss_total"], "in": batch, "out": out}

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
        metrics = self.metrics(batch, out)
        self.log_metrics(metrics, mode="val", on_step=False, on_epoch=True)
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
        metrics = self.metrics(batch, out)
        self.log_metrics(metrics, mode="test", on_step=False, on_epoch=True)
        return {"loss": losses["loss_total"], "out": out}

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
            self.alpha_l,
            dpt_2,
            ptc_2,
            rgb_1,
            outputs["nxtdpt"],
            outputs["tfmptc_t"],
            outputs["sflow_t"],
        )
        loss_3d = rcd_loss + knn_loss + sfs_loss

        rcr_loss, ofs_loss = calc_2d_loss(
            self.alpha_rcr, self.alpha_ofs, self.alpha_l, rgb_1, rgb_2, outputs["nxtrgb"], outputs["oflow_t"]
        )
        loss_2d = rcr_loss + ofs_loss

        if self.stochastic:
            prior = self.dist.repeat_to_device(
                self.prior, outputs["mu_t"].device, outputs["mu_t"].size(0), outputs["mu_t"].size(1)
            )
            posterior = ContState(outputs["mu_t"], outputs["std_t"])
            loss_kl = calc_kl_loss(self.alpha_kl, self.dist, prior, posterior)

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

    @torch.no_grad()
    def metrics(self, batch: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]):
        true_img, pred_img = batch_seq_view(batch["rgb_obs"][:, 1:]), batch_seq_view(outputs["nxtrgb"])
        true_dpt, pred_dpt = batch_seq_view(batch["depth_obs"][:, 1:]), batch_seq_view(outputs["nxtdpt"])

        ssim = SSIM(pred_img, true_img, data_range=1.0)
        ipsnr = PSNR(pred_img, true_img, data_range=1.0)
        dpsnr = PSNR(pred_dpt, true_dpt, data_range=(self.max_dpt - self.min_dpt))
        spsnr = ipsnr + dpsnr
        pred_img = torch.clamp((pred_img - 0.5) * 2, min=-1, max=1)
        true_img = torch.clamp((true_img - 0.5) * 2, min=-1, max=1)
        lpips = 1 - self.lpips(pred_img, true_img)
        rmse = RMSE(pred_dpt, true_dpt, squared=False)
        mare = MARE(pred_dpt, true_dpt)

        metrics = {
            "metrics_VGG": lpips,
            "metrics_SSIM": ssim,
            "metrics_IPSNR": ipsnr,
            "metrics_DPSNR": dpsnr,
            "metrics_SPSNR": spsnr,
            "metrics_RMSE": rmse,
            "metrics_MARE": mare,
        }

        return metrics
