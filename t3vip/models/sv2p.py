import logging
from typing import Dict, Union, Any, List
import hydra
from omegaconf import DictConfig
import torch
from t3vip.models.video import VideoModel

from t3vip.utils.net_utils import gen_nxtrgb, scheduled_sampling
from t3vip.helpers.losses import calc_2d_loss, calc_kl_loss
from t3vip.utils.distributions import ContState
from t3vip.utils.cam_utils import batch_seq_view
from torchmetrics.functional import peak_signal_noise_ratio as PSNR
from torchmetrics.functional import structural_similarity_index_measure as SSIM
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS

logger = logging.getLogger(__name__)


class SV2P(VideoModel):
    """
    The lightning module used for training self-supervised t3vip.
    Args:
        obs_encoder: DictConfig for ptc_encoder.
        act_encoder: DictConfig for act_encoder.
        msk_decoder: DictConfig for msk_encoder.
        knl_decoder: DictConfig for se3_decoder.
        inference_net: DictConfig for rgbd_inpainter.
        optimizer: DictConfig for optimizer.
        lr_scheduler: DictConfig for learning rate scheduler.
    """

    def __init__(
        self,
        obs_encoder: DictConfig,
        act_encoder: DictConfig,
        msk_decoder: DictConfig,
        knl_decoder: DictConfig,
        inference_net: DictConfig,
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
        num_priors: int,
        gen_iters: int,
    ):
        super(SV2P, self).__init__()
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
        if self.stochastic:
            self.prior = self.dist.set_unit_dist(self.inference_net.dim_latent)
            self.num_priors = num_priors
        self.lpips = LPIPS(net_type="vgg").to(self.device)
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.optimizer, params=self.parameters())
        return {"optimizer": optimizer}

    def forward(
        self, dpts: torch.Tensor, rgbs: torch.Tensor, acts: torch.Tensor, stts: torch.Tensor, inference: bool, p: float
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

        if inference:
            rgb_complete = rgbs

        rgb_1 = rgbs[:, 0]

        outputs_cell = {}
        outputs = {
            "s_t": [],
            "s_prime_t": [],
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
                None,
                act_t,
                stt_t,
                rgb_1,
                rgb_complete,
                None,
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
        else:
            lstm_states = [None] * 7
            obs_lstms, act_lstms, msk_lstms = (None, None, None)

        if self.stochastic:
            prior = self.dist.repeat_to_device(self.prior, rgb_t.device, rgb_t.size(0))
            latent_dist = self.dist.get_dist(prior)
            latent_state = prior

            if latent is None:
                # infer posterior distribution q(z|x)
                if rgb_complete is not None:
                    posterior = self.inference_net(rgb_complete)
                    latent_dist = self.dist.get_dist(posterior)
                    latent_state = posterior
                latent = self.dist.sample_latent_code(latent_dist).to(act_t.device)

        emb_t, obs_lstms = self.obs_encoder(rgb_t, obs_lstms)
        s_t = emb_t[-1].clone()
        s_prime_t, act_lstms = self.act_encoder(emb_t[-1], act_t, stt_t, latent, act_lstms)
        emb_t[-1] = s_prime_t

        masks_t, _, msk_lstms = self.msk_decoder(emb_t, msk_lstms)
        tfmrgb_t = self.knl_decoder(emb_t[-1], rgb_t)

        rgb_extra = rgb_1 if self.reuse_first_rgb else None
        nxt_rgb = gen_nxtrgb(rgb_t, masks_t, tfmrgb_t, rgb_extra)

        outputs = {
            "s_t": s_t,
            "s_prime_t": s_prime_t,
            "masks_t": masks_t,
            "nxtrgb": nxt_rgb,
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

        out = self(None, batch["rgb_obs"], acts, stts, inference, p)
        losses = self.loss(batch, out)
        self.log_loss(losses, mode="train")
        metrics = self.metrics(batch, out)
        self.log_metrics(metrics, mode="train", on_step=True, on_epoch=False)
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
        out = self(None, batch["rgb_obs"], acts, stts, inference, p)
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
        inference = False
        p = 0.0
        assert batch["rgb_obs"].shape[0] == 1
        if self.stochastic:
            priors_metrics = []
            outs = []
            for i in range(self.num_priors):
                out = self(None, batch["rgb_obs"], acts, stts, inference, p)
                m = self.metrics(batch, out)
                outs.append(out)
                priors_metrics.append(m)
            metrics = max(priors_metrics, key=lambda x: x["metrics_VGG"])
            out = outs[priors_metrics.index(metrics)]
        else:
            out = self(None, batch["rgb_obs"], acts, stts, inference, p)
            metrics = self.metrics(batch, out)

        self.log_metrics(metrics, mode="test", on_step=False, on_epoch=True)
        return {"out": out, "metrics": metrics}

    def loss(self, batch: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        rgb_1, rgb_2 = batch["rgb_obs"][:, :-1], batch["rgb_obs"][:, 1:]

        rcr_loss, _ = calc_2d_loss(self.alpha_rcr, 0, self.alpha_l, rgb_1, rgb_2, outputs["nxtrgb"], None)

        if self.stochastic:
            prior = self.dist.repeat_to_device(
                self.prior, outputs["mu_t"].device, outputs["mu_t"].size(0), outputs["mu_t"].size(1)
            )
            posterior = ContState(outputs["mu_t"], outputs["std_t"])
            loss_kl = calc_kl_loss(self.alpha_kl, self.dist, prior, posterior)
        else:
            loss_kl = torch.tensor(0.0).to(self.device)

        total_loss = rcr_loss + loss_kl

        losses = {
            "loss_total": total_loss,
            "loss2d_rgbrcs": rcr_loss,
            "loss_kl": loss_kl,
        }

        return losses

    @torch.no_grad()
    def metrics(self, batch: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]):
        true_img, pred_img = batch_seq_view(batch["rgb_obs"][:, 1:]), batch_seq_view(outputs["nxtrgb"])

        ssim = SSIM(pred_img, true_img, data_range=1.0)
        ipsnr = PSNR(pred_img, true_img, data_range=1.0)
        pred_img = torch.clamp((pred_img - 0.5) * 2, min=-1, max=1)
        true_img = torch.clamp((true_img - 0.5) * 2, min=-1, max=1)
        lpips = 1 - self.lpips(pred_img, true_img)

        metrics = {"metrics_VGG": lpips, "metrics_SSIM": ssim, "metrics_IPSNR": ipsnr}

        return metrics
