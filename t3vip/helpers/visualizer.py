import random
import torch
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
import wandb
from t3vip.utils.cam_utils import flow_to_rgb
import numpy as np
from t3vip.utils.running_stats import RunningStats


class PlotCallback(pl.Callback):
    def __init__(
        self,
        vis_imgs=False,
        vis_freq=500,
    ):
        super().__init__()

        self.vis_imgs = vis_imgs
        self.vis_freq = vis_freq

        self.psnr, self.ssim, self.vgg = RunningStats(), RunningStats(), RunningStats()
        self.rmse, self.mare = RunningStats(), RunningStats()

    @torch.no_grad()
    def log_images(self, pl_module, batch, outputs, mode="train"):
        # if (not self.vis_imgs) or pl_module.global_step % self.vis_freq != 0:
        #     return
        if pl_module.global_step % self.vis_freq != 0:
            return
        commit = False if mode == "train" else True
        B, S, K, H, W = outputs["masks_t"].size()
        id = random.randint(0, B - 1)
        if "oflow_t" in outputs:
            flows = [flow_to_rgb(outputs["oflow_t"][:, i].narrow(0, id, 1)) for i in range(S)]
            flows = [transforms.functional.to_tensor(np.moveaxis(flow.squeeze(), 0, -1)) for flow in flows]
            flowdisp = torchvision.utils.make_grid(torch.stack(flows))
            if isinstance(pl_module.logger, WandbLogger):
                pl_module.logger.experiment.log({"OFlow/pred-{}".format(mode): wandb.Image(flowdisp)}, commit=commit)
            elif isinstance(pl_module.logger, TensorBoardLogger):
                pl_module.logger.experiment.add_image("OFlow/pred-{}".format(mode), flowdisp, pl_module.global_step)

        if not self.vis_imgs:
            return

        if "occmap_t" in outputs:
            occmapdisp = torchvision.utils.make_grid(
                torch.cat([outputs["occmap_t"].narrow(0, id, 1)], 0).cpu().view(-1, 1, H, W),
                normalize=True,
                range=(0, 1),
            )

        gt_rgbdisp = torchvision.utils.make_grid(batch["rgb_obs"][id][1:])
        rgbdisp = torchvision.utils.make_grid(outputs["nxtrgb"][id])

        masksdisp = torchvision.utils.make_grid(
            outputs["masks_t"].narrow(0, id, 1).view(-1, 1, H, W),
            nrow=K,
            normalize=False,
        )  # value_range=(0, 1)

        if isinstance(pl_module.logger, WandbLogger):
            if "occmap_t" in outputs:
                pl_module.logger.experiment.log(
                    {
                        "Masks/Occ_{}".format(mode): wandb.Image(occmapdisp),
                    },
                    commit=commit,
                )

            pl_module.logger.experiment.log(
                {"RGBs/gt-{}".format(mode): wandb.Image(gt_rgbdisp)},
                commit=commit,
            )

            pl_module.logger.experiment.log(
                {"RGBs/pred-{}".format(mode): wandb.Image(rgbdisp)},
                commit=commit,
            )

            pl_module.logger.experiment.log(
                {"Masks/{}".format(mode): wandb.Image(masksdisp)},
                commit=commit,
            )

        elif isinstance(pl_module.logger, TensorBoardLogger):
            if "occmap_t" in outputs:
                pl_module.logger.experiment.add_image("Masks/Occ_{}".format(mode), occmapdisp, pl_module.global_step)
            pl_module.logger.experiment.add_image("RGBs/gt-{}".format(mode), gt_rgbdisp, pl_module.global_step)
            pl_module.logger.experiment.add_image("RGBs/pred-{}".format(mode), rgbdisp, pl_module.global_step)
            pl_module.logger.experiment.add_image("Masks/{}".format(mode), masksdisp, pl_module.global_step)

        else:
            raise ValueError

    @torch.no_grad()
    def log_tables(self, pl_module):
        data = [
            ["mean", self.psnr.mean(), self.ssim.mean(), self.vgg.mean()],
            ["std", self.psnr.std(), self.ssim.std(), self.vgg.std()],
        ]
        columns = ["stat", "psnr", "ssim", "vgg"]
        if self.rmse.size() > 0:
            data[0] += [self.rmse.mean(), self.mare.mean()]
            data[1] += [self.rmse.std(), self.mare.std()]
            columns += ["rmse", "mare"]
        pl_module.logger.experiment.log({"Stats": wandb.Table(data=data, columns=columns)})

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, **kwargs):
        self.log_images(pl_module, batch, outputs["out"], mode="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.log_images(pl_module, batch, outputs["out"], mode="val")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.log_images(pl_module, batch, outputs["out"], mode="test")
        self.psnr.push(outputs["metrics"]["metrics_IPSNR"])
        self.ssim.push(outputs["metrics"]["metrics_SSIM"])
        self.vgg.push(outputs["metrics"]["metrics_VGG"])
        if "metrics_RMSE" in outputs["metrics"]:
            self.rmse.push(outputs["metrics"]["metrics_RMSE"])
            self.mare.push(outputs["metrics"]["metrics_MARE"])

    def on_test_epoch_end(self, trainer, pl_module):
        self.log_tables(pl_module)
