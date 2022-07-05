import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from t3vip.utils.cam_utils import batch_seq_view


# calculate psnr between two frames
def peak_signal_to_noise_ratio(img1, img2, max_val: float = 1.0):
    """
    Args:
        img1 (Tensor): tensor of shape (batch_size, 3, 240, 320)
        img2 (Tensor): tensor of shape (batch_size, 3, 240, 320)
    Returns:
        tensor of shape (batch_size, )
        psnr ratio of every batch element
    """
    num_pixels = img1.size(-1) * img1.size(-2)
    num_channels = img1.size(-3)

    pred_err = torch.sum((img2 - img1).pow(2), dim=[1, 2, 3])
    pred_err /= num_pixels * num_channels

    return 20 * np.log10(max_val) - 10 * torch.log10(pred_err)


# calculate ssim between two frames
def structural_similarity_index_measure(img1, img2, window_size=11, size_average=False):
    # following functions are from: https://github.com/Po-Hsun-Su/pytorch-ssim
    def gaussian(sigma):
        gauss = torch.Tensor(
            [math.exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2)) for x in range(window_size)]
        )
        return gauss / gauss.sum()

    def create_window():
        _1D_window = gaussian(1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window

    def _ssim():
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    (_, channel, _, _) = img1.size()
    window = create_window()
    window = window.type_as(img1)

    return _ssim()


def compute_rgb_metrics(true_img, pred_img):

    size = true_img.size()
    true_img = batch_seq_view(true_img, size)
    pred_img = batch_seq_view(pred_img, size)

    psnr_score = peak_signal_to_noise_ratio(pred_img, true_img)
    ssim_score = structural_similarity_index_measure(pred_img, true_img)

    psnr_avg = torch.mean(psnr_score, dim=0)
    ssim_avg = torch.mean(ssim_score, dim=0)

    return psnr_avg, ssim_avg


def compute_frame_metrics(true_img, pred_img):
    with torch.no_grad():
        true_img = true_img[:, 1:]
        psnr_avg, ssim_avg = compute_rgb_metrics(true_img, pred_img)

        return psnr_avg, ssim_avg

        # if pred_dpt is None:
        #     return psnr_avg, ssim_avg, vgg_avg
        # else:
        #     true_dpt = true_dpt[:, 1:]
        #     (
        #         dpt_psnr_avg,
        #         dpt_ssim_avg,
        #         dpt_invl1,
        #         dpt_rmse,
        #         dpt_avglog,
        #         dpt_pixbelther,
        #     ) = compute_dpt_metrics(true_dpt, pred_dpt, max_val)
        #     return (
        #         psnr_avg,
        #         ssim_avg,
        #         vgg_avg,
        #         dpt_psnr_avg,
        #         dpt_ssim_avg,
        #         dpt_invl1,
        #         dpt_rmse,
        #         dpt_avglog,
        #         dpt_pixbelther,
        #     )
