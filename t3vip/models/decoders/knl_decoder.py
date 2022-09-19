from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from t3vip.utils.net_utils import cdna_convolve


class KnlDec(nn.Module):
    """
    Convolution Kernal Decoder
    Args:
        chn: channel sizes for different layers
        dims: dimensions of conv outputs
        num_masks: number of predicted masks
    """

    def __init__(self, chn: List[int], dims: List[int], num_masks: int):
        super(KnlDec, self).__init__()

        self.chn = chn
        self.dims = dims
        self.num_kernels = num_masks - 1
        self.KERNEL_CDNA = 5
        self.RELU_SHIFT = 1e-5

        self.fc = nn.Linear(
            self.chn[5] * self.dims[5] * self.dims[5],
            self.num_kernels * self.KERNEL_CDNA**2,
        )

    def forward(self, emb, rgb):

        # compute cdna kernels & transformed image
        cdna_kernels = self.fc(torch.flatten(emb, start_dim=1))
        cdna_kernels = F.relu(cdna_kernels - self.RELU_SHIFT) + self.RELU_SHIFT
        cdna_kernels = cdna_kernels.view(-1, self.num_kernels, self.KERNEL_CDNA, self.KERNEL_CDNA)
        norm_factor = torch.sum(cdna_kernels, dim=[2, 3])[:, :, None, None]
        cdna_kernels /= norm_factor

        rgb_transform = cdna_convolve(rgb, cdna_kernels)

        return rgb_transform
