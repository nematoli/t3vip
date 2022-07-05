import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class KNN(nn.Module):
    def __init__(self, search):
        super().__init__()
        self.search = search

    def forward(self, tfm_range, obs_range):

        B, C, H, W = obs_range.shape

        # check if size of kernel is odd and complain
        if self.search % 2 == 0:
            raise ValueError("Nearest neighbor kernel must be odd number")

        # calculate padding
        pad = int((self.search - 1) / 2)

        # unfold neighborhood to get nearest neighbors for each pixel (range image)
        unfold_obs_ptc = F.unfold(obs_range, kernel_size=(self.search, self.search), padding=(pad, pad)).view(
            B, C, -1, H * W
        )
        unfold_tfm_ptc = tfm_range.view(B, C, H * W).view(B, C, -1, H * W)

        k2_distances = torch.abs(unfold_tfm_ptc - unfold_obs_ptc)
        if C != 1:
            k2_distances = torch.linalg.norm(k2_distances, ord=2, dim=1)
        else:
            k2_distances = k2_distances.view(B, -1, H * W)

        knn_val, knn_idx = torch.topk(k2_distances, 1, dim=1, largest=False, sorted=True)

        return knn_val
