from collections import namedtuple
from typing import List
import torch
import torch.nn as nn
from t3vip.utils.transforms import ScaleDepthTensor
from t3vip.utils.net_utils import create_conv2d

ContState = namedtuple("ContState", ["mean", "std"])


class QNet(nn.Module):
    def __init__(
        self,
        chn: List[int],
        dims: List[int],
        input_chn: int,
        dim_latent: int,
        seq_len: int,
        min_logvar: float,
        norm: str,
        activation: str,
        min_dpt: float,
        max_dpt: float,
    ):
        super(QNet, self).__init__()

        self.chn = chn
        self.dims = dims
        self.input_chn = input_chn
        self.dim_latent = dim_latent
        self.seq_len = seq_len
        self.min_logvar = min_logvar
        self.norm = norm
        self.activation = activation
        self.min_dpt = min_dpt
        self.max_dpt = max_dpt
        self.scale_dpt = ScaleDepthTensor(self.min_dpt, self.max_dpt)

        self.conv1 = create_conv2d(
            self.input_chn * self.seq_len,
            self.chn[0],
            kernel_size=3,
            stride=2,
            padding=1,
            norm=self.norm,
            activation=self.activation,
        )

        self.conv2 = create_conv2d(
            self.chn[0],
            self.chn[1],
            kernel_size=3,
            stride=2,
            padding=1,
            norm=self.norm,
            activation=self.activation,
        )

        self.conv3 = create_conv2d(
            self.chn[1],
            self.chn[2],
            kernel_size=3,
            stride=1,
            padding=1,
            norm=self.norm,
            activation=self.activation,
        )

        self.mu = create_conv2d(
            self.chn[2],
            1,
            kernel_size=3,
            stride=2,
            padding=1,
            norm=self.norm,
            activation="None",
        )

        self.logvar = create_conv2d(
            self.chn[2],
            1,
            kernel_size=3,
            stride=2,
            padding=1,
            norm=self.norm,
            activation=self.activation,
        )

        self.layer_norm_conv1 = nn.LayerNorm([self.chn[0], self.dims[0], self.dims[0]])
        self.layer_norm_conv2 = nn.LayerNorm([self.chn[1], self.dims[1], self.dims[1]])
        self.layer_norm_conv3 = nn.LayerNorm([self.chn[2], self.dims[2], self.dims[2]])
        self.layer_norm_mu = nn.LayerNorm([1, self.dim_latent, self.dim_latent])
        self.layer_norm_logvar = nn.LayerNorm([1, self.dim_latent, self.dim_latent])

    def forward(self, rgbs, dpts=None):

        if dpts is not None:
            x = torch.cat([rgbs, self.scale_dpt(dpts)], dim=2)
        else:
            x = rgbs

        B, S, C, H, W = x.size()
        x = x.view(B, S * C, H, W)

        x = self.conv1(x)
        x = self.layer_norm_conv1(x)

        x = self.conv2(x)
        x = self.layer_norm_conv2(x)

        x = self.conv3(x)
        x = self.layer_norm_conv3(x)

        mu = self.mu(x)
        mu = self.layer_norm_mu(mu)

        logvar = self.logvar(x)
        logvar = self.layer_norm_logvar(logvar)
        logvar = logvar + self.min_logvar

        stdev = torch.exp(logvar / 2.0)

        state = ContState(mu, stdev)

        return state
