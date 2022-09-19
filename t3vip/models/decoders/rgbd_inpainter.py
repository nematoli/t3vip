from typing import List
import torch
import torch.nn as nn
from t3vip.utils.net_utils import create_conv2d, create_deconv2d, ConvLSTMCell


class RGBDInp(nn.Module):
    """
    RGBD Inpainter
    Args:
        chn: channel sizes for different layers
        dims: dimensions of conv outputs
        norm: normalization layer after each convolution
        activation: activation layer after each convolution
    """

    def __init__(self, chn: List[int], dims: List[int], norm: str, activation: str):
        super(RGBDInp, self).__init__()

        self.chn = chn
        self.dims = dims
        self.norm = norm
        self.activation = activation

        self.deconv0 = create_deconv2d(
            self.chn[5],
            self.chn[6],
            kernel_size=3,
            stride=2,
            norm=self.norm,
            activation=self.activation,
        )  # 3x3, 8x8 -> 16x16
        self.deconv1 = create_deconv2d(
            self.chn[3] + self.chn[6],
            self.chn[7],
            kernel_size=3,
            stride=2,
            norm=self.norm,
            activation=self.activation,
        )  # 3y3, 16x16 -> 32x32
        self.deconv2 = create_deconv2d(
            self.chn[0] + self.chn[7],
            self.chn[8],
            kernel_size=3,
            stride=2,
            norm=self.norm,
            activation=self.activation,
        )  # 3x3, 32x32 -> 64x64

        self.lstm6 = ConvLSTMCell(self.chn[6], self.chn[6], kernel_size=5, stride=1, padding=2)
        self.lstm7 = ConvLSTMCell(self.chn[7], self.chn[7], kernel_size=5, stride=1, padding=2)

        self.layer_norm_lstm6 = nn.LayerNorm([self.chn[6], self.dims[6], self.dims[6]])
        self.layer_norm_lstm7 = nn.LayerNorm([self.chn[7], self.dims[7], self.dims[7]])
        self.layer_norm_deconv = nn.LayerNorm([self.chn[8], 64, 64])

        self.conv4 = create_conv2d(
            self.chn[7],
            4,
            kernel_size=1,
            stride=1,
            norm=self.norm,
            activation="sigmoid",
        )

    def forward(self, emb, lstm_states=None):
        if lstm_states is None:
            lstm_states = [None] * 2

        enc0, enc1, mixemb = emb

        enc4 = self.deconv0(mixemb)[:, :, 1:, 1:]

        hidden6, lstm_states[0] = self.lstm6(enc4, lstm_states[0])
        hidden6 = self.layer_norm_lstm6(hidden6)
        hidden6 = torch.cat([hidden6, enc1], dim=1)

        enc5 = self.deconv1(hidden6)[:, :, 1:, 1:]

        hidden7, lstm_states[1] = self.lstm7(enc5, lstm_states[1])
        hidden7 = self.layer_norm_lstm7(hidden7)
        hidden7 = torch.cat([hidden7, enc0], dim=1)

        enc6 = self.deconv2(hidden7)[:, :, 1:, 1:]
        enc6 = self.layer_norm_deconv(enc6)

        inp_rgbd = self.conv4(enc6)
        inp_rgb = inp_rgbd[:, :3, :, :]
        inp_dpt = inp_rgbd[:, 3, :, :].unsqueeze(1)

        return inp_rgb, inp_dpt, lstm_states
