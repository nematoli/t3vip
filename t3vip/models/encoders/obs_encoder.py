from typing import List
import torch.nn as nn
from t3vip.utils.net_utils import create_conv2d, ConvLSTMCell


class ObsEnc(nn.Module):
    """
    Observatoin Encoder
    Args:
        chn: channel sizes for different layers
        dims: dimensions of conv outputs
        input_chn: number of input channels (3 for RGB and 4 for RGBD)
        norm: normalization layer after each convolution
        activation: activation layer after each convolution
    """

    def __init__(self, chn: List[int], dims: List[int], input_chn: int, norm: str, activation: str):
        super(ObsEnc, self).__init__()

        self.chn = chn
        self.dims = dims

        self.input_chn = input_chn
        self.norm = norm
        self.activation = activation

        self.conv0 = create_conv2d(
            self.input_chn,
            self.chn[0],
            kernel_size=5,
            stride=2,
            padding=2,
            norm=self.norm,
            activation=self.activation,
        )  # 5x5, 64x64 -> 32x32
        self.conv1 = create_conv2d(
            self.chn[2],
            self.chn[3],
            kernel_size=3,
            stride=2,
            padding=1,
            norm=self.norm,
            activation=self.activation,
        )  # 3x3, 32x32 -> 16x16
        self.conv2 = create_conv2d(
            self.chn[4],
            self.chn[5],
            kernel_size=3,
            stride=2,
            padding=1,
            norm=self.norm,
            activation=self.activation,
        )  # 3x3, 16x16 -> 8x8

        self.lstm1 = ConvLSTMCell(self.chn[1], self.chn[1], kernel_size=5, stride=1, padding=2)
        self.lstm2 = ConvLSTMCell(self.chn[2], self.chn[2], kernel_size=5, stride=1, padding=2)
        self.lstm3 = ConvLSTMCell(self.chn[3], self.chn[3], kernel_size=5, stride=1, padding=2)
        self.lstm4 = ConvLSTMCell(self.chn[4], self.chn[4], kernel_size=5, stride=1, padding=2)

        self.layer_norm_conv1 = nn.LayerNorm([self.chn[0], self.dims[0], self.dims[0]])
        self.layer_norm_lstm1 = nn.LayerNorm([self.chn[1], self.dims[1], self.dims[1]])
        self.layer_norm_lstm2 = nn.LayerNorm([self.chn[2], self.dims[2], self.dims[2]])
        self.layer_norm_lstm3 = nn.LayerNorm([self.chn[3], self.dims[3], self.dims[3]])
        self.layer_norm_lstm4 = nn.LayerNorm([self.chn[4], self.dims[4], self.dims[4]])

        self.latent_size = self.chn[5] * 8 * 8

    def forward(self, rgb, lstm_states=None):
        if lstm_states is None:
            lstm_states = [None] * 4

        enc0 = self.conv0(rgb)
        enc0 = self.layer_norm_conv1(enc0)

        hidden1, lstm_states[0] = self.lstm1(enc0, lstm_states[0])
        hidden1 = self.layer_norm_lstm1(hidden1)

        hidden2, lstm_states[1] = self.lstm2(hidden1, lstm_states[1])
        hidden2 = self.layer_norm_lstm2(hidden2)

        enc1 = self.conv1(hidden2)

        hidden3, lstm_states[2] = self.lstm3(enc1, lstm_states[2])
        hidden3 = self.layer_norm_lstm3(hidden3)

        hidden4, lstm_states[3] = self.lstm4(hidden3, lstm_states[3])
        hidden4 = self.layer_norm_lstm4(hidden4)

        enc2 = self.conv2(hidden4)

        emb = [enc0, enc1, enc2]

        return emb, lstm_states
