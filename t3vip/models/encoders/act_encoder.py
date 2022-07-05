from typing import List
import torch
import torch.nn as nn
from t3vip.utils.net_utils import create_conv2d, ConvLSTMCell


class ActEnc(nn.Module):
    """
    Point Cloud encoder
    Args:
        chn: channel sizes for different layers.
        norm: normalization layer after each convolution
        activation: activation layer after each convolution
    """

    def __init__(
        self,
        chn: List[int],
        dims: List[int],
        dim_action: int,
        dim_state: int,
        dim_latent: int,
        norm: str,
        activation: str,
    ):
        super(ActEnc, self).__init__()

        self.chn = chn
        self.dims = dims
        self.dim_action = dim_action
        self.dim_state = dim_state
        self.dim_latent = dim_latent
        self.norm = norm
        self.activation = activation
        self.dim_action_state = dim_action + dim_state
        self.input_size = dim_action + dim_state + dim_latent

        self.conv3 = create_conv2d(
            self.chn[5] + self.input_size,
            self.chn[5],
            kernel_size=1,
            stride=1,
            norm=self.norm,
            activation=self.activation,
        )  # 1x1, 8x8 -> 8x8

        self.lstm5 = ConvLSTMCell(self.chn[5], self.chn[5], kernel_size=5, stride=1, padding=2)
        self.layer_norm_lstm5 = nn.LayerNorm([self.chn[5], self.dims[5], self.dims[5]])

    def forward(self, enc_t, action=None, state=None, latent=None, lstm_states=None):

        if lstm_states is None:
            lstm_states = [None] * 1

        batch_size = enc_t.size(0)

        if self.dim_action > 0 and self.dim_state > 0:
            action_state = torch.cat([action, state], dim=1)
        elif self.dim_action > 0:
            action_state = action
        elif self.dim_state > 0:
            action_state = state
        else:
            action_state = None

        # tile action-state to DIM_LSTM5 x DIM_LSTM5 response map
        if action_state is not None:
            action_state_map = torch.ones(
                batch_size,
                self.dim_action_state,
                self.dims[5],
                self.dims[5],
                device=action.device,
            )
            action_state_map *= action_state[:, :, None, None]
            if latent is not None:
                action_state_map = torch.cat([action_state_map, latent], dim=1)

        else:
            action_state_map = latent

        if action_state_map is not None:
            enc_t = torch.cat([enc_t, action_state_map], dim=1)

        enc_tc = self.conv3(enc_t)

        hidden5, lstm_states[0] = self.lstm5(enc_tc, lstm_states[0])
        hidden5 = self.layer_norm_lstm5(hidden5)

        return hidden5, lstm_states
