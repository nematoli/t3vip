import torch
import torch.nn as nn
import pytorch3d.transforms as t3d
import numpy as np
from typing import Optional, Tuple, List, Union
import torch.jit as jit


# Choose non-linearities
def get_nonlinearity(nonlinearity):
    if nonlinearity == "prelu":
        return nn.PReLU()
    elif nonlinearity == "relu":
        return nn.ReLU(inplace=True)
    elif nonlinearity == "tanh":
        return nn.Tanh()
    elif nonlinearity == "sigmoid":
        return nn.Sigmoid()
    elif nonlinearity == "softplus":
        return nn.Softplus()
    elif nonlinearity == "elu":
        return nn.ELU(inplace=True)
    elif nonlinearity == "selu":
        return nn.SELU()
    elif nonlinearity == "leakyrelu":
        return nn.LeakyReLU(negative_slope=0.1)
    elif nonlinearity == "softmax":
        return nn.Softmax(dim=1)
    elif nonlinearity == "none":
        return lambda x: x  # Return input as is
    else:
        assert False, "Unknown non-linearity: {}".format(nonlinearity)


# Get SE3 dimension based on se3 type & pivot
def get_se3_dimension(se3_type, use_pivot):
    # Get dimension (6 for se3aa, se3euler, se3spquat, se3aar)
    se3_dim = 6
    if se3_type == "se3quat" or se3_type == "se3aa4":
        se3_dim = 7
    elif se3_type == "affine":
        se3_dim = 12
    elif se3_type == "bingham":
        se3_dim = 13
    elif se3_type == "se3six":
        se3_dim = 9
    # Add pivot dimensions
    if use_pivot:
        se3_dim += 3
    return se3_dim


# Initialize the SE3 prediction layer to identity
def init_se3layer_identity(layer, num_se3=8, se3_type="se3aa"):
    layer.weight.data.uniform_(-0.0001, 0.0001)  # Initialize weights to near identity
    layer.bias.data.uniform_(-0.0001, 0.0001)  # Initialize biases to near identity
    # Special initialization for specific SE3 types
    if se3_type == "affine":
        bs = layer.bias.data.view(num_se3, 3, -1)
        bs.narrow(2, 0, 3).copy_(torch.eye(3).view(1, 3, 3).expand(num_se3, 3, 3))
    elif se3_type == "se3quat":
        bs = layer.bias.data.view(num_se3, -1)
        bs.narrow(1, 3, 1).fill_(1)  # ~ [1,0,0,0]
    elif se3_type == "se3six":
        bs = layer.bias.data.view(num_se3, -1)
        bs.narrow(1, 3, 1).fill_(1)  # ~ [1,0,0,0,0,0]
        bs.narrow(1, 7, 1).fill_(1)  # ~ [1,0,0,0,1,0]


def se3_aa_to_mat(se3_6d):
    # input shape BxKx6
    K = se3_6d.shape[1]
    se3_6d = se3_6d.view(-1, 6)
    B, _ = se3_6d.shape
    dtype = se3_6d.dtype
    device = se3_6d.device

    se3mat = torch.zeros(size=(B, 4, 4), dtype=dtype, device=device)

    se3mat[:, :3, :3] = t3d.so3_exponential_map(se3_6d[:, :3])
    se3mat[:, :3, 3] = se3_6d[:, 3:]
    se3mat[:, 3, 3] = 1.0

    se3mat = se3mat.view(-1, K, 4, 4)
    se3mat = se3mat[:, :, :3, :]

    return se3mat


def se3_quat_to_mat(x):
    if x.ndim == 1:
        x = x.unsqueeze(0).unsqueeze(0)
    B, K, L = x.shape
    se3mat = torch.zeros(size=(B, K, 3, 4), dtype=x.dtype, device=x.device)

    quat = x[:, :, 3:]
    se3mat[:, :, :3, :3] = t3d.quaternion_to_matrix(quat)
    se3mat[:, :, :3, 3] = x[:, :, :3]

    return se3mat


def se3_9d_to_mat(x):
    B, K, L = x.shape

    se3mat = torch.zeros(size=(B, K, 3, 4), dtype=x.dtype, device=x.device)

    six_d_rot = x[:, :, 3:]
    se3mat[:, :, :3, :3] = t3d.rotation_6d_to_matrix(six_d_rot)
    se3mat[:, :, :3, 3] = x[:, :, :3]

    return se3mat


def transform_ptc(ptc_t, masks_t, se3s_t):
    B, K, H, W = masks_t.shape
    num_masks = K
    num_se3 = K - 1
    masks_t = masks_t.chunk(num_masks, 1)
    ptc_tf = masks_t[0] * ptc_t
    rot, trn = se3s_t[:, :, :, :3], se3s_t[:, :, :, -1].unsqueeze(3)
    ptc_t = ptc_t.unsqueeze(1).repeat(1, num_se3, 1, 1, 1)
    ptc_transform = torch.matmul(rot, ptc_t.view(B, num_se3, 3, H * W)) + trn
    ptc_transform = ptc_transform.view(B, num_se3, 3, H, W)

    for i in range(num_se3):
        ptc_tf += masks_t[i + 1] * ptc_transform[:, i, :, :, :]

    return ptc_tf


def cdna_convolve(img, kernels):
    """
    Args:
        img (Tensor): tensor of shape (batch_size, 3, 64, 64)
        kernel (Tensor): tensor of shape (batch_size, self.num_kernels, kernel_size, kernel_size)
    """

    batch_size = img.size()[0]
    num_kernels = kernels.size(1)

    img = img.permute(1, 0, 2, 3)
    kernels = kernels.split(1, dim=1)
    img_transforms = []

    for i in range(num_kernels):
        kernel = kernels[i]
        img_transform = torch.conv2d(img, kernel, padding=2, groups=batch_size)
        img_transform = img_transform.permute(1, 0, 2, 3)
        img_transforms.append(img_transform)

    img_transforms = torch.stack(img_transforms, dim=1)

    return img_transforms


def gen_nxtrgb(rgb_t, masks_t, tfmrgb_t, rgb_extra=None):
    B, K, H, W = masks_t.shape
    num_masks = K
    num_knls = K - 1

    masks_t = masks_t.chunk(num_masks, 1)

    nxtrgb = masks_t[0] * rgb_t
    for i in range(num_knls):
        nxtrgb += masks_t[i + 1] * tfmrgb_t[:, i, :, :, :]

    if rgb_extra is not None:
        nxtrgb += masks_t[-1] * rgb_extra
    return nxtrgb


def scheduled_sampling(l_true: List, l_pred: List, num_samples_true: int) -> List:
    """
    Args:
        l_true (List): list of tensors
        l_pred (List): list of tensors
        num_samples_true (int): number of ground truths to sample
    """
    batch_size = l_true[0].size(0)
    l_sampled = []

    idx_shuffle = np.arange(batch_size)
    np.random.shuffle(idx_shuffle)
    idx_reshuffle = np.argsort(idx_shuffle)

    for i in range(len(l_true)):
        x_sampled_true = l_true[i][idx_shuffle[:num_samples_true]]
        x_sampled_pred = l_pred[i][idx_shuffle[num_samples_true:]]

        x_sampled = torch.cat([x_sampled_true, x_sampled_pred], dim=0)
        x_sampled = x_sampled[idx_reshuffle]

        l_sampled.append(x_sampled)

    return l_sampled


def make_layer(layer_modules: List, activation: str, norm: str, **kwargs):
    if norm == "instance" and kwargs["num_features"]:
        layer_modules.append(nn.InstanceNorm2d(kwargs["num_features"]))
    elif norm == "spectral":
        layer_modules[-1] = nn.utils.spectral_norm(layer_modules[-1])
    elif norm == "batch":
        layer_modules.append(nn.BatchNorm2d(kwargs["num_features"], eps=0.001))

    if activation == "relu":
        layer_modules.append(nn.ReLU())
    elif activation == "leaky_relu" and kwargs["negative_slope"]:
        layer_modules.append(nn.LeakyReLU(kwargs["negative_slope"]))
    elif activation == "sigmoid":
        layer_modules.append(nn.Sigmoid())
    elif activation == "softmax":
        layer_modules.append(nn.Softmax(dim=1))

    layer = nn.Sequential(*layer_modules)

    return layer


def create_conv2d(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    padding: int = 0,
    activation: str = "relu",
    norm: str = "instance",
    **kwargs,
):
    """
    Create convolution layer consisting of sub-modules:
        1. convolution 2d
        2. normalization
        3. activation
    """

    layer_modules = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]
    layer = make_layer(layer_modules, activation, norm, num_features=out_channels, **kwargs)

    return layer


def create_deconv2d(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    padding: int = 0,
    activation: str = "relu",
    norm: str = "instance",
    **kwargs,
):
    """
    Create convolution layer consisting of sub-modules:
        1. convolution 2d
        2. normalization
        3. activation
    """

    layer_modules = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)]
    layer = make_layer(layer_modules, activation, norm, num_features=out_channels, **kwargs)

    return layer


class ConvLSTMCell(jit.ScriptModule):
    __constants__ = ["forget_bias", "out_channels"]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        forget_bias: float = 1.0,
        norm: str = None,
        num_groups: int = None,
        layer_dim: int = None,
    ) -> None:
        super(ConvLSTMCell, self).__init__()
        self.out_channels = out_channels
        self.forget_bias = forget_bias

        self.conv = nn.Conv2d(in_channels + out_channels, 4 * out_channels, kernel_size, stride, padding)

        if norm == "batch":
            self.norm = nn.BatchNorm2d(out_channels, eps=0.001)
        elif norm == "instance":
            self.norm = nn.InstanceNorm2d(out_channels, eps=0.001)
        elif norm == "layer":
            # self.norm = nn.GroupNorm(num_groups=1, num_channels=out_channels, eps=0.001)
            self.norm = nn.LayerNorm([out_channels, layer_dim, layer_dim], eps=0.001)
        elif norm == "group":
            if num_groups is None:
                num_groups = int(out_channels / 16)
            self.norm = nn.GroupNorm(num_groups, out_channels, eps=0.001)
        else:
            self.norm = nn.Identity()

    @jit.script_method
    def forward(
        self, x: torch.Tensor, hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size = x.size(0)
        hidden_shape = x.size()[2:]

        if hx is None:
            hx = (
                torch.zeros(
                    batch_size,
                    self.out_channels,
                    hidden_shape[0],
                    hidden_shape[1],
                    dtype=x.dtype,
                    device=x.device,
                ),
                torch.zeros(
                    batch_size,
                    self.out_channels,
                    hidden_shape[0],
                    hidden_shape[1],
                    dtype=x.dtype,
                    device=x.device,
                ),
            )
        h_t, c_t = hx

        x = torch.cat([x, h_t], dim=1)

        gates = self.conv(x)
        i, f, o, c = gates.chunk(4, 1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f + self.forget_bias)
        c_t = f * c_t + i * torch.tanh(c)
        o = torch.sigmoid(o)

        c_t = self.norm(c_t)
        h_t = o * torch.tanh(c_t)

        return h_t, (h_t, c_t)
