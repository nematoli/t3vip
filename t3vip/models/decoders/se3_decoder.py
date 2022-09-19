from typing import List
import torch
import torch.nn as nn
from t3vip.utils.net_utils import (
    get_se3_dimension,
    get_nonlinearity,
    init_se3layer_identity,
    se3_aa_to_mat,
    se3_quat_to_mat,
    se3_9d_to_mat,
)


class SE3Dec(nn.Module):
    """
    SE3 Transformation Decoder
    Args:
        chn: channel sizes for different layers
        dims: dimensions of conv outputs
        num_masks: number of predicted masks
        se3_type: type of se3
        activation: activation layer after each convolution
    """

    def __init__(self, chn: List[int], dims: List[int], num_masks: int, se3_type: str, activation: str):
        super(SE3Dec, self).__init__()

        self.chn = chn
        self.dims = dims
        self.num_se3 = num_masks - 1
        self.se3_type = se3_type
        self.se3_dim = get_se3_dimension(se3_type=self.se3_type, use_pivot=False)

        self.fc = nn.Sequential(
            nn.Linear(self.chn[5] * self.dims[5] * self.dims[5], 256),
            get_nonlinearity(activation),
            nn.Linear(256, 128),
            get_nonlinearity(activation),
            nn.Linear(128, self.num_se3 * self.se3_dim),
        )
        self.last_layer = self.fc[4]  # Get final SE3 prediction module

        init_se3layer_identity(self.last_layer, self.num_se3, self.se3_type)

    def forward(self, emb_ta):
        """
        SE3 decoder
        Args:
            emb_ta: action_conditioned embedding of time step t
        Returns:
            se3s_t: K-1 3x4 rotation-translation matrices
        """
        se3_tfms = self.fc(torch.flatten(emb_ta, start_dim=1))
        se3_tfms = se3_tfms.view(-1, self.num_se3, self.se3_dim)

        if self.se3_type == "se3aa":
            se3s_t = se3_aa_to_mat(se3_tfms)
        elif self.se3_type == "se3quat":
            se3_tfms = se3_tfms / se3_tfms.norm(dim=2, keepdim=True)
            se3s_t = se3_quat_to_mat(se3_tfms)
        else:
            se3s_t = se3_9d_to_mat(se3_tfms)

        return se3s_t
