import logging
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
from typing import Dict, Union, Tuple
from t3vip.datasets.utils.load_utils import get_transforms

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    """
    Abstract datamodule base class.
    Args:

    """

    def __init__(
        self,
        data_dir: Path,
        img_ht: int,
        img_wd: int,
        dim_action: int,
        dim_state: int,
        far_val: float,
        min_dpt: float,
        max_dpt: float,
        frac_used: float,
        seq_len: int,
        skip_frames: int,
        train: bool,
        transforms: Dict,
        intrinsics: Dict,
        xygrid: torch.Tensor,
        env: str = None,
    ):
        self.data_dir = data_dir
        self.img_ht = img_ht
        self.img_wd = img_wd
        self.dim_action = dim_action
        self.dim_state = dim_state
        self.far_val = far_val
        self.seq_len = seq_len
        self.skip_frames = skip_frames
        self.step_len = self.skip_frames + 1
        self.min_dpt = min_dpt
        self.max_dpt = max_dpt
        self.transforms = transforms
        self.intrinsics = intrinsics
        self.xygrid = xygrid
        self.frac_used = frac_used
        self.train = train
        self.video_paths = []
        self.num_frames = dict()
        self.rgb_prefix = "rgbsub"
        self.dpt_prefix = "depthsub"
        self.env = env

        self.transform_rgb, self.transform_dpt, self.transform_act = None, None, None
        if "rgb" in self.transforms:
            self.transform_rgb = get_transforms(self.transforms.rgb)
        if "depth" in self.transforms:
            self.transform_dpt = get_transforms(self.transforms.depth)
        if "action" in self.transforms:
            self.transform_act = get_transforms(self.transforms.action)

    def __getitem__(self, idx: Union[int, Tuple[int, int]]) -> Dict:

        """
        Get sequence of datamodule.
        Args:
            idx: Index of the sequence.
        Returns:
            Loaded sequence.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        """
        Returns:
            Size of the datamodule.
        """
        # return len(self.episode_lookup)
        raise NotImplementedError
