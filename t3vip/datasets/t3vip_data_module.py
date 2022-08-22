import logging
from pathlib import Path
from typing import Dict, List

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import t3vip
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class T3VIPDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: DictConfig,
        batch_size: int,
        num_workers: int,
        transforms: Dict,
        resolution: int,
        intrinsics: Dict,
        xygrid: torch.Tensor,
        seq_len: int,
        skip_frames: int,
        **kwargs: Dict,
    ):
        super().__init__()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.dataset = dataset
        root_data_path = Path(self.dataset.data_dir).expanduser()
        if not root_data_path.is_absolute():
            root_data_path = Path(t3vip.__file__).parent / root_data_path

        dataset_name = self.dataset["_target_"].split(".")[-1]
        if "calvin" in dataset_name.lower():
            if self.dataset.env != "env_d":
                self.train_dir = self.val_dir = root_data_path / "task_ABC_D" / "training"
            else:
                self.train_dir = root_data_path / "task_D_D" / "training"
                self.val_dir = root_data_path / "task_D_D" / "validation"

            self.train_episodes_info = root_data_path / "task_idx" / self.dataset.env / "training"
            self.val_episodes_info = root_data_path / "task_idx" / self.dataset.env / "validation"

            self.test_dir = root_data_path / "task_D_D" / "validation"
            self.test_episodes_info = root_data_path / "task_idx" / "env_d" / "validation"

        else:
            self.train_dir = root_data_path / "training"
            self.val_dir = root_data_path / "validation"
            self.test_dir = root_data_path / "test"
            self.train_episodes_info = self.val_episodes_info = self.test_episodes_info = None

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resolution = resolution
        self.transforms = transforms
        self.intrinsics = intrinsics
        self.xygrid = xygrid
        self.skip_frames = skip_frames
        self.seq_len = seq_len

    def setup(self, stage=None):
        self.train_dataset = hydra.utils.instantiate(
            self.dataset,
            data_dir=self.train_dir,
            img_ht=self.resolution,
            img_wd=self.resolution,
            seq_len=self.seq_len,
            skip_frames=self.skip_frames,
            train=True,
            transforms=self.transforms,
            intrinsics=self.intrinsics,
            xygrid=self.xygrid,
            ep_info=self.train_episodes_info,
        )
        self.val_dataset = hydra.utils.instantiate(
            self.dataset,
            data_dir=self.val_dir,
            img_ht=self.resolution,
            img_wd=self.resolution,
            seq_len=self.seq_len,
            skip_frames=self.skip_frames,
            train=False,
            transforms=self.transforms,
            intrinsics=self.intrinsics,
            xygrid=self.xygrid,
            ep_info=self.val_episodes_info,
        )
        self.test_dataset = hydra.utils.instantiate(
            self.dataset,
            data_dir=self.test_dir,
            img_ht=self.resolution,
            img_wd=self.resolution,
            seq_len=self.seq_len,
            skip_frames=self.skip_frames,
            train=False,
            transforms=self.transforms,
            intrinsics=self.intrinsics,
            xygrid=self.xygrid,
            ep_info=self.test_episodes_info,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            shuffle=False,
            drop_last=False,
        )

    def subset_dataloader(self, skip=10):
        idx = list(range(0, len(self.train_dataset), skip))
        sub_dataset = torch.utils.data.Subset(self.train_dataset, idx)
        return DataLoader(
            sub_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            shuffle=True,
            drop_last=True,
        )
