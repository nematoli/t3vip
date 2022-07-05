import logging
import os
import numpy as np
import torch
from t3vip.datasets.base_dataset import BaseDataset
from t3vip.datasets.utils.load_utils import (
    load_depths,
    load_rgbs,
    load_actions,
    get_ptc_from_dpt,
    count_frames,
)

logger = logging.getLogger(__name__)


class OmnipushDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super(OmnipushDataset, self).__init__(*args, **kwargs)

        episode_dirs = os.listdir(self.data_dir)

        for episode_dir in episode_dirs:

            path = os.path.join(self.data_dir, episode_dir)

            num_frames = count_frames(path, prefix=self.rgb_prefix)

            if num_frames >= self.seq_len * self.step_len:
                self.video_paths.append(episode_dir)
                self.num_frames[episode_dir] = num_frames

        self.num_videos = len(self.video_paths)

    def __len__(self):
        return self.num_videos

    def __getitem__(self, idx):
        path_relative = self.video_paths[idx]
        episode_path = os.path.join(self.data_dir, self.video_paths[idx])

        if self.train:
            first_frame = np.random.randint(self.num_frames[path_relative] - self.seq_len * self.step_len + 1)
        else:
            first_frame = 0

        dpts = load_depths(
            episode_path,
            far_val=self.far_val,
            idx_start=first_frame,
            idx_stop=first_frame + self.seq_len * self.step_len,
            idx_step=self.step_len,
        )
        rgbs = load_rgbs(
            episode_path,
            idx_start=first_frame,
            idx_stop=first_frame + self.seq_len * self.step_len,
            idx_step=self.step_len,
        )
        acts = load_actions(
            episode_path,
            idx_start=first_frame,
            idx_stop=first_frame + self.seq_len * self.step_len,
            idx_step=self.step_len,
        )

        if self.transform_dpt:
            dpts = [self.transform_dpt(dpt) for dpt in dpts]
        if self.transform_rgb:
            rgbs = [self.transform_rgb(rgb) for rgb in rgbs]
        if self.transform_act:
            acts = [self.transform_act(act) for act in acts]

        ptcs = [get_ptc_from_dpt(dpt, self.xygrid).squeeze(0) for dpt in dpts]

        ptcs, dpts, rgbs, acts = (
            torch.stack(ptcs).to(torch.float32),
            torch.stack(dpts).to(torch.float32),
            torch.stack(rgbs).to(torch.float32),
            torch.stack(acts).to(torch.float32),
        )

        batch = {
            "ptc_obs": ptcs,
            "depth_obs": dpts,
            "rgb_obs": rgbs,
            "actions": acts,
        }

        return batch
