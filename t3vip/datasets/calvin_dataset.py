import logging
import os
import re
from pathlib import Path
import math
import numpy as np
import torch
from t3vip.datasets.base_dataset import BaseDataset
from typing import Dict, List, Tuple, Union, Callable
from t3vip.datasets.utils.load_utils import get_ptc_from_dpt, load_npz, to_relative_action

logger = logging.getLogger(__name__)


class CalvinDataset(BaseDataset):
    def __init__(self, ep_info, *args, **kwargs):
        super(CalvinDataset, self).__init__(*args, **kwargs)
        self.episode_lookup = self.load_file_indices(ep_info)
        self.naming_pattern, self.n_digits = self.lookup_naming_pattern()

    def __len__(self):
        """
        returns
        ----------
        number of possible starting frames
        """
        self.num_videos = len(self.episode_lookup) - (self.seq_len * self.step_len - (self.step_len - 1))

        return self.num_videos

    def __getitem__(self, idx: Union[int, Tuple[int, int]]) -> Dict:

        window_size = self.seq_len
        return self.get_sequences(idx, window_size)

    def lookup_naming_pattern(self):
        it = os.scandir(self.data_dir)
        while True:
            filename = Path(next(it))
            if "npz" in filename.suffix:
                break
        aux_naming_pattern = re.split(r"\d+", filename.stem)
        naming_pattern = [filename.parent / aux_naming_pattern[0], filename.suffix]
        n_digits = len(re.findall(r"\d+", filename.stem)[0])
        assert len(naming_pattern) == 2
        assert n_digits > 0
        return naming_pattern, n_digits

    def get_episode_name(self, idx: int) -> Path:
        """
        Convert frame idx to file name
        """
        return Path(f"{self.naming_pattern[0]}{idx:0{self.n_digits}d}{self.naming_pattern[1]}")

    def zip_sequence(self, start_idx: int, end_idx: int) -> Dict[str, np.ndarray]:
        """
        Load consecutive individual frames saved as npy files and combine to episode dict
        parameters:
        -----------
        start_idx: index of first frame
        end_idx: index of last frame
        returns:
        -----------
        episode: dict of numpy arrays containing the episode where keys are the names of modalities
        """
        episodes = [load_npz(self.get_episode_name(file_idx)) for file_idx in range(start_idx, end_idx, self.step_len)]
        episode = {key: np.stack([ep[key] for ep in episodes]) for key, _ in episodes[0].items()}
        return episode

    def get_sequences(self, idx: int, window_size: int) -> Dict:
        """
        parameters
        ----------
        idx: index of starting frame
        window_size:    length of sampled episode
        returns
        ----------
        seq_state_obs:  numpy array of state observations
        seq_rgb_obs:    tuple of numpy arrays of rgb observations
        seq_depth_obs:  tuple of numpy arrays of depths observations
        seq_acts:       numpy array of actions
        """
        start_file_indx = self.episode_lookup[idx]
        end_file_indx = start_file_indx + (window_size * self.step_len - (self.step_len - 1))

        episode = self.zip_sequence(start_file_indx, end_file_indx)
        dpts = [self.transform_dpt(dpt) for dpt in episode["depth_static"]]
        rgbs = [self.transform_rgb(rgb) for rgb in episode["rgb_static"]]
        if self.step_len > 1:
            actions = to_relative_action(episode["robot_obs"][:-1, :7], episode["robot_obs"][1:, :7])
            actions = [self.transform_act(act) for act in actions]
        else:
            actions = [self.transform_act(act) for act in episode["rel_actions"][:-1]]

        # Calculate point clouds from depth
        ptcs = [get_ptc_from_dpt(dpt, self.xygrid).squeeze(0) for dpt in dpts]

        ptcs = torch.stack(ptcs)
        dpts = torch.stack(dpts)
        rgbs = torch.stack(rgbs)
        actions = torch.stack(actions)
        batch = {
            "ptc_obs": ptcs,
            "depth_obs": dpts,
            "rgb_obs": rgbs,
            "actions": actions,
        }
        return batch

    def load_file_indices(self, ep_info: Path) -> Tuple[List, List]:
        """
        this method builds the mapping from index to file_name used for loading the episodes
        parameters
        ----------
        ep_info:               absolute path of the directory containing the dataset
        returns
        ----------
        episode_lookup:                 list for the mapping from training example index to episode (file) index
        max_batched_length_per_demo:    list of possible starting indices per episode
        """
        assert ep_info.is_dir()

        episode_lookup = []

        ep_start_end_ids = np.load(ep_info / "ep_start_end_ids.npy")
        # if ep_start_end_ids.ndim == 1:
        #     ep_start_end_ids = np.expand_dims(ep_start_end_ids, axis=0)
        frac_used_idx = np.random.choice(
            a=ep_start_end_ids.shape[0],
            size=math.floor(self.frac_used * ep_start_end_ids.shape[0]),
        )

        ep_start_end_ids = ep_start_end_ids[frac_used_idx]
        logger.info(f'Found "ep_start_end_ids.npy" with {len(ep_start_end_ids)} episodes.')
        for start_idx, end_idx in ep_start_end_ids:
            assert end_idx > self.seq_len * self.step_len
            for idx in range(
                start_idx,
                end_idx + 1 - (self.seq_len * self.step_len - (self.step_len - 1)),
            ):
                episode_lookup.append(idx)
        return episode_lookup
