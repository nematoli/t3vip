import os
import re
import hydra
import csv
import torch
import cv2
import numpy as np
from PIL import Image
from typing import List, Dict
from omegaconf import OmegaConf
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from pathlib import Path


def count_frames(path, prefix=""):
    regexp = "{0}\d+.png".format(prefix)
    file_names = os.listdir(path)
    num_files = len(file_names)

    matches = [re.fullmatch(regexp, file_names[i]) for i in range(num_files)]
    num_matches = sum([matches[i] is not None for i in range(len(matches))])

    return num_matches


# Read camera data file
def read_intrinsics_file(filename):
    try:
        # Read lines in the file
        ret = {}
        with open(filename, "rt") as csvfile:
            spamreader = csv.reader(csvfile, delimiter=" ")
            label = next(spamreader)
            data = next(spamreader)
            assert len(label) == len(data)
            for k in range(len(label)):
                ret[label[k]] = float(data[k])
        return ret
    except (FileNotFoundError, OSError) as e:
        raise Exception(f"Could not read intrinsic file: {filename}") from e


# Helper functions for perspective projection stuff
def cam_xygrid(height, width, intrinsics):
    assert height > 1 and width > 1
    xygrid = torch.ones(2, height, width)  # (x,y,1)
    for j in range(width):  # +x is increasing columns
        xygrid[0, :, j].fill_((j - intrinsics["cx"]) / intrinsics["fx"])
    for i in range(height):  # +y is increasing rows
        xygrid[1, i, :].fill_((i - intrinsics["cy"]) / intrinsics["fy"])
    return xygrid


def read_calvin_intrinsics(filename):
    # This is used for Calvin dataset
    # NOTE: Assuming that height == width
    try:
        conf = OmegaConf.load(filename)["cameras"]["static"]
        fov = conf["fov"]
        height = conf["height"]
        center = height // 2
        foc = height / (2 * np.tan(np.deg2rad(fov) / 2))
        intrinsics = {
            "fx": foc,
            "fy": foc,
            "cx": center,
            "cy": center,
            "offx": 0,
            "offy": 0,
            "sx": 1,
            "sy": 1,
        }
        return intrinsics
    except (FileNotFoundError, OSError) as e:
        raise Exception(f"Could not read yaml intrinsic file: {filename}") from e


def get_intrinsics(cfg, dataset_name):
    load_dir = Path(cfg.dataset.data_dir).expanduser()
    if dataset_name == "CalvinDataset":

        if cfg.dataset.env != "env_d":
            yaml_file = load_dir / "task_ABC_D" / "validation" / ".hydra" / "merged_config.yaml"
        else:
            yaml_file = load_dir / "task_D_D" / "validation" / ".hydra" / "merged_config.yaml"
        print("load_dir", yaml_file)

        intrinsics = read_calvin_intrinsics(yaml_file)
    else:
        print("load_dir", load_dir)
        intrinsics = read_intrinsics_file(load_dir.joinpath("intrinsics.txt"))
    # Setup camera intrinsics
    img_scale = cfg.resolution / cfg.dataset.img_ht
    cfg.dataset.img_ht = cfg.dataset.img_wd = cfg.resolution
    cam_intrinsics = {
        "fx": img_scale * intrinsics["fx"],
        "fy": img_scale * intrinsics["fy"],
        "cx": img_scale * intrinsics["cx"],
        "cy": img_scale * intrinsics["cy"],
        "offx": intrinsics["offx"],
        "offy": intrinsics["offy"],
        "sx": intrinsics["sx"],
        "sy": intrinsics["sy"],
    }
    print(
        "Intrinsics => ht: {}, wd: {}, fx: {}, fy: {}, cx: {}, cy: {}, offx: {}, offy: {}, sx: {}, sy: {}".format(
            cfg.dataset.img_ht,
            cfg.dataset.img_wd,
            cam_intrinsics["fx"],
            cam_intrinsics["fy"],
            cam_intrinsics["cx"],
            cam_intrinsics["cy"],
            cam_intrinsics["offx"],
            cam_intrinsics["offy"],
            cam_intrinsics["sx"],
            cam_intrinsics["sy"],
        )
    )
    # Compute intrinsic grid
    xygrid = cam_xygrid(cfg.dataset.img_ht, cfg.dataset.img_wd, cam_intrinsics)
    return cam_intrinsics, xygrid


def get_transforms(transforms_list: List[dict] = None):
    map_interp_mode = {
        "bilinear": InterpolationMode.BILINEAR,
        "nearest": InterpolationMode.NEAREST,
    }
    inst_transf_list = []
    for transform in transforms_list:
        if "interpolation" in transform:
            transform = OmegaConf.to_container(transform)
            transform["interpolation"] = map_interp_mode[transform["interpolation"]]
        inst_transf_list.append(hydra.utils.instantiate(transform))
    return transforms.Compose(inst_transf_list)


def load_rgbs(path, idx_start=1, idx_stop=3, idx_step=1, prefix="rgbsub"):
    frames = []

    for i in range(idx_start, idx_stop, idx_step):
        file_name = prefix + str(i) + ".png"
        file_path = os.path.join(path, file_name)

        frame = Image.open(file_path)
        frames += [frame]

    return frames


def load_depths(path, far_val, idx_start=1, idx_stop=3, idx_step=1, prefix="depthsub"):
    dpts = []

    for i in range(idx_start, idx_stop, idx_step):
        file_name = prefix + str(i) + ".png"
        file_path = os.path.join(path, file_name)

        dpt = cv2.imread(file_path, 2).astype(np.uint16)
        dpt = dpt / (2**16 - 1)
        dpt = dpt * far_val  # multiplied by far value
        dpt = Image.fromarray(dpt)
        dpts += [dpt]

    return dpts


def load_joints(path, idx_start, idx_stop, idx_step, prefix="joints", *args, **kwargs):
    joints_video = []

    for i in range(idx_start, idx_stop, idx_step):
        file_name = prefix + str(i) + ".txt"
        file_path = os.path.join(path, file_name)

        joints = np.loadtxt(file_path, *args, **kwargs)

        joints_video += [joints]

    return joints_video


def load_actions(path, idx_start, idx_stop, idx_step, prefix="actions"):
    actions = []
    file_name = prefix + ".npy"
    path_actions = os.path.join(path, file_name)
    actions_file = np.load(path_actions)

    for i in range(idx_start, idx_stop, idx_step):
        actions += [actions_file[i]]

    return actions


def get_ptc_from_dpt(dpt, xygrid):
    if dpt.dim() == 3:
        dpt = dpt.unsqueeze(0)
        xygrid = xygrid.unsqueeze(0)

    B, C, H, W = dpt.shape
    xygrid = xygrid.repeat(B, 1, 1, 1).to(dpt.device)

    ptc = torch.FloatTensor(B, 3, H, W).to(dpt.device)

    ptc[:, 2] = dpt.squeeze(1)

    xy = ptc[:, 0:2]
    xy.copy_(xygrid.expand_as(xy))
    xy.mul_(dpt.repeat(1, 2, 1, 1))

    return ptc


def load_npz(filename: Path) -> Dict[str, np.ndarray]:
    return np.load(filename.as_posix())


# for calvin dataset
def angle_between_angles(a, b):
    diff = b - a
    return (diff + np.pi) % (2 * np.pi) - np.pi


def to_relative_action(robot_obs_1, robot_obs_2, max_pos=0.02, max_orn=0.05):
    assert isinstance(robot_obs_1, np.ndarray)
    assert isinstance(robot_obs_2, np.ndarray)

    rel_pos = robot_obs_2[:, :3] - robot_obs_1[:, :3]
    rel_pos = rel_pos / max_pos

    rel_orn = angle_between_angles(robot_obs_1[:, 3:6], robot_obs_2[:, 3:6])
    rel_orn = rel_orn / max_orn

    gripper = robot_obs_2[:, -1:]
    return np.concatenate([rel_pos, rel_orn, gripper], axis=1)
