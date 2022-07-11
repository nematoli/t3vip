import os
from pathlib import Path
from typing import Dict, List, Union

import git
import numpy as np
import pytorch_lightning
from pytorch_lightning.utilities.cloud_io import load as pl_load
import torch
import tqdm
import t3vip.models.t3vip as m_t3vip
import t3vip.models.sv2p as m_sv2p


def get_git_commit_hash(repo_path: Path) -> str:
    repo = git.Repo(search_parent_directories=True, path=repo_path.parent)
    assert repo, "not a repo"
    changed_files = [item.a_path for item in repo.index.diff(None)]
    if changed_files:
        print("WARNING uncommitted modified files: {}".format(",".join(changed_files)))
    return repo.head.object.hexsha


def info_packages() -> Dict[str, str]:
    return {
        "numpy": np.__version__,
        "pyTorch_version": torch.__version__,
        "pyTorch_debug": str(torch.version.debug),
        "pytorch-lightning": pytorch_lightning.__version__,
        "tqdm": tqdm.__version__,
    }


def get_all_checkpoints(experiment_folder: Path) -> List:
    if experiment_folder.is_dir():
        checkpoint_folder = experiment_folder / "saved_models"
        if checkpoint_folder.is_dir():
            checkpoints = sorted(Path(checkpoint_folder).iterdir(), key=lambda chk: chk.stat().st_mtime)
            if len(checkpoints):
                return [chk for chk in checkpoints if chk.suffix == ".ckpt"]
    return []


def get_last_checkpoint(experiment_folder: Path) -> Union[Path, None]:
    # return newest checkpoint according to creation time
    checkpoints = get_all_checkpoints(experiment_folder)
    if len(checkpoints):
        return checkpoints[-1]
    return None


def info_cuda() -> Dict[str, Union[str, List[str]]]:
    return {
        "GPU": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        # 'nvidia_driver': get_nvidia_driver_version(run_lambda),
        "available": str(torch.cuda.is_available()),
        "version": torch.version.cuda,
    }


def print_system_env_info():
    details = {
        "Packages": info_packages(),
        "CUDA": info_cuda(),
    }
    lines = nice_print(details)
    text = os.linesep.join(lines)
    return text


def nice_print(details: Dict, level: int = 0) -> List:
    lines = []
    LEVEL_OFFSET = "\t"
    KEY_PADDING = 20
    for k in sorted(details):
        key = f"* {k}:" if level == 0 else f"- {k}:"
        if isinstance(details[k], dict):
            lines += [level * LEVEL_OFFSET + key]
            lines += nice_print(details[k], level + 1)
        elif isinstance(details[k], (set, list, tuple)):
            lines += [level * LEVEL_OFFSET + key]
            lines += [(level + 1) * LEVEL_OFFSET + "- " + v for v in details[k]]
        else:
            template = "{:%is} {}" % KEY_PADDING
            key_val = template.format(key, details[k])
            lines += [(level * LEVEL_OFFSET) + key_val]
    return lines


def get_model_via_name(model_name):
    if model_name == "T3VIP":
        models_m = m_t3vip
    elif model_name == "SV2P":
        models_m = m_sv2p
    else:
        raise ValueError

    return models_m
