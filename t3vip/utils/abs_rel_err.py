from typing import Tuple

import torch
from torch import Tensor

from torchmetrics.utilities.checks import _check_same_shape


def _mean_absolute_relative_error_update(preds: Tensor, target: Tensor) -> Tuple[Tensor, int]:
    """Updates and returns variables required to compute Mean Absolute Relative Error.

    Checks for same shape of input tensors.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
    """

    _check_same_shape(preds, target)
    sum_abs_rel_error = torch.sum(torch.abs(preds - target) / torch.abs(target))
    n_obs = target.numel()
    return sum_abs_rel_error, n_obs


def _mean_absolute_relative_error_compute(sum_abs_rel_error: Tensor, n_obs: int) -> Tensor:
    """Computes Mean Absolute Relative Error.

    Args:
        sum_abs_rel_error: Sum of absolute relative value of errors over all observations
        n_obs: Number of predictions or observations

    Example:
        >>> preds = torch.tensor([0., 1, 2, 3])
        >>> target = torch.tensor([0., 1, 2, 2])
        >>> sum_abs_error, n_obs = _mean_absolute_relative_error_update(preds, target)
        >>> _mean_absolute_relative_error_compute(sum_abs_error, n_obs)
        tensor(0.2500)
    """

    return sum_abs_rel_error / n_obs


def mean_absolute_relative_error(preds: Tensor, target: Tensor) -> Tensor:
    """Computes mean absolute relative error.

    Args:
        preds: estimated labels
        target: ground truth labels

    Return:
        Tensor with MARE

    Example:
        >>> from torchmetrics.functional import mean_absolute_relative_error
        >>> x = torch.tensor([0., 1, 2, 3])
        >>> y = torch.tensor([0., 1, 2, 2])
        >>> mean_absolute_relative_error(x, y)
        tensor(0.2500)
    """
    sum_abs_rel_error, n_obs = _mean_absolute_relative_error_update(preds, target)
    return _mean_absolute_relative_error_compute(sum_abs_rel_error, n_obs)
