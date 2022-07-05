import torch
import numpy as np


class ScaleDepthTensor(object):
    def __init__(self, min_depth: float = 0.01, max_depth: float = 2.0):
        self.min_depth = min_depth
        self.max_depth = max_depth

    def __call__(self, depth: torch.Tensor) -> torch.Tensor:
        normalized_depth = (depth - self.min_depth) / (self.max_depth - self.min_depth)
        return normalized_depth.clip(0, 1)

    def __repr__(self):
        return self.__class__.__name__ + +"(min_depth={0}, max_depth={1})".format(self.min_depth, self.max_depth)


class RealDepthTensor(object):
    def __init__(self, min_depth: float = 0.01, max_depth: float = 2.0):
        self.min_depth = min_depth
        self.max_depth = max_depth

    def __call__(self, depth: torch.Tensor) -> torch.Tensor:
        real_depth = depth * (self.max_depth - self.min_depth) + self.min_depth
        return real_depth.clip(self.min_depth, self.max_depth)

    def __repr__(self):
        return self.__class__.__name__ + +"(min_depth={0}, max_depth={1})".format(self.min_depth, self.max_depth)


class ToNumpy(object):
    def __call__(self, sample):
        return np.array(sample)


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.std = torch.tensor(std)
        self.mean = torch.tensor(mean)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        assert isinstance(tensor, torch.Tensor)
        device = tensor.device
        if device != self.std.device:
            self.std = self.std.to(device)
        if device != self.mean.device:
            self.mean = self.mean.to(device)
        return tensor + torch.randn(tensor.size(), device=device) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(self.mean, self.std)


class ArrayToTensor(object):
    """Transforms np array to tensor."""

    def __call__(self, array: np.ndarray, device: torch.device = "cpu") -> torch.Tensor:
        assert isinstance(array, np.ndarray)
        return torch.from_numpy(array).to(device)
