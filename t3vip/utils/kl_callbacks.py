from pytorch_lightning import Callback, LightningModule, Trainer
import torch
from typing import Any


def sigmoid(scale: float, shift: float, x: int) -> float:
    return torch.sigmoid(torch.Tensor([(x - shift) / (scale / 12)])).item()


class KLSchedule(Callback):
    """
    Base class for KL Annealing
    """

    def __init__(self, start_iteration: int, end_iteration: int, max_kl_beta: float):
        self.start_iteration = start_iteration
        self.end_iteration = end_iteration
        self.max_kl_beta = max_kl_beta

    def on_train_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int, unused: int = 0
    ) -> None:
        iteration = pl_module.global_step
        kl_beta = self._anneal_fn(iteration)
        pl_module.set_kl_beta(kl_beta)  # type: ignore

    def _anneal_fn(self, iteration):
        raise NotImplementedError


class KLConstantSchedule(KLSchedule):
    def __init__(self):
        pass

    def on_train_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int, unused: int = 0
    ) -> None:
        pass

    def _anneal_fn(self, iteration: int) -> None:
        pass


class KLSigmoidSchedule(KLSchedule):
    def _anneal_fn(self, iteration: int) -> float:
        if iteration < self.start_iteration:
            kl_beta = 0.0
        elif iteration > self.end_iteration:
            kl_beta = self.max_kl_beta
        else:
            scale = self.end_iteration - self.start_iteration
            shift = (self.end_iteration + self.start_iteration) / 2
            kl_beta = sigmoid(scale=scale, shift=shift, x=iteration) * self.max_kl_beta
        return kl_beta


class KLLinearSchedule(KLSchedule):
    def _anneal_fn(self, iteration: int) -> float:
        if iteration < self.start_iteration:
            kl_beta = 0.0
        elif iteration > self.end_iteration:
            kl_beta = self.max_kl_beta
        else:
            kl_beta = (
                self.max_kl_beta * (iteration - self.start_iteration) / (self.end_iteration - self.start_iteration)
            )
        return kl_beta


if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("TkAgg")
    import numpy as np

    kl = KLLinearSchedule(100000, 120000, 1)
    x = np.arange(200000)
    y = [kl._anneal_fn(i) for i in x]
    plt.plot(x, y)

    kl2 = KLSigmoidSchedule(100000, 120000, 1)
    x = np.arange(200000)
    y = [kl2._anneal_fn(i) for i in x]
    plt.plot(x, y)

    plt.show()
