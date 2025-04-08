"""
scheduler.py

The custom step level scheduler used by MAE models.
"""
import math
from torch import Tensor
from torch.optim.lr_scheduler import _LRScheduler, _enable_get_lr_call


class CosineStepDecay(_LRScheduler):
    """The custom scheduler performs a cosine decay at the step level."""

    def __init__(self,
                 optimizer,
                 last_epoch: int=-1,
                 base_lr: float=1e-5,
                 min_lr: float=0.0,
                 warmup: int=0,
                 epochs: int=0):
        """
        :param optimizer: The model optimizer
        :type optimizer: any
        :param last_epoch: The latest epoch (optional, default: -1)
        :type last_epoch: int
        """
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.warmup = warmup
        self.epochs = epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Epoch here is a float as it represents the epoch and fraction of the epoch
            as the decay happens per batch."""
        if self.last_epoch < self.warmup:
            return [self.base_lr * self.last_epoch / self.warmup]

        lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5
        lr *= 1. + math.cos(math.pi * (self.last_epoch - self.warmup) / (self.epochs - self.warmup))

        return [lr]

    def step(self, epoch: float=0):
        """Perform a step in the scheduler."""
        with _enable_get_lr_call(self):
            self.last_epoch = epoch
            lr = self.get_lr()[0]

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr * param_group.get("lr_scale", 1.0)

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]
