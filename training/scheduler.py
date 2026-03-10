"""Learning-rate scheduler helpers (wrappers around PyTorch schedulers).

DEAD CODE — NOT CALLED ANYWHERE.
Ultralytics manages its own cosine-decay schedule internally via the ``lrf``
training parameter (final LR = lr0 * lrf).  A custom LambdaLR cannot be
injected into the Ultralytics training loop without patching the trainer
subclass, so this module is unused.  It is kept here for reference only;
do NOT wire it into trainer.py unless you also subclass Ultralytics' own
trainer and override ``build_optimizer``.
"""

from __future__ import annotations

import math

import torch
from torch.optim.lr_scheduler import LambdaLR, LRScheduler


def cosine_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    min_lr_factor: float = 0.01,
) -> LRScheduler:
    """Cosine annealing schedule with linear warmup.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
    warmup_epochs : int
        Number of warmup epochs with linearly increasing LR.
    total_epochs : int
        Total number of training epochs.
    min_lr_factor : float
        Minimum LR as a fraction of initial LR.
    """

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return epoch / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return min_lr_factor + 0.5 * (1.0 - min_lr_factor) * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)
