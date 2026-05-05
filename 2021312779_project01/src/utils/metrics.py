from __future__ import annotations

import torch
from torch import Tensor


IGNORE_INDEX = 255


def accuracy(logits: Tensor, targets: Tensor) -> float:
    """Return pixel accuracy as a Python percentage."""
    with torch.no_grad():
        pred = logits.argmax(dim=1)
        valid = targets != IGNORE_INDEX
        if valid.sum() == 0:
            return 0.0
        correct = pred[valid] == targets[valid]
        return correct.float().mean().item() * 100.0


@torch.no_grad()
def accuracy_counts(logits: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
    """Return GPU-side correct/total counts for low-overhead training logs."""
    pred = logits.argmax(dim=1)
    valid = targets != IGNORE_INDEX
    correct = (pred[valid] == targets[valid]).sum()
    total = valid.sum()
    return correct, total


@torch.no_grad()
def mIoU(logits: Tensor, targets: Tensor, num_class: int) -> float:
    """Compute mean IoU while ignoring VOC label 255."""
    pred = logits.argmax(dim=1)
    valid = targets != IGNORE_INDEX
    if valid.sum() == 0:
        return 0.0

    index = targets[valid] * num_class + pred[valid]
    conf = torch.bincount(index, minlength=num_class * num_class).reshape(num_class, num_class)

    true_positive = conf.diag()
    false_negative = conf.sum(dim=1) - true_positive
    false_positive = conf.sum(dim=0) - true_positive
    denom = true_positive + false_positive + false_negative
    valid_class = denom > 0
    iou = true_positive[valid_class] / denom[valid_class]
    return iou.mean().item()


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum: float = 0.0
        self.count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        self.sum += val * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count > 0 else 0.0
