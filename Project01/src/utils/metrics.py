from __future__ import annotations

import torch
from torch import Tensor

IGNORE_INDEX = 255


def accuracy(logits: Tensor, targets: Tensor) -> float:
    """Return pixel accuracy as a Python float. Intended for validation."""
    with torch.no_grad():
        correct, total = accuracy_counts(logits, targets)
        if total.item() == 0:
            return 0.0
        return (correct.float() / total.float()).item() * 100.0


@torch.no_grad()
def accuracy_counts(logits: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
    pred = logits.argmax(dim=1)
    valid = targets != IGNORE_INDEX
    correct = (pred[valid] == targets[valid]).sum()
    total = valid.sum()
    return correct, total


@torch.no_grad()
def confusion_matrix(logits: Tensor, targets: Tensor, num_classes: int) -> Tensor:
    pred = logits.argmax(dim=1)
    valid = targets != IGNORE_INDEX
    pred = pred[valid]
    target = targets[valid]

    index = target * num_classes + pred
    return torch.bincount(
        index,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)


def miou_from_confusion(conf: Tensor) -> Tensor:
    conf = conf.float()
    iou = per_class_iou_from_confusion(conf)
    valid = ~torch.isnan(iou)
    if not valid.any():
        return conf.new_tensor(0.0)
    return iou[valid].mean()


def per_class_iou_from_confusion(conf: Tensor) -> Tensor:
    conf = conf.float()
    tp = conf.diag()
    fp = conf.sum(dim=0) - tp
    fn = conf.sum(dim=1) - tp
    denom = tp + fp + fn
    return torch.where(denom > 0, tp / denom, torch.full_like(tp, float("nan")))


def mIoU(logits: Tensor, targets: Tensor, num_class: int) -> float:
    """Backward-compatible batch mIoU helper."""
    with torch.no_grad():
        return miou_from_confusion(confusion_matrix(logits, targets, num_class)).item()


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
