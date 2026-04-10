import torch
from torch import Tensor


def accuracy(logits: Tensor, targets: Tensor) -> float:
    """
    Compute top-1 accuracy for a batch.

    Args:
        logits:  [B, num_classes] raw model outputs
        targets: [B] integer class indices

    Returns:
        Accuracy in [0, 100].
    """
    with torch.no_grad():
        pred = logits.argmax(dim=1)
        return (pred == targets).float().mean().item() * 100.0


class AverageMeter:
    """Tracks a running weighted average of a scalar (loss, accuracy, etc.)."""

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
