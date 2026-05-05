from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.lovasz import lovasz_softmax


class DiceLoss(nn.Module):
    """Multi-class soft Dice loss with ignore-index masking."""

    def __init__(self, num_classes: int, ignore_index: int = 255, smooth: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        valid = targets != self.ignore_index

        targets_safe = targets.clone()
        targets_safe[~valid] = 0
        targets_onehot = F.one_hot(targets_safe, num_classes=self.num_classes)
        targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()

        valid = valid.unsqueeze(1).float()
        probs = probs * valid
        targets_onehot = targets_onehot * valid

        dims = (0, 2, 3)
        intersection = (probs * targets_onehot).sum(dims)
        cardinality = probs.sum(dims) + targets_onehot.sum(dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class CEDiceLoss(nn.Module):
    """Cross-entropy plus Dice loss."""

    def __init__(self, num_classes: int, ignore_index: int = 255, dice_weight: float = 1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(num_classes, ignore_index)
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce(logits, targets) + self.dice_weight * self.dice(logits, targets)


class CEDiceLovaszLoss(nn.Module):
    """Cross-entropy + Dice + Lovasz-Softmax for mIoU-oriented fine-tuning."""

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        dice_weight: float = 1.0,
        lovasz_weight: float = 0.5,
    ):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(num_classes, ignore_index)
        self.ignore_index = ignore_index
        self.dice_weight = dice_weight
        self.lovasz_weight = lovasz_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_ce = self.ce(logits, targets)
        loss_dice = self.dice(logits, targets)
        loss_lovasz = lovasz_softmax(F.softmax(logits.float(), dim=1), targets, ignore=self.ignore_index)
        return loss_ce + self.dice_weight * loss_dice + self.lovasz_weight * loss_lovasz
