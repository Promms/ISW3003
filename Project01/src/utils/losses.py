from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, num_classes: int, ignore_index: int = 255, smooth: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        valid = targets != self.ignore_index

        safe_targets = targets.clone()
        safe_targets[~valid] = 0
        target_onehot = F.one_hot(safe_targets, self.num_classes).permute(0, 3, 1, 2)
        target_onehot = target_onehot.float()

        valid = valid.unsqueeze(1).float()
        probs = probs * valid
        target_onehot = target_onehot * valid

        dims = (0, 2, 3)
        intersection = (probs * target_onehot).sum(dims)
        cardinality = probs.sum(dims) + target_onehot.sum(dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    positives = gt_sorted.sum()
    intersection = positives - gt_sorted.float().cumsum(0)
    union = positives + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union.clamp_min(1)
    if gt_sorted.numel() > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def _lovasz_softmax_flat(probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if probs.numel() == 0:
        return probs.sum() * 0.0

    losses = []
    num_classes = probs.size(1)
    for cls in range(num_classes):
        fg = (labels == cls).float()
        if fg.sum() == 0:
            continue
        errors = (fg - probs[:, cls]).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, _lovasz_grad(fg_sorted)))

    if not losses:
        return probs.sum() * 0.0
    return torch.stack(losses).mean()


def lovasz_softmax_loss(
    probs: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = 255,
) -> torch.Tensor:
    valid = targets != ignore_index
    if valid.sum() == 0:
        return probs.sum() * 0.0
    probs_flat = probs.permute(0, 2, 3, 1)[valid]
    labels_flat = targets[valid]
    return _lovasz_softmax_flat(probs_flat, labels_flat)


class CEDiceLovaszLoss(nn.Module):
    """CE + Dice + Lovasz loss for VOC mIoU optimization."""

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
        ce = self.ce(logits, targets)
        dice = self.dice(logits, targets)
        probs = F.softmax(logits.float(), dim=1)
        lovasz = lovasz_softmax_loss(probs, targets, self.ignore_index)
        return ce + self.dice_weight * dice + self.lovasz_weight * lovasz
