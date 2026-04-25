import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.lovasz_losses import lovasz_softmax


class DiceLoss(nn.Module):
    def __init__(self, num_classes: int, ignore_index: int = 255, smooth: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        # 특정 class가 하나도 없을 때 분모가 0이 되는 것을 방지
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits:  (B, C, H, W) ← 모델이 예측한 class
        # targets: (B, H, W)   ← 정답 class index

        probs = F.softmax(logits, dim=1)

        valid = (targets != self.ignore_index)
        targets_safe = targets.clone()
        targets_safe[~valid] = 0

        targets_onehot = F.one_hot(targets_safe, num_classes=self.num_classes)
        targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()

        valid = valid.unsqueeze(1).float()
        probs = probs * valid
        targets_onehot = targets_onehot * valid

        dims = (0, 2, 3)
        intersection = (probs * targets_onehot).sum(dims)
        cardinality  = probs.sum(dims) + targets_onehot.sum(dims)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class CEDiceLoss(nn.Module):
    def __init__(self, num_classes: int, ignore_index: int = 255, dice_weight: float = 1.0):
        super().__init__()
        self.ce   = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(num_classes, ignore_index)
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce(logits, targets) + self.dice_weight * self.dice(logits, targets)


class CEDiceLovaszLoss(nn.Module):
    """
    CE + Dice + Lovasz-Softmax 결합.

    - CE     : pixel-wise cross entropy. gradient 안정.
    - Dice   : 영역 겹침 직접 최적화. 작은 class에 강함.
    - Lovasz : mIoU 직접 최적화 (Berman 2018). 채점 metric과 일치.

    AMP 주의: lovasz_softmax는 sorting 기반이라 fp16에서 수치 불안정 가능 →
    내부에서 fp32로 캐스팅해서 계산.
    """

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
        loss_ce   = self.ce(logits, targets)
        loss_dice = self.dice(logits, targets)

        # Lovasz는 softmax 확률 입력. fp32 강제로 AMP 환경에서 sorting 안정성 확보.
        probs = F.softmax(logits.float(), dim=1)
        loss_lovasz = lovasz_softmax(probs, targets, ignore=self.ignore_index)

        return loss_ce + self.dice_weight * loss_dice + self.lovasz_weight * loss_lovasz
