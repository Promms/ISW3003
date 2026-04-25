import torch
import torch.nn as nn
import torch.nn.functional as F


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

        # Step 1: raw 점수 → 확률로 변환 (0~1, 21개 합=1)
        probs = F.softmax(logits, dim=1)   # (B, C, H, W)

        # Step 2: 255(ignore) 픽셀 마스크 생성
        # F.one_hot()은 0 ~ num_classes-1 범위만 받음 → 255가 있으면 오류
        valid = (targets != self.ignore_index)
        targets_safe = targets.clone()
        targets_safe[~valid] = 0                  # 255 → 0으로 임시 교체

        # Step 3: 정답을 one-hot으로 변환
        # targets_safe: (B, H, W) → one_hot: (B, H, W, C) → permute: (B, C, H, W)
        targets_onehot = F.one_hot(targets_safe, num_classes=self.num_classes)
        targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()

        # Step 4: 255 픽셀은 임시로 0으로 바꿔놨으므로, 그대로 두면 경계 픽셀이 배경인 척 학습에 끼어듦
        # → (B, C, H, W)인 probs/targets_onehot과 곱할 때 C개 채널로 broadcast됨
        # 255였던 픽셀: 예측도 0, 정답도 0 → Dice 계산에 기여 없음
        valid = valid.unsqueeze(1).float()            # (B, H, W) → (B, 1, H, W)
        probs = probs * valid                         # 255 픽셀의 예측 확률 전부 0으로
        targets_onehot = targets_onehot * valid       # 255 픽셀의 정답 전부 0으로

        # Step 5: class별 Dice 계산
        # dims=(0,2,3): B, H, W 방향으로 합산 → class C개짜리 숫자만 남음
        # 즉, 전체 배치/이미지의 모든 픽셀을 class별로 집계
        dims = (0, 2, 3)
        # intersection: 예측 확률 × 정답(0 or 1) → 정답 class에 얼마나 높은 확률을 줬는지
        intersection = (probs * targets_onehot).sum(dims)
        # cardinality: 예측 합 + 정답 합 → Dice 분모
        cardinality  = probs.sum(dims) + targets_onehot.sum(dims)

        # Step 6: Dice score → Loss
        # dice: 0~1, 1에 가까울수록 예측이 정답과 잘 겹침
        # smooth로 0/0 방지
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()   # C개 class 평균 → 하나의 loss 값 (0에 가까울수록 좋음)


class CEDiceLoss(nn.Module):
    def __init__(self, num_classes: int, ignore_index: int = 255, dice_weight: float = 1.0):
        super().__init__()
        self.ce   = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(num_classes, ignore_index)
        # dice_weight: CE Loss 대비 Dice Loss의 비중
        # 1.0으로 시작해서 결과 보고 조정 (너무 크면 CE 학습 방해, 너무 작으면 효과 없음)
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # CE:   픽셀 단위 분류 → gradient 안정적
        # Dice: 영역 겹침 직접 최적화 → 작은 class(bottle, bird 등)에 강함
        return self.ce(logits, targets) + self.dice_weight * self.dice(logits, targets)
