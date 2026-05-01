"""
공용 building block 모듈.

- SEBlock: Squeeze-and-Excitation (Hu et al. 2018)
    채널 축의 전역 문맥(GAP)을 작은 MLP로 요약 → 시그모이드 게이트로 채널별 가중치 재보정.
    ASPP project 출력 또는 decoder refine 출력 뒤에 한 번만 걸어도 0.3~0.7 mIoU 개선 흔함.
    파라미터는 channels^2/reduction 수준이라 매우 가벼움 (256ch, r=16 → ~8K params).
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block.

    Args:
        channels : 입력 채널 수
        reduction: bottleneck 비율 (보통 16). 작을수록 표현력 ↑ 파라미터 ↑
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)  # 너무 작아지지 않게 최소 8
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        w = self.fc(self.avg_pool(x))   # (B, C, 1, 1)
        return x * w                     # broadcast 곱
