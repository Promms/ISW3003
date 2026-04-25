"""
모델 팩토리 — train.py / eval.py / predict.py가 공용으로 사용.

EfficientNet 전용 빌드.
"""

from __future__ import annotations

import torch.nn as nn


def build_model(backbone: str, num_classes: int) -> nn.Module:
    """
    DeepLabV3+ EfficientNet 모델 생성.

    Args:
        backbone: "efficientnet" (호환성 유지용 인자, 다른 값은 거부)
        num_classes: 출력 클래스 수 (VOC=21)
    """
    if backbone != "efficientnet":
        raise ValueError(f"Unknown backbone: {backbone} (이 폴더는 'efficientnet' 전용)")
    from models.deeplabv3plus_efficientnet import deeplab_v3_efficientnet
    return deeplab_v3_efficientnet(num_classes=num_classes)
