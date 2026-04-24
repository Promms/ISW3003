"""
모델 팩토리 — train.py / eval.py / predict.py가 공용으로 사용.

yaml의 model.backbone 필드로 백본 선택.
새 백본 추가 시 여기 한 곳만 수정.
"""

from __future__ import annotations

import torch.nn as nn


def build_model(backbone: str, num_classes: int) -> nn.Module:
    """
    백본 이름으로 DeepLabV3+ 모델 생성.

    Args:
        backbone: "mobilenet" | "efficientnet"
        num_classes: 출력 클래스 수 (VOC=21)

    Returns:
        nn.Module — 학습/추론에 바로 사용 가능
    """
    if backbone == "mobilenet":
        from models.deeplabv3plus_mobilenet import deeplab_v3
        return deeplab_v3(num_classes=num_classes)
    elif backbone == "efficientnet":
        from models.deeplabv3plus_efficientnet import deeplab_v3_efficientnet
        return deeplab_v3_efficientnet(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown backbone: {backbone} (expected 'mobilenet' | 'efficientnet')")
