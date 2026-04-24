"""
Exponential Moving Average (EMA) of model weights.

아이디어:
    학습 중 매 iter마다 현재 모델 가중치와 "그림자 가중치(shadow)"를 지수 평균.
      shadow = decay * shadow + (1 - decay) * current
    이 shadow를 eval/predict에 사용하면 변동이 줄어들어 1~2% mIoU 향상이 흔함.

사용 예 (train):
    ema = ModelEMA(model, decay=0.999)
    for ...:
        optimizer.step()
        ema.update(model)
    # 저장
    torch.save({..., "ema_state_dict": ema.state_dict()}, ckpt_path)

사용 예 (eval/predict):
    ckpt = torch.load(path)
    if args.use_ema and "ema_state_dict" in ckpt:
        model.load_state_dict(ckpt["ema_state_dict"])
    else:
        model.load_state_dict(ckpt["model_state_dict"])

주의:
    - BN running_mean/var 등 buffer도 함께 평균 (segmentation에 중요)
    - decay는 0.999~0.9999 권장. 학습 길수록 높게
"""

from __future__ import annotations

import copy
from typing import Iterable

import torch
import torch.nn as nn


class ModelEMA:
    """
    Simple EMA wrapper.

    내부에 model의 deep copy를 유지하고, update() 호출마다 가중치/버퍼를 지수 평균.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        # deep copy해서 같은 device에 올림. requires_grad=False 로 고정 (gradient 안 흐름)
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """
        주기적으로 호출. current_model 가중치로 shadow 업데이트.

        parameters + buffers 모두 평균 (BN running stats 포함).
        dtype이 float가 아닌 buffer(long 등)는 그대로 복사.
        """
        src = model
        d = self.decay

        # 파라미터 평균
        for p_ema, p in zip(self.ema_model.parameters(), src.parameters()):
            p_ema.data.mul_(d).add_(p.data, alpha=1 - d)

        # 버퍼 평균 (BN running_mean, running_var 등)
        for b_ema, b in zip(self.ema_model.buffers(), src.buffers()):
            if b.dtype.is_floating_point:
                b_ema.data.mul_(d).add_(b.data, alpha=1 - d)
            else:
                # num_batches_tracked 같은 int buffer는 그냥 복사
                b_ema.data.copy_(b.data)

    def state_dict(self) -> dict:
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self.ema_model.load_state_dict(state_dict)

    def to(self, device: torch.device) -> "ModelEMA":
        self.ema_model.to(device)
        return self
