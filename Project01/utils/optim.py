"""
Optimizer / LR scheduling 유틸.

- Differential LR: backbone은 head보다 작은 lr 사용 (pretrained 가중치 보호)
- Poly LR decay: SemSeg에서 일반적으로 쓰는 (1 - t/T)^0.9 스케줄
"""

from __future__ import annotations

import torch
import torch.nn as nn


def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    """
    AdamW + Differential LR 파라미터 그룹 구성.

    yaml.training.learning_rate        → head lr
    yaml.training.backbone_lr_scale    → backbone lr = head lr × scale
    yaml.training.weight_decay         → weight decay

    param_groups 순서 (Poly decay에서도 이 순서 전제):
      [0] backbone_low
      [1] backbone_high
      [2] aspp    (head)
      [3] decoder (head)
    """
    head_lr = cfg["training"]["learning_rate"]
    backbone_lr = head_lr * cfg["training"]["backbone_lr_scale"]

    return torch.optim.AdamW(
        [
            {"params": model.backbone_low.parameters(),  "lr": backbone_lr, "initial_lr": backbone_lr},
            {"params": model.backbone_high.parameters(), "lr": backbone_lr, "initial_lr": backbone_lr},
            {"params": model.aspp.parameters(),          "lr": head_lr,     "initial_lr": head_lr},
            {"params": model.decoder.parameters(),       "lr": head_lr,     "initial_lr": head_lr},
        ],
        weight_decay=cfg["training"]["weight_decay"],
    )


def poly_lr_step(optimizer: torch.optim.Optimizer, iter_count: int, total_iters: int,
                 power: float = 0.9) -> None:
    """
    Poly LR: 각 group의 initial_lr 기준으로 (1 - t/T)^power 비율로 감소.
    backbone/head 비율은 유지됨.
    """
    decay = (1 - iter_count / total_iters) ** power
    for pg in optimizer.param_groups:
        pg["lr"] = pg["initial_lr"] * decay


def set_backbone_requires_grad(model: nn.Module, requires_grad: bool) -> None:
    """
    Backbone 파라미터의 requires_grad를 일괄 변경 (freeze/unfreeze용).

    compile된 모델에도 안전하게 동작하도록 호출 측에서 _orig_mod 언랩 후 전달 권장.
    """
    for param in model.backbone_low.parameters():
        param.requires_grad = requires_grad
    for param in model.backbone_high.parameters():
        param.requires_grad = requires_grad
