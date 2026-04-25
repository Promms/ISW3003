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
                 power: float = 0.9, warmup_iters: int = 0) -> None:
    """
    Linear warmup → Poly decay 복합 스케줄.

    - iter_count < warmup_iters : lr = initial_lr * (iter_count + 1) / warmup_iters  (선형 증가)
    - 그 이후                   : lr = initial_lr * (1 - progress)^power
         progress = (iter_count - warmup_iters) / (total_iters - warmup_iters)

    warmup_iters=0 이면 기존 동작과 동일.
    각 param group의 initial_lr 비율(backbone/head)은 항상 유지됨.
    """
    if warmup_iters > 0 and iter_count < warmup_iters:
        scale = (iter_count + 1) / warmup_iters
    else:
        denom = max(1, total_iters - warmup_iters)
        progress = (iter_count - warmup_iters) / denom
        progress = min(max(progress, 0.0), 1.0)
        scale = (1 - progress) ** power

    for pg in optimizer.param_groups:
        pg["lr"] = pg["initial_lr"] * scale


def set_backbone_requires_grad(model: nn.Module, requires_grad: bool) -> None:
    """Backbone 파라미터의 requires_grad를 일괄 변경 (freeze/unfreeze용)."""
    for param in model.backbone_low.parameters():
        param.requires_grad = requires_grad
    for param in model.backbone_high.parameters():
        param.requires_grad = requires_grad
