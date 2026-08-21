from __future__ import annotations

import torch
import torch.nn as nn


def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    """AdamW with a lower learning rate for the pretrained backbone."""
    head_lr = cfg["training"]["learning_rate"]
    backbone_lr = head_lr * cfg["training"]["backbone_lr_scale"]

    return torch.optim.AdamW(
        [
            {"params": model.backbone_low.parameters(), "lr": backbone_lr, "initial_lr": backbone_lr},
            {"params": model.backbone_high.parameters(), "lr": backbone_lr, "initial_lr": backbone_lr},
            {"params": model.aspp.parameters(), "lr": head_lr, "initial_lr": head_lr},
            {"params": model.decoder.parameters(), "lr": head_lr, "initial_lr": head_lr},
        ],
        weight_decay=cfg["training"]["weight_decay"],
    )


def poly_lr_step(
    optimizer: torch.optim.Optimizer,
    iter_count: int,
    total_iters: int,
    power: float = 0.9,
    warmup_iters: int = 0,
) -> None:
    """Linear warmup followed by polynomial learning-rate decay."""
    if warmup_iters > 0 and iter_count < warmup_iters:
        scale = (iter_count + 1) / warmup_iters
    else:
        denom = max(1, total_iters - warmup_iters)
        progress = (iter_count - warmup_iters) / denom
        progress = min(max(progress, 0.0), 1.0)
        scale = (1.0 - progress) ** power

    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["initial_lr"] * scale


def set_backbone_requires_grad(model: nn.Module, requires_grad: bool) -> None:
    """Freeze or unfreeze both EfficientNet backbone stages."""
    for param in model.backbone_low.parameters():
        param.requires_grad = requires_grad
    for param in model.backbone_high.parameters():
        param.requires_grad = requires_grad
