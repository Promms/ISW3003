"""
체크포인트 저장/복원 유틸.

AMP/EMA state 선택적 포함.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn


def save_checkpoint(
    path: str,
    iter_count: int,
    model: nn.Module,
    optimizer,
    scaler=None,
    ema=None,
    cfg: Optional[dict] = None,
    wandb_run_id: Optional[str] = None,
    best_raw_top1: float = 0.0,
    best_ema_top1: float = 0.0,
) -> None:
    """ckpt 저장. wandb_run_id 포함하면 resume 시 같은 run에 이어 기록 가능."""
    state = {
        "iter": iter_count,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_raw_top1": best_raw_top1,
        "best_ema_top1": best_ema_top1,
    }
    if scaler is not None:
        state["scaler_state_dict"] = scaler.state_dict()
    if ema is not None:
        state["ema_state_dict"] = ema.state_dict()
    if cfg is not None:
        state["config"] = cfg
    if wandb_run_id is not None:
        state["wandb_run_id"] = wandb_run_id

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer=None,
    scaler=None,
    ema=None,
    device: Optional[torch.device] = None,
) -> dict:
    """체크포인트 로드 후 메타 정보 반환."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scaler is not None and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    if ema is not None and "ema_state_dict" in ckpt:
        ema.load_state_dict(ckpt["ema_state_dict"])

    return {
        "iter":          ckpt.get("iter", 0),
        "best_raw_top1": ckpt.get("best_raw_top1", 0.0),
        "best_ema_top1": ckpt.get("best_ema_top1", 0.0),
        "has_ema":       "ema_state_dict" in ckpt,
    }
