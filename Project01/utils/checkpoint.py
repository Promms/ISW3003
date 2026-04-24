"""
체크포인트 저장/복원 유틸.

torch.compile 래핑 자동 해제, AMP/EMA state 선택적 포함.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn


def _unwrap(model: nn.Module) -> nn.Module:
    """torch.compile된 모델은 _orig_mod로 언랩. 그 외엔 그대로."""
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def save_checkpoint(
    path: str,
    iter_count: int,
    model: nn.Module,
    optimizer,
    scaler=None,
    ema=None,
    best_val_top1: float = 0.0,
    cfg: Optional[dict] = None,
) -> None:
    """
    best ckpt 저장.

    compile 모델의 경우 원본 state_dict ("_orig_mod." 접두사 없음)로 저장해야
    eval.py / predict.py에서 깔끔하게 로드됨.
    """
    save_model = _unwrap(model)
    state = {
        "iter": iter_count,
        "model_state_dict": save_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_top1": best_val_top1,
    }
    if scaler is not None:
        state["scaler_state_dict"] = scaler.state_dict()  # AMP resume용
    if ema is not None:
        state["ema_state_dict"] = ema.state_dict()        # EMA resume / --use_ema용
    if cfg is not None:
        state["config"] = cfg

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
    """
    체크포인트 로드 후 메타 정보 반환.

    반환: {"iter": ..., "best_val_top1": ...}
    """
    ckpt = torch.load(path, map_location=device)
    load_target = _unwrap(model)
    load_target.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scaler is not None and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    if ema is not None and "ema_state_dict" in ckpt:
        ema.load_state_dict(ckpt["ema_state_dict"])

    return {
        "iter": ckpt.get("iter", 0),
        "best_val_top1": ckpt.get("best_val_top1", 0.0),
        "has_ema": "ema_state_dict" in ckpt,
    }
