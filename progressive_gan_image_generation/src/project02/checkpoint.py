from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer


def _unwrap_compiled_module(module: nn.Module) -> nn.Module:
    original = getattr(module, "_orig_mod", None)
    return original if isinstance(original, nn.Module) else module


def _normalized_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefix = "_orig_mod."
    if not any(key.startswith(prefix) for key in state):
        return state
    return {
        key[len(prefix) :] if key.startswith(prefix) else key: value
        for key, value in state.items()
    }


def _load_module_strict_compatible(module: nn.Module, source_state: dict[str, torch.Tensor]) -> None:
    normalized = _normalized_state_dict(source_state)
    result = module.load_state_dict(normalized, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            "Checkpoint state_dict keys do not match target module: "
            f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )


def save_checkpoint(
    path: str | Path,
    step: int,
    generator: nn.Module,
    discriminator: nn.Module,
    g_optimizer: Optimizer,
    d_optimizer: Optimizer,
    config: dict[str, Any],
    wandb_run_id: str | None = None,
    generator_ema: nn.Module | None = None,
    extra_state: dict[str, Any] | None = None,
) -> None:
    """Save training state for strict resume and submission export.

    The checkpoint stores generator/discriminator states, optimizer states,
    config, optional generator_ema, optional wandb_run_id, and optional
    extra_state such as progressive stage metadata.
    """

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generator_to_save = _unwrap_compiled_module(generator)
    discriminator_to_save = _unwrap_compiled_module(discriminator)
    generator_ema_to_save = _unwrap_compiled_module(generator_ema) if generator_ema is not None else None
    state = {
        "step": step,
        "generator": generator_to_save.state_dict(),
        "discriminator": discriminator_to_save.state_dict(),
        "g_optimizer": g_optimizer.state_dict(),
        "d_optimizer": d_optimizer.state_dict(),
        "config": config,
    }
    if wandb_run_id is not None:
        state["wandb_run_id"] = wandb_run_id
    if generator_ema_to_save is not None:
        state["generator_ema"] = generator_ema_to_save.state_dict()
    if extra_state is not None:
        state["extra_state"] = extra_state
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(state, tmp_path)
    tmp_path.replace(output_path)


def load_checkpoint(
    path: str | Path,
    generator: nn.Module,
    discriminator: nn.Module | None = None,
    g_optimizer: Optimizer | None = None,
    d_optimizer: Optimizer | None = None,
    generator_ema: nn.Module | None = None,
    prefer_ema: bool = False,
    map_location: str | torch.device = "cpu",
    return_extra_state: bool = False,
) -> int | tuple[int, dict[str, Any]]:
    """Restore checkpoint state into the provided modules and optimizers.

    Loads generator or generator_ema weights, optional discriminator and
    optimizer states, and can return extra_state containing progressive stage
    metadata used by strict resume validation.
    """

    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    generator_key = "generator_ema" if prefer_ema and "generator_ema" in ckpt else "generator"
    _load_module_strict_compatible(generator, ckpt[generator_key])
    if generator_ema is not None:
        _load_module_strict_compatible(generator_ema, ckpt.get("generator_ema", ckpt["generator"]))
    if discriminator is not None and "discriminator" in ckpt:
        _load_module_strict_compatible(discriminator, ckpt["discriminator"])
    if g_optimizer is not None and "g_optimizer" in ckpt:
        g_optimizer.load_state_dict(ckpt["g_optimizer"])
    if d_optimizer is not None and "d_optimizer" in ckpt:
        d_optimizer.load_state_dict(ckpt["d_optimizer"])
    step = int(ckpt.get("step", 0))
    if return_extra_state:
        return step, ckpt.get("extra_state", {})
    return step


def config_from_checkpoint(ckpt_path: str | Path) -> dict | None:
    ckpt = torch.load(Path(ckpt_path), map_location="cpu", weights_only=False)
    return ckpt.get("config") if isinstance(ckpt, dict) else None


def find_wandb_run_id(path: str | Path) -> str | None:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        return None
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"[wandb] Could not read run id from {checkpoint_path}: {exc}")
        return None
    if not isinstance(ckpt, dict):
        return None
    run_id = ckpt.get("wandb_run_id")
    return str(run_id) if run_id else None
