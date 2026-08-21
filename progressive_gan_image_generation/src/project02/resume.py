"""Progressive stage configuration + checkpoint resume/init_from helpers.

Carved out of the training entrypoint so the training loop only
deals with step orchestration. All helpers here are pure (no I/O beyond
`torch.load`) and have unchanged signatures relative to the trainer-local
versions they replaced.

Covers:

- `_stage_config` / `progressive_state_from_config`: derive the resolution,
  stage_mode, alpha spec, and stage_id from the config.
- `_validate_resume_state` / `read_progressive_state`: enforce/inspect the
  `extra_state.progressive` block of a saved checkpoint.
- `_normalized_state_dict`: strip a `torch.compile` `_orig_mod.` prefix.
- `_generator_init_prefixes` / `_discriminator_init_prefixes`: pick which
  parameter prefixes are eligible for cross-stage init_from.
- `_load_filtered_state_dict` / `load_previous_stage_init`: load the matching
  subset of a previous-stage checkpoint into the current generator /
  discriminator (and EMA).
- `_init_from_config`: parse the `training.init_from` config knob.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from project02.models.generator import SUPPORTED_INTERNAL_RESOLUTIONS, STAGE_RESOLUTIONS


def _previous_resolution(resolution: int) -> int | None:
    index = STAGE_RESOLUTIONS.index(int(resolution))
    if index == 0:
        return None
    return STAGE_RESOLUTIONS[index - 1]


def _stage_config(cfg: dict) -> dict:
    training_cfg = cfg.get("training", {})
    progressive_cfg = dict(cfg.get("progressive", {}))
    progressive_cfg.update(training_cfg.get("progressive", {}))
    resolution = int(progressive_cfg.get("resolution", cfg.get("model", {}).get("image_size", 64)))
    if resolution not in STAGE_RESOLUTIONS:
        raise ValueError(f"progressive resolution must be one of {STAGE_RESOLUTIONS}, got {resolution}.")
    stage_mode = str(progressive_cfg.get("stage_mode", "stabilize")).lower()
    if stage_mode == "base":
        stage_mode = "stabilize"
    if stage_mode not in {"transition", "stabilize"}:
        raise ValueError(f"progressive stage_mode must be transition or stabilize, got {stage_mode!r}.")
    if resolution == STAGE_RESOLUTIONS[0] and stage_mode == "transition":
        raise ValueError("64 stage cannot run in transition mode.")
    prev_resolution = _previous_resolution(resolution)
    return {
        "resolution": resolution,
        "prev_resolution": prev_resolution,
        "stage_mode": stage_mode,
        "alpha_spec": progressive_cfg.get("alpha", 1.0),
        "stage_id": str(progressive_cfg.get("stage_id", f"{resolution}_{stage_mode}")),
    }


def progressive_state_from_config(cfg: dict) -> dict:
    stage = _stage_config(cfg)
    return {
        "resolution": stage["resolution"],
        "prev_resolution": stage["prev_resolution"],
        "stage_mode": stage["stage_mode"],
        "stage_id": stage["stage_id"],
    }


def _validate_resume_state(extra_state: dict, expected_state: dict) -> None:
    progressive = extra_state.get("progressive") if isinstance(extra_state, dict) else None
    if not isinstance(progressive, dict):
        raise ValueError("Resume checkpoint does not contain extra_state.progressive.")
    for key in ("resolution", "prev_resolution", "stage_mode", "stage_id"):
        if progressive.get(key) != expected_state.get(key):
            raise ValueError(
                "Progressive checkpoint state does not match config: "
                f"{key} checkpoint={progressive.get(key)!r}, config={expected_state.get(key)!r}"
            )


def read_progressive_state(checkpoint_path: str | Path) -> dict:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    progressive = ckpt.get("extra_state", {}).get("progressive") if isinstance(ckpt, dict) else None
    if not isinstance(progressive, dict):
        raise ValueError("Checkpoint does not contain extra_state.progressive.")
    return progressive


def _normalized_state_dict(state: dict[str, Tensor]) -> dict[str, Tensor]:
    prefix = "_orig_mod."
    if not any(key.startswith(prefix) for key in state):
        return state
    return {
        key[len(prefix):] if key.startswith(prefix) else key: value
        for key, value in state.items()
    }


def _generator_init_prefixes(prev_resolution: int) -> list[str]:
    prefixes: list[str] = []
    for resolution in SUPPORTED_INTERNAL_RESOLUTIONS:
        if resolution <= int(prev_resolution):
            prefixes.append(f"stages.{resolution}.")
    prefixes.append(f"to_rgb.{int(prev_resolution)}.")
    return prefixes


def _discriminator_init_prefixes(prev_resolution: int) -> list[str]:
    prev_resolution = int(prev_resolution)
    prefixes = [f"from_rgb.{prev_resolution}.", "tail."]
    for resolution in SUPPORTED_INTERNAL_RESOLUTIONS[1:]:
        if resolution <= prev_resolution:
            prefixes.append(f"down_stages.{resolution}.")
    return prefixes


def _load_filtered_state_dict(
    module: torch.nn.Module,
    source_state: dict[str, Tensor],
    allowed_prefixes: list[str],
) -> dict:
    """Load only allowed-prefix, shape-compatible parameters into a module.

    This powers progressive stage init_from, where a previous-stage checkpoint
    contributes compatible subsets rather than a final strict resume state.
    """

    target = module.state_dict()
    source_state = _normalized_state_dict(source_state)
    loadable = {}
    skipped_keys = []
    skipped_shape_keys = []
    unexpected_keys = []
    for key, value in source_state.items():
        if not any(key.startswith(prefix) for prefix in allowed_prefixes):
            skipped_keys.append(key)
            continue
        if key not in target:
            unexpected_keys.append(key)
            continue
        if tuple(target[key].shape) != tuple(value.shape):
            skipped_shape_keys.append(key)
            continue
        loadable[key] = value

    updated = dict(target)
    updated.update(loadable)
    module.load_state_dict(updated, strict=True)
    return {
        "loaded": len(loadable),
        "skipped": len(skipped_keys),
        "skipped_shape": len(skipped_shape_keys),
        "unexpected": len(unexpected_keys),
        "loaded_keys": sorted(loadable),
        "skipped_keys": sorted(skipped_keys),
        "skipped_shape_keys": sorted(skipped_shape_keys),
        "unexpected_keys": sorted(unexpected_keys),
    }


def load_previous_stage_init(
    checkpoint_path: str | Path,
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    generator_ema: torch.nn.Module | None,
    load_resolution: int,
    map_location: str | torch.device,
) -> dict:
    """Initialize a new progressive stage from a previous-resolution checkpoint.

    Only shape-compatible parameters under the generator/discriminator allowed
    prefixes are loaded, e.g. for 512->1024 stage growth. This path is separate
    from final strict resume.
    """

    ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    generator_prefixes = _generator_init_prefixes(load_resolution)
    discriminator_prefixes = _discriminator_init_prefixes(load_resolution)
    report = {
        "path": str(checkpoint_path),
        "load_resolution": int(load_resolution),
        "generator": _load_filtered_state_dict(generator, ckpt["generator"], generator_prefixes),
        "discriminator": _load_filtered_state_dict(discriminator, ckpt["discriminator"], discriminator_prefixes),
    }
    if generator_ema is not None:
        ema_source = ckpt.get("generator_ema", ckpt["generator"])
        report["generator_ema"] = _load_filtered_state_dict(generator_ema, ema_source, generator_prefixes)
    return report


def _init_from_config(training_cfg: dict) -> dict:
    init_from_cfg = training_cfg.get("init_from")
    if init_from_cfg is None or init_from_cfg is False:
        return {"enabled": False, "path": None}
    if isinstance(init_from_cfg, str):
        return {"enabled": True, "path": init_from_cfg}
    if not isinstance(init_from_cfg, dict):
        raise TypeError("training.init_from must be null, false, a checkpoint path string, or a dict.")
    enabled = bool(init_from_cfg.get("enabled", False))
    path = init_from_cfg.get("path") or init_from_cfg.get("checkpoint") or init_from_cfg.get("checkpoint_path")
    if enabled and not path:
        raise ValueError("training.init_from.enabled=true requires a checkpoint path.")
    return {"enabled": enabled, "path": path}
