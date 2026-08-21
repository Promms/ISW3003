from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from project02.models.layers import (
    ConvLayer,
    FixedFilterUpsample,
    GenerativeBasis,
    R3ResidualBlock,
)


SUPPORTED_STAGE_RESOLUTIONS = (64, 128, 256, 512, 1024)
SUPPORTED_INTERNAL_RESOLUTIONS = (4, 8, 16, 32, 64, 128, 256, 512, 1024)
STAGE_RESOLUTIONS = SUPPORTED_STAGE_RESOLUTIONS
_INTERNAL_RESOLUTIONS = SUPPORTED_INTERNAL_RESOLUTIONS[:8]


def _check_resolution(resolution: int, supported_resolutions: tuple[int, ...] = STAGE_RESOLUTIONS) -> int:
    resolution = int(resolution)
    if resolution not in supported_resolutions:
        raise ValueError(f"resolution must be one of {supported_resolutions}, got {resolution}.")
    return resolution


def _previous_resolution(resolution: int, supported_resolutions: tuple[int, ...] = STAGE_RESOLUTIONS) -> int:
    index = supported_resolutions.index(_check_resolution(resolution, supported_resolutions))
    if index == 0:
        raise ValueError("64 stage has no previous output resolution.")
    return supported_resolutions[index - 1]


def _normalize_stage_mode(stage_mode: str) -> str:
    stage_mode = str(stage_mode).lower()
    if stage_mode in {"base", "stabilize"}:
        return "stabilize"
    if stage_mode == "transition":
        return stage_mode
    raise ValueError(f"stage_mode must be 'stabilize' or 'transition', got {stage_mode!r}.")


def _as_int_list(value, length: int, name: str) -> list[int]:
    if isinstance(value, int):
        return [int(value)] * length
    if value is None:
        raise ValueError(f"{name} is required.")
    result = [int(item) for item in value]
    if len(result) != length:
        raise ValueError(f"{name} must have {length} values, got {len(result)}.")
    return result


def _default_widths() -> list[int]:
    width_by_resolution = {
        4: 768,
        8: 768,
        16: 768,
        32: 768,
        64: 384,
        128: 192,
        256: 96,
        512: 64,
    }
    return [width_by_resolution[resolution] for resolution in _INTERNAL_RESOLUTIONS]


def _active_internal_resolutions(widths: list[int]) -> tuple[int, ...]:
    if len(widths) not in {8, 9}:
        raise ValueError("widths must have 8 values for <=512 or 9 values for <=1024.")
    return SUPPORTED_INTERNAL_RESOLUTIONS[:len(widths)]


class _GeneratorStage(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        blocks: int,
        expansion: int,
        cardinality: int,
        total_blocks: int,
        is_first: bool,
        z_dim: int,
    ) -> None:
        super().__init__()
        transition: nn.Module
        if is_first:
            transition = GenerativeBasis(z_dim, out_ch)
        else:
            transition = nn.Sequential(
                ConvLayer(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity(),
                FixedFilterUpsample(),
            )
        layers: list[nn.Module] = [transition]
        layers.extend(R3ResidualBlock(out_ch, expansion, cardinality, total_blocks) for _ in range(blocks))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class ProgressiveResidualGenerator(nn.Module):
    """Progressive residual R3GAN generator with stage-aware forward paths.

    The active internal resolutions come from the width list length. The
    forward call uses resolution, alpha, and stage_mode; transition blends the
    previous upsampled RGB path with the current path by alpha.
    """

    def __init__(
        self,
        z_dim: int = 512,
        widths: list[int] | None = None,
        blocks_per_stage: int | list[int] = 2,
        expansion: int = 2,
        cardinality: int | list[int] = 16,
    ) -> None:
        super().__init__()
        self.z_dim = int(z_dim)
        widths = _default_widths() if widths is None else [int(width) for width in widths]
        self.internal_resolutions = _active_internal_resolutions(widths)
        self.supported_resolutions = tuple(
            resolution for resolution in self.internal_resolutions if resolution >= 64
        )
        blocks = _as_int_list(blocks_per_stage, len(self.internal_resolutions), "blocks_per_stage")
        cardinalities = _as_int_list(cardinality, len(self.internal_resolutions), "cardinality")

        total_blocks = sum(blocks)
        stages = {}
        in_ch = self.z_dim
        for index, resolution in enumerate(self.internal_resolutions):
            out_ch = widths[index]
            stages[str(resolution)] = _GeneratorStage(
                in_ch=in_ch,
                out_ch=out_ch,
                blocks=blocks[index],
                expansion=expansion,
                cardinality=cardinalities[index],
                total_blocks=total_blocks,
                is_first=index == 0,
                z_dim=self.z_dim,
            )
            in_ch = out_ch
        self.stages = nn.ModuleDict(stages)
        self.to_rgb = nn.ModuleDict(
            {str(resolution): ConvLayer(widths[self.internal_resolutions.index(resolution)], 3, kernel_size=1)
             for resolution in self.supported_resolutions}
        )
        self.upsample_rgb = FixedFilterUpsample()

    def features_to_resolution(self, z: Tensor, resolution: int) -> Tensor:
        resolution = _check_resolution(resolution, self.supported_resolutions)
        x: Tensor = z
        for stage_resolution in self.internal_resolutions:
            x = self.stages[str(stage_resolution)](x)
            if stage_resolution == resolution:
                return x
        raise AssertionError("unreachable resolution path")

    def forward(
        self,
        z: Tensor,
        resolution: int,
        alpha: float = 1.0,
        stage_mode: str = "stabilize",
    ) -> Tensor:
        resolution = _check_resolution(resolution, self.supported_resolutions)
        stage_mode = _normalize_stage_mode(stage_mode)
        if stage_mode == "transition":
            prev_resolution = _previous_resolution(resolution, self.supported_resolutions)
            prev_features = self.features_to_resolution(z, prev_resolution)
            prev_rgb = self.to_rgb[str(prev_resolution)](prev_features)
            up_prev_rgb = self.upsample_rgb(prev_rgb)
            current_features = prev_features
            for stage_resolution in self.internal_resolutions:
                if stage_resolution <= prev_resolution:
                    continue
                current_features = self.stages[str(stage_resolution)](current_features)
                if stage_resolution == resolution:
                    break
            current_rgb = self.to_rgb[str(resolution)](current_features)
            alpha_t = torch.as_tensor(float(alpha), dtype=current_rgb.dtype, device=current_rgb.device)
            return torch.lerp(up_prev_rgb, current_rgb, alpha_t)

        features = self.features_to_resolution(z, resolution)
        return self.to_rgb[str(resolution)](features)

    def sample_z(self, batch_size: int, device: torch.device) -> Tensor:
        return torch.randn(batch_size, self.z_dim, device=device)


def build_progressive_residual_generator(
    z_dim: int = 512,
    widths: list[int] | None = None,
    blocks_per_stage: int | list[int] = 2,
    expansion: int = 2,
    cardinality: int | list[int] = 16,
) -> ProgressiveResidualGenerator:
    return ProgressiveResidualGenerator(
        z_dim=z_dim,
        widths=widths,
        blocks_per_stage=blocks_per_stage,
        expansion=expansion,
        cardinality=cardinality,
    )


def build_generator_from_config(model_cfg: dict) -> ProgressiveResidualGenerator:
    g_cfg = dict(model_cfg)
    nested = model_cfg.get("generator")
    if isinstance(nested, dict):
        g_cfg.update(nested)
    return build_progressive_residual_generator(
        z_dim=int(model_cfg.get("z_dim", 512)),
        widths=g_cfg.get("widths"),
        blocks_per_stage=g_cfg.get("blocks_per_stage", model_cfg.get("blocks_per_stage", 2)),
        expansion=int(model_cfg.get("expansion", 2)),
        cardinality=g_cfg.get("cardinality", model_cfg.get("cardinality", 16)),
    )
