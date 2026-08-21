from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from project02.models.layers import (
    ConvLayer,
    DiscriminativeBasis,
    FixedFilterDownsample,
    R3ResidualBlock,
)
from project02.models.generator import (
    _active_internal_resolutions,
    _as_int_list,
    _check_resolution,
    _default_widths,
    _normalize_stage_mode,
    _previous_resolution,
)


class _DiscriminatorDownStage(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        blocks: int,
        expansion: int,
        cardinality: int,
        total_blocks: int,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            R3ResidualBlock(in_ch, expansion, cardinality, total_blocks) for _ in range(blocks)
        )
        self.downsample = FixedFilterDownsample()
        self.proj = ConvLayer(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        x = self.downsample(x)
        return self.proj(x)


class _DiscriminatorTail(nn.Module):
    def __init__(
        self,
        channels: int,
        blocks: int,
        expansion: int,
        cardinality: int,
        total_blocks: int,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            R3ResidualBlock(channels, expansion, cardinality, total_blocks) for _ in range(blocks)
        )
        self.score = DiscriminativeBasis(channels)

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return self.score(x)


class ProgressiveResidualDiscriminator(nn.Module):
    """Progressive residual R3GAN discriminator with stage-aware RGB inputs.

    The active internal resolutions come from the width list length. The
    forward call uses resolution, alpha, and stage_mode; transition blends the
    previous downsampled RGB path with the current path by alpha.
    """

    def __init__(
        self,
        widths: list[int] | None = None,
        blocks_per_stage: int | list[int] = 2,
        expansion: int = 2,
        cardinality: int | list[int] = 16,
    ) -> None:
        super().__init__()
        widths = _default_widths() if widths is None else [int(width) for width in widths]
        self.internal_resolutions = _active_internal_resolutions(widths)
        self.supported_resolutions = tuple(
            resolution for resolution in self.internal_resolutions if resolution >= 64
        )
        width_by_resolution = dict(zip(self.internal_resolutions, widths))
        blocks = _as_int_list(blocks_per_stage, len(self.internal_resolutions), "blocks_per_stage")
        cardinalities = _as_int_list(cardinality, len(self.internal_resolutions), "cardinality")
        block_by_resolution = dict(zip(self.internal_resolutions, blocks))
        cardinality_by_resolution = dict(zip(self.internal_resolutions, cardinalities))
        total_blocks = sum(blocks)

        self.from_rgb = nn.ModuleDict(
            {str(resolution): ConvLayer(3, width_by_resolution[resolution], kernel_size=1)
             for resolution in self.supported_resolutions}
        )
        down_stages = {}
        for resolution in reversed(self.internal_resolutions[1:]):
            next_resolution = resolution // 2
            down_stages[str(resolution)] = _DiscriminatorDownStage(
                in_ch=width_by_resolution[resolution],
                out_ch=width_by_resolution[next_resolution],
                blocks=block_by_resolution[resolution],
                expansion=expansion,
                cardinality=cardinality_by_resolution[resolution],
                total_blocks=total_blocks,
            )
        self.down_stages = nn.ModuleDict(down_stages)
        self.tail = _DiscriminatorTail(
            channels=width_by_resolution[4],
            blocks=block_by_resolution[4],
            expansion=expansion,
            cardinality=cardinality_by_resolution[4],
            total_blocks=total_blocks,
        )
        self.downsample_rgb = FixedFilterDownsample()

    def _down_one(self, x: Tensor, resolution: int) -> Tensor:
        return self.down_stages[str(resolution)](x)

    def score_from_resolution(self, features: Tensor, resolution: int) -> Tensor:
        resolution = int(resolution)
        x = features
        while resolution > 4:
            x = self._down_one(x, resolution)
            resolution //= 2
        return self.tail(x)

    def forward(
        self,
        x: Tensor,
        resolution: int,
        alpha: float = 1.0,
        stage_mode: str = "stabilize",
    ) -> Tensor:
        resolution = _check_resolution(resolution, self.supported_resolutions)
        stage_mode = _normalize_stage_mode(stage_mode)
        if stage_mode == "transition":
            prev_resolution = _previous_resolution(resolution, self.supported_resolutions)
            low_x = self.downsample_rgb(x)
            low_features = self.from_rgb[str(prev_resolution)](low_x)
            high_features = self.from_rgb[str(resolution)](x)
            high_features = self._down_one(high_features, resolution)
            alpha_t = torch.as_tensor(float(alpha), dtype=high_features.dtype, device=high_features.device)
            features = torch.lerp(low_features, high_features, alpha_t)
            return self.score_from_resolution(features, prev_resolution)

        features = self.from_rgb[str(resolution)](x)
        return self.score_from_resolution(features, resolution)


def build_progressive_residual_discriminator(
    widths: list[int] | None = None,
    blocks_per_stage: int | list[int] = 2,
    expansion: int = 2,
    cardinality: int | list[int] = 16,
) -> ProgressiveResidualDiscriminator:
    return ProgressiveResidualDiscriminator(
        widths=widths,
        blocks_per_stage=blocks_per_stage,
        expansion=expansion,
        cardinality=cardinality,
    )


def build_discriminator_from_config(model_cfg: dict) -> ProgressiveResidualDiscriminator:
    d_cfg = dict(model_cfg)
    nested = model_cfg.get("discriminator")
    if isinstance(nested, dict):
        d_cfg.update(nested)
    return build_progressive_residual_discriminator(
        widths=d_cfg.get("widths"),
        blocks_per_stage=d_cfg.get("blocks_per_stage", model_cfg.get("blocks_per_stage", 2)),
        expansion=int(model_cfg.get("expansion", 2)),
        cardinality=d_cfg.get("cardinality", model_cfg.get("cardinality", 16)),
    )
