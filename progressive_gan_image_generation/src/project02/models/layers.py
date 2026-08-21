"""Shared building blocks for the generator, discriminator, and exact GP backend.

- `ConvLayer`: bias-free Conv2d with MSR-style init.
- `BiasLeakyReLU`: per-channel bias + leaky_relu.
- `R3ResidualBlock`: ResNeXt-style 1x1 -> grouped 3x3 -> 1x1 residual block.
- `GenerativeBasis` / `DiscriminativeBasis`: 4x4 entry/exit modules.
- `FixedFilterUpsample` / `FixedFilterDownsample`: stride-2 depthwise resamplers.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _msr_init(module: nn.Module, gain: float = 1.0, zero: bool = False) -> nn.Module:
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        if zero:
            nn.init.zeros_(module.weight)
        else:
            if isinstance(module, nn.Conv2d):
                fan_in = module.in_channels * module.kernel_size[0] * module.kernel_size[1] // module.groups
            else:
                fan_in = module.in_features
            nn.init.normal_(module.weight, mean=0.0, std=gain / math.sqrt(fan_in))
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    return module


class BiasLeakyReLU(nn.Module):
    def __init__(self, channels: int, slope: float = 0.2) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channels))
        self.slope = slope

    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(dtype=x.dtype).view(1, -1, 1, 1)
        return F.leaky_relu(x + bias, negative_slope=self.slope, inplace=True)


class ConvLayer(nn.Module):
    """Bias-free convolution wrapper used by both generator and discriminator blocks."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, groups: int = 1, gain: float = 1.0, zero: bool = False) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = _msr_init(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, groups=groups, bias=False),
            gain=gain,
            zero=zero,
        )

    def forward(self, x: Tensor) -> Tensor:
        if torch.onnx.is_in_onnx_export():
            return self.conv(x)
        return F.conv2d(
            x,
            self.conv.weight.to(dtype=x.dtype),
            bias=None,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )


class R3ResidualBlock(nn.Module):
    """Residual three-convolution block with grouped middle convolution."""

    def __init__(
        self,
        channels: int,
        expansion: int,
        cardinality: int,
        total_blocks: int,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        expanded = channels * expansion
        groups = min(cardinality, expanded)
        while expanded % groups != 0:
            groups -= 1
        gain = math.sqrt(2.0) * (max(total_blocks, 1) ** (-1.0 / 4.0))
        self.conv1 = ConvLayer(channels, expanded, kernel_size=1, gain=gain)
        self.act1 = BiasLeakyReLU(expanded)
        self.conv2 = ConvLayer(expanded, expanded, kernel_size=kernel_size, groups=groups, gain=gain)
        self.act2 = BiasLeakyReLU(expanded)
        self.conv3 = ConvLayer(expanded, channels, kernel_size=1, zero=True)

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv1(x)
        h = self.conv2(self.act1(h))
        h = self.conv3(self.act2(h))
        return x + h


class GenerativeBasis(nn.Module):
    """Learned 4x4 feature seed modulated by latent z."""

    def __init__(self, z_dim: int, out_ch: int) -> None:
        super().__init__()
        self.basis = nn.Parameter(torch.randn(out_ch, 4, 4))
        self.linear = _msr_init(nn.Linear(z_dim, out_ch, bias=False))

    def forward(self, z: Tensor) -> Tensor:
        scale = F.linear(z, self.linear.weight.to(dtype=z.dtype), None).view(z.size(0), -1, 1, 1)
        return self.basis.to(dtype=z.dtype).unsqueeze(0) * scale


class DiscriminativeBasis(nn.Module):
    """Final discriminator projection from a 4x4 feature map to one logit."""

    def __init__(self, in_ch: int) -> None:
        super().__init__()
        self.depthwise = _msr_init(nn.Conv2d(in_ch, in_ch, kernel_size=4, groups=in_ch, bias=False))
        self.linear = _msr_init(nn.Linear(in_ch, 1, bias=False))

    def forward(self, x: Tensor) -> Tensor:
        x = F.conv2d(
            x,
            self.depthwise.weight.to(dtype=x.dtype),
            bias=None,
            stride=self.depthwise.stride,
            padding=self.depthwise.padding,
            dilation=self.depthwise.dilation,
            groups=self.depthwise.groups,
        ).flatten(1)
        return F.linear(x, self.linear.weight.to(dtype=x.dtype), None).squeeze(1)


def _fixed_filter_kernel(weights: tuple[int, ...], gain: float) -> Tensor:
    taps = torch.tensor(weights, dtype=torch.float32)
    taps = taps / taps.sum()
    kernel = torch.outer(taps, taps) * gain
    return kernel.view(1, 1, *kernel.shape)


class FixedFilterUpsample(nn.Module):
    """Fixed FIR-style stride-2 spatial upsampling layer."""

    def __init__(self, weights: tuple[int, ...] = (1, 2, 1)) -> None:
        super().__init__()
        self.register_buffer("kernel", _fixed_filter_kernel(weights, gain=4.0))

    def forward(self, x: Tensor) -> Tensor:
        if torch.onnx.is_in_onnx_export():
            channels = int(x.shape[1])
            kernel = self.kernel.repeat(channels, 1, 1, 1)
        else:
            kernel = self.kernel.to(dtype=x.dtype).expand(x.size(1), -1, -1, -1)
        return F.conv_transpose2d(
            x,
            kernel,
            stride=2,
            padding=1,
            output_padding=1,
            groups=x.size(1),
        )


class FixedFilterDownsample(nn.Module):
    """Fixed FIR-style stride-2 spatial downsampling layer."""

    def __init__(self, weights: tuple[int, ...] = (1, 2, 1)) -> None:
        super().__init__()
        self.register_buffer("kernel", _fixed_filter_kernel(weights, gain=1.0))

    def forward(self, x: Tensor) -> Tensor:
        kernel = self.kernel.to(dtype=x.dtype).expand(x.size(1), -1, -1, -1)
        return F.conv2d(x, kernel, stride=2, padding=1, groups=x.size(1))
