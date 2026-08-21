import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.parametrizations import spectral_norm


def sn_conv(in_ch: int, out_ch: int, kernel_size: int, padding: int = 0) -> nn.Module:
    conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding)
    nn.init.xavier_uniform_(conv.weight)
    nn.init.zeros_(conv.bias)
    return spectral_norm(conv)


def sn_linear(in_features: int, out_features: int) -> nn.Module:
    lin = nn.Linear(in_features, out_features)
    nn.init.xavier_uniform_(lin.weight)
    nn.init.zeros_(lin.bias)
    return spectral_norm(lin)


class DResBlock(nn.Module):
    """
    Residual block for the discriminator.

    Main path: LeakyReLU -> Conv 3x3 -> LeakyReLU -> Conv 3x3 -> (AvgPool)
    Skip path: (AvgPool) -> Conv 1x1 (only if shape changes)

    All convolutions are spectrally normalized; D has no activation
    normalization — Lipschitz constraint comes from SN on weights.

    `first=True` skips the input activation (raw image has no upstream act).
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        downsample: bool = True,
        first: bool = False,
        leaky_slope: float = 0.2,
    ) -> None:
        super().__init__()
        self.downsample = downsample
        self.first = first
        self.leaky_slope = leaky_slope

        self.conv1 = sn_conv(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = sn_conv(out_ch, out_ch, kernel_size=3, padding=1)

        self.learnable_sc = (in_ch != out_ch) or downsample
        if self.learnable_sc:
            self.conv_sc = sn_conv(in_ch, out_ch, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        # WHY: x is reused by the skip path, so the input activation must NOT be in-place.
        h = x if self.first else F.leaky_relu(x, self.leaky_slope)
        h = self.conv1(h)
        h = F.leaky_relu(h, self.leaky_slope, inplace=True)
        h = self.conv2(h)
        if self.downsample:
            h = F.avg_pool2d(h, 2)

        sc = x
        if self.downsample:
            sc = F.avg_pool2d(sc, 2)
        if self.learnable_sc:
            sc = self.conv_sc(sc)
        return h + sc


class Discriminator(nn.Module):
    """
    ResNet-style discriminator with spectral normalization on all weights.
    32x32 RGB image -> scalar logit.

    Stages (with c = base_channels):
      [B, 3, 32, 32] -> DResBlock down (first) -> [B,  c, 16, 16]
                     -> DResBlock down          -> [B,  c,  8,  8]
                     -> DResBlock down          -> [B, 2c,  4,  4]
                     -> DResBlock no-down       -> [B, 2c,  4,  4]
                     -> LeakyReLU -> sum-pool over spatial -> Linear -> [B, 1]
    """

    def __init__(self, base_channels: int = 128, leaky_slope: float = 0.2) -> None:
        super().__init__()
        c = base_channels
        self.leaky_slope = leaky_slope

        self.block1 = DResBlock(3, c, downsample=True, first=True)
        self.block2 = DResBlock(c, c, downsample=True)
        self.block3 = DResBlock(c, 2 * c, downsample=True)
        self.block4 = DResBlock(2 * c, 2 * c, downsample=False)
        self.head = sn_linear(2 * c, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = F.leaky_relu(x, self.leaky_slope, inplace=True)
        x = x.sum(dim=[2, 3])           # global sum pool: [B, 2c]
        return self.head(x).squeeze(1)  # [B] logits


def build_discriminator(base_channels: int = 128) -> Discriminator:
    return Discriminator(base_channels=base_channels)
