import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GResBlock(nn.Module):
    """
    Pre-activation residual block for the generator.

    Main path : InstanceNorm -> ReLU -> (Upsample) -> Conv 3x3
              -> InstanceNorm -> ReLU -> Conv 3x3
    Skip path : (Upsample) -> Conv 1x1 (only if shape changes)
    """

    def __init__(self, in_ch: int, out_ch: int, upsample: bool = True) -> None:
        super().__init__()
        self.upsample = upsample
        self.norm1 = nn.InstanceNorm2d(in_ch, affine=True)
        self.norm2 = nn.InstanceNorm2d(out_ch, affine=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        self.learnable_sc = (in_ch != out_ch) or upsample
        if self.learnable_sc:
            self.conv_sc = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def _up(self, x: Tensor) -> Tensor:
        return F.interpolate(x, scale_factor=2, mode="nearest")

    def forward(self, x: Tensor) -> Tensor:
        h = F.relu(self.norm1(x), inplace=True)
        if self.upsample:
            h = self._up(h)
        h = self.conv1(h)
        h = F.relu(self.norm2(h), inplace=True)
        h = self.conv2(h)

        sc = self._up(x) if self.upsample else x
        if self.learnable_sc:
            sc = self.conv_sc(sc)
        return h + sc


class Generator(nn.Module):
    """
    ResNet-style generator for 32x32 RGB images.

    Stages (with c = base_channels):
      z              -> Linear  -> [B, c,   4,  4]
      GResBlock up   -> [B, c,   8,  8]
      GResBlock up   -> [B, c,  16, 16]
      GResBlock up   -> [B, c/2,32, 32]
      InstanceNorm + ReLU + Conv 3x3 + Tanh -> [B, 3, 32, 32] in [-1, 1]
    """

    def __init__(self, z_dim: int = 128, base_channels: int = 256) -> None:
        super().__init__()
        self.z_dim = z_dim
        c = base_channels

        self.project = nn.Linear(z_dim, c * 4 * 4)
        self.block1 = GResBlock(c, c, upsample=True)        # 4 -> 8
        self.block2 = GResBlock(c, c, upsample=True)        # 8 -> 16
        self.block3 = GResBlock(c, c // 2, upsample=True)   # 16 -> 32
        self.out_norm = nn.InstanceNorm2d(c // 2, affine=True)
        self.to_rgb = nn.Conv2d(c // 2, 3, kernel_size=3, padding=1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: Tensor) -> Tensor:
        x = self.project(z).view(z.size(0), -1, 4, 4)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = F.relu(self.out_norm(x), inplace=True)
        return torch.tanh(self.to_rgb(x))

    def sample_z(self, batch_size: int, device: torch.device) -> Tensor:
        return torch.randn(batch_size, self.z_dim, device=device)


def build_generator(z_dim: int = 128, base_channels: int = 256) -> Generator:
    return Generator(z_dim=z_dim, base_channels=base_channels)
