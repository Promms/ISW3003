"""MonoDepth2-like U-Net skeleton for practice."""

import torch
import torch.nn as nn
from torch import Tensor


def conv3x3(in_ch: int, out_ch: int) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)


def conv1x1(in_ch: int, out_ch: int) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)


class MonoDepthUNet(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 64, out_channels: int = 1) -> None:
        super().__init__()
        # TODO: implement encoder-decoder with skip connections
        self.stub = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, 3, H, W)
        # TODO: implement forward
        raise NotImplementedError("MonoDepthUNet forward is not implemented.")


def monodepth_unet() -> MonoDepthUNet:
    return MonoDepthUNet()
