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

        self.enc1 = nn.Sequential(
            conv3x3(in_channels, in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),

            conv3x3(in_channels, base_channels),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        self.enc2 = nn.Sequential(
            conv3x3(base_channels, base_channels*2),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True),

            conv3x3(base_channels*2, base_channels*2),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True),
        )

        self.enc3 = nn.Sequential(
            conv3x3(base_channels*2, base_channels*4),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(inplace=True),

            conv3x3(base_channels*4, base_channels*4),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(inplace=True),
        )

        self.enc4 = nn.Sequential(
            conv3x3(base_channels*4, base_channels*8),
            nn.BatchNorm2d(base_channels*8),
            nn.ReLU(inplace=True),

            conv3x3(base_channels*8, base_channels*8),
            nn.BatchNorm2d(base_channels*8),
            nn.ReLU(inplace=True),
        )

        self.dec4 = nn.Sequential(
            conv3x3(base_channels*12, base_channels*4),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(inplace=True),

            conv3x3(base_channels*4, base_channels*4),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(inplace=True),
        )

        self.dec3 = nn.Sequential(
            conv3x3(base_channels*6, base_channels*4),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(inplace=True),

            conv3x3(base_channels*4, base_channels*4),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(inplace=True),
        )

        self.dec2 = nn.Sequential(
            conv3x3(base_channels*5, base_channels*4),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(inplace=True),

            conv3x3(base_channels*4, base_channels*4),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(inplace=True),
        )

        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.head = conv1x1(base_channels*4, out_channels)

        self.stub = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, 3, H, W)
        # TODO: implement forward

        e1 = self.enc1(x)
        e2 = self.enc2(self.maxpool2d(e1))
        e3 = self.enc3(self.maxpool2d(e2))
        e4 = self.enc4(self.maxpool2d(e3))

        up4 = self.upsample(e4)
        d4 = self.dec4(torch.cat([up4, self.stub(e3)], dim=1))

        up3 = self.upsample(d4)
        d3 = self.dec3(torch.cat([up3, self.stub(e2)], dim=1))

        up2 = self.upsample(d3)
        d2 = self.dec2(torch.cat([up2, self.stub(e1)], dim=1))

        out = self.head(d2)
        
        return out


def monodepth_unet() -> MonoDepthUNet:
    return MonoDepthUNet()