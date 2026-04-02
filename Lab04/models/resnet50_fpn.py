"""ResNet-50 + FPN skeleton for practice."""

from typing import List, Tuple

# import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.resnet50 import ResNet50


def conv1x1(in_ch: int, out_ch: int) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)


def conv3x3(in_ch: int, out_ch: int) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)


class ResNetBackbone(ResNet50):
    def forward_features(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # x: (N, 3, H, W)
        # TODO: return C2, C3, C4, C5 feature maps

        # Stem convolution block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        C2 = self.layer1(x)
        C3 = self.layer2(C2)
        C4 = self.layer3(C3)
        C5 = self.layer4(C4)

        return (C2, C3, C4, C5)


class FPN(nn.Module):
    def __init__(self, in_channels: List[int], out_channels: int = 256) -> None:
        super().__init__()
        # TODO: build lateral and smoothing layers
        # in_channels = [256, 512, 1024, 2048]  (C2~C5)

        self.lateral = nn.ModuleList([
            conv1x1(in_channels[0], out_channels),
            conv1x1(in_channels[1], out_channels),
            conv1x1(in_channels[2], out_channels),
            conv1x1(in_channels[3], out_channels),
        ])
        self.smooth = nn.ModuleList([
            conv3x3(out_channels, out_channels),
            conv3x3(out_channels, out_channels),
            conv3x3(out_channels, out_channels),
            conv3x3(out_channels, out_channels),
        ])

    def forward(self, feats: List[Tensor]) -> List[Tensor]:
        # feats: [c2, c3, c4, c5]
        # TODO: build top-down FPN

        l_c2 = self.lateral[0](feats[0])
        l_c3 = self.lateral[1](feats[1])
        l_c4 = self.lateral[2](feats[2])
        l_c5 = self.lateral[3](feats[3])

        # (N, 256, 14, 14) + (N, 256, 14, 14)  ← shape 맞춰서 더함
        p5 = l_c5
        p4 = l_c4 + F.interpolate(p5, size=l_c4.shape[-2:], mode='bilinear')
        p3 = l_c3 + F.interpolate(p4, size=l_c3.shape[-2:], mode='bilinear')
        p2 = l_c2 + F.interpolate(p3, size=l_c2.shape[-2:], mode='bilinear')

        s_c2 = self.smooth[0](p2)
        s_c3 = self.smooth[1](p3)
        s_c4 = self.smooth[2](p4)
        s_c5 = self.smooth[3](p5)

        return [s_c2, s_c3, s_c4, s_c5]


class ResNet50FPN(nn.Module):
    def __init__(self, fpn_channels: int = 256) -> None:
        super().__init__()
        # TODO: create backbone + FPN
        self.backbone = ResNetBackbone([3, 4, 6, 3])
        self.fpn = FPN([256, 512, 1024, 2048], out_channels=fpn_channels)

    def forward(self, x: Tensor) -> List[Tensor]:
        # x: (N, 3, H, W)
        # TODO: return [p2, p3, p4, p5]
        c2, c3, c4 ,c5 = self.backbone.forward_features(x)
        fpn_outs = self.fpn([c2, c3, c4, c5])
        return fpn_outs


def resnet50_fpn() -> ResNet50FPN:
    return ResNet50FPN()
