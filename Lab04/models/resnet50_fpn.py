"""ResNet-50 + FPN skeleton for practice."""

from typing import List, Tuple

# import torch
import torch.nn as nn
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
        raise NotImplementedError("ResNetBackbone forward_features is not implemented.")


class FPN(nn.Module):
    def __init__(self, in_channels: List[int], out_channels: int = 256) -> None:
        super().__init__()
        # TODO: build lateral and smoothing layers
        self.lateral = nn.ModuleList()
        self.smooth = nn.ModuleList()

    def forward(self, feats: List[Tensor]) -> List[Tensor]:
        # feats: [c2, c3, c4, c5]
        # TODO: build top-down FPN
        raise NotImplementedError("FPN forward is not implemented.")


class ResNet50FPN(nn.Module):
    def __init__(self, fpn_channels: int = 256) -> None:
        super().__init__()
        # TODO: create backbone + FPN
        self.backbone = ResNetBackbone([3, 4, 6, 3])
        self.fpn = FPN([256, 512, 1024, 2048], out_channels=fpn_channels)

    def forward(self, x: Tensor) -> List[Tensor]:
        # x: (N, 3, H, W)
        # TODO: return [p2, p3, p4, p5]
        raise NotImplementedError("ResNet50FPN forward is not implemented.")


def resnet50_fpn() -> ResNet50FPN:
    return ResNet50FPN()
