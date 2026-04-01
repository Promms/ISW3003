"""DeepLabv3-style segmentation model skeleton for practice."""

from typing import List

import torch
import torch.nn as nn
from torch import Tensor

from models.resnet50 import ResNet50


def conv1x1(in_ch: int, out_ch: int) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)


def conv3x3(in_ch: int, out_ch: int, dilation: int) -> nn.Conv2d:
    return nn.Conv2d(
        in_ch,
        out_ch,
        kernel_size=3,
        padding=dilation,
        dilation=dilation,
        bias=False,
    )


class ResNetBackbone(ResNet50):
    def forward_features(self, x: Tensor) -> Tensor:
        # x: (N, 3, H, W)
        # TODO: return final feature map
        raise NotImplementedError("ResNetBackbone forward_features is not implemented.")


class ASPP(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, rates: List[int]) -> None:
        super().__init__()
        # TODO: build ASPP branches and projection
        self.branches = nn.ModuleList()
        self.global_pool = nn.Identity()
        self.project = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, C, H, W)
        # TODO: implement ASPP forward
        raise NotImplementedError("ASPP forward is not implemented.")


class DeepLabV3(nn.Module):
    def __init__(self, num_classes: int = 21) -> None:
        super().__init__()
        # TODO: create backbone, ASPP, and head
        self.backbone = ResNetBackbone([3, 4, 6, 3])
        self.aspp = ASPP(2048, 256, rates=[6, 12, 18])
        self.head = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, 3, H, W)
        # TODO: implement forward with resize to input size
        raise NotImplementedError("DeepLabV3 forward is not implemented.")


def deeplab_v3(num_classes: int = 21) -> DeepLabV3:
    return DeepLabV3(num_classes=num_classes)
