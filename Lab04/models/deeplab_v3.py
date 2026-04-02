"""DeepLabv3-style segmentation model skeleton for practice."""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    def __init__(self, layers: list):
        super().__init__(layers)
        self.in_channels = 1024
        self.layer4 = self._make_layer(512, layers[3], stride=1)
    def forward_features(self, x: Tensor) -> Tensor:
        # x: (N, 3, H, W)
        # TODO: return final feature map

        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)     

        return x


class ASPP(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, rates: List[int]) -> None:
        super().__init__()
        # TODO: build ASPP branches and projection
        self.branches = nn.ModuleList([
            nn.Sequential(conv3x3(in_ch, out_ch, dilation=rates[0]), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)),
            nn.Sequential(conv3x3(in_ch, out_ch, dilation=rates[1]), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)),
            nn.Sequential(conv3x3(in_ch, out_ch, dilation=rates[2]), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)),
        ])
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            conv1x1(in_ch, out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.project = nn.Sequential(
            conv1x1(out_ch * 4, out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, C, H, W)
        # TODO: implement ASPP forward
        results = [branch(x) for branch in self.branches]

        h, w = x.shape[-2:]
        res_pool = self.global_pool(x)
        res_pool = F.interpolate(res_pool, size=(h, w), mode='bilinear', align_corners=False)
        results.append(res_pool)

        combined = torch.cat(results, dim=1)
        out = self.project(combined)

        return out

class DeepLabV3(nn.Module):
    def __init__(self, num_classes: int = 21) -> None:
        super().__init__()
        # TODO: create backbone, ASPP, and head
        self.backbone = ResNetBackbone([3, 4, 6, 3])
        self.aspp = ASPP(2048, 256, rates=[6, 12, 18])
        self.head = nn.Sequential(
            conv3x3(in_ch=256, out_ch=256, dilation=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            conv1x1(in_ch=256, out_ch=num_classes),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, 3, H, W)
        # TODO: implement forward with resize to input size
        h, w = x.shape[-2:]

        feat = self.backbone.forward_features(x)
        feat = self.aspp(feat)
        out = self.head(feat)
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
        return out


def deeplab_v3(num_classes: int = 21) -> DeepLabV3:
    return DeepLabV3(num_classes=num_classes)
