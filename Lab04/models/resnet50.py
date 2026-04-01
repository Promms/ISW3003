"""ResNet-50 implementation for practice."""

from typing import List

# import torch
import torch.nn as nn
from torch import Tensor

# 3x3 conv를 만들기 위한 helper func
def conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)

# 
class Bottleneck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        mid_channels = channels

        #conv -> bn -> conv -> bn -> conv -> bn -> relu
        self.conv1 = conv1x1(in_channels, mid_channels)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = conv3x3(mid_channels, mid_channels, stride=stride)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = conv1x1(mid_channels, channels * 4)
        self.bn3 = nn.BatchNorm2d(channels * 4)
        self.relu = nn.ReLU()

        self.downsample = None
        # stride가 달라서 size가 달라지거나, channel이 변해서 달라지면
        if stride != 1 or in_channels != channels * 4:
            self.downsample = nn.Sequential(
                conv1x1(in_channels, channels * 4, stride),
                nn.BatchNorm2d(channels * 4),
            )
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, C, H, W)
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet50(nn.Module):
    def __init__(
        self,
        layers: List[int],
        num_classes: int = 1000,
    ) -> None:
        super().__init__()

        self.in_channels = 64
        self.dilation = 1

        # Stem convolution block
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stages
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

        # Output projection
        self.fc = nn.Linear(512 * 4, num_classes)

    def _make_layer(
        self,
        channels: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        layers = []
        layers.append(
            Bottleneck(
                self.in_channels,
                channels,
                stride=stride,
            )
        )
        self.in_channels = channels * 4
        for _ in range(1, blocks):
            layers.append(
                Bottleneck(
                    self.in_channels,
                    channels,
                    stride=1,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, 3, H, W)

        # Stem convolution block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.mean(dim=(2, 3))  # (B, C, H, W) -> (B, C)
        x = self.fc(x)
        # x: (N, num_classes)
        return x


def resnet50(num_classes: int = 1000) -> ResNet50:
    return ResNet50([3, 4, 6, 3], num_classes=num_classes)
