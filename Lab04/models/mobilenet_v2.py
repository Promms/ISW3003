"""MobileNetV2 skeleton for practice."""

# import torch
import torch.nn as nn
from torch import Tensor


def conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_ch: int, out_ch: int) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)


class InvertedResidual(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int, expand_ratio: int) -> None:
        super().__init__()
        # TODO: implement inverted residual block

        expand_ch = in_ch * expand_ratio

        self.conv1 = nn.Conv2d(in_ch, expand_ch, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_ch)
        self.relu1 = nn.ReLU6(inplace=True)

        self.conv2 = nn.Conv2d(expand_ch, expand_ch, kernel_size=3, stride=stride, 
                            padding=1, groups=expand_ch, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_ch)
        self.relu2 = nn.ReLU6(inplace=True)

        self.conv3 = nn.Conv2d(expand_ch, out_ch, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)

        self.stride = stride
        self.in_ch = in_ch
        self.out_ch = out_ch

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, C, H, W)
        # TODO: implement forward
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.stride == 1 and self.in_ch == self.out_ch:
            return identity + out
        else:
            return out


class MobileNetV2(nn.Module):
    def __init__(self, num_classes: int = 1000, width_mult: float = 1.0) -> None:
        super().__init__()
        # TODO: implement MobileNetV2 architecture

        self.config = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )

        curr_ch = 32

        features = []
        for t, c, n, s in self.config:
            output_ch = int(c * width_mult)
            features.append(self._make_layer(curr_ch, output_ch, s, t, n))
            curr_ch = output_ch

        self.layers = nn.Sequential(*features)

        last_ch = int(1280 * width_mult)
        self.last_conv = nn.Sequential(
            nn.Conv2d(curr_ch, last_ch, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(last_ch),
            nn.ReLU6(inplace=True)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(1280, num_classes)

    def _make_layer(self, in_ch: int, out_ch: int, stride: int, expand_ratio: int, blocks: int) -> nn.Sequential:
        layers = []
        layers.append(InvertedResidual(in_ch,out_ch,stride,expand_ratio))

        for _ in range(1, blocks):
            layers.append(InvertedResidual(out_ch,out_ch,1,expand_ratio))

        return nn.Sequential(*layers)



    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.layers(x)
        x = self.last_conv(x)
        
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


def mobilenet_v2(num_classes: int = 1000) -> MobileNetV2:
    return MobileNetV2(num_classes=num_classes)