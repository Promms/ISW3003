from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch import Tensor

from utils.attention import SEBlock


def conv1x1(in_ch: int, out_ch: int) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)


def depthwise_conv3x3(in_ch: int, dilation: int) -> nn.Conv2d:
    return nn.Conv2d(
        in_ch,
        in_ch,
        kernel_size=3,
        padding=dilation,
        dilation=dilation,
        groups=in_ch,
        bias=False,
    )


def depthwise_separable_conv(in_ch: int, out_ch: int, dilation: int) -> nn.Sequential:
    return nn.Sequential(
        depthwise_conv3x3(in_ch, dilation),
        nn.BatchNorm2d(in_ch),
        nn.ReLU(inplace=True),
        conv1x1(in_ch, out_ch),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def _apply_dilation_to_efficientnet(features: nn.Sequential, start: int, dilation: int) -> None:
    """Replace the final downsampling stage with dilation to keep output stride 16."""
    for i in range(start, len(features)):
        for module in features[i].modules():
            if isinstance(module, nn.Conv2d) and module.kernel_size != (1, 1):
                if module.stride == (2, 2):
                    module.stride = (1, 1)
                module.dilation = (dilation, dilation)
                kh, kw = module.kernel_size
                module.padding = ((kh - 1) * dilation // 2, (kw - 1) * dilation // 2)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling with depthwise separable branches."""

    def __init__(self, in_ch: int, out_ch: int, rates: tuple[int, int, int]) -> None:
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(conv1x1(in_ch, out_ch), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)),
            depthwise_separable_conv(in_ch, out_ch, dilation=rates[0]),
            depthwise_separable_conv(in_ch, out_ch, dilation=rates[1]),
            depthwise_separable_conv(in_ch, out_ch, dilation=rates[2]),
        ])
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            conv1x1(in_ch, out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            conv1x1(out_ch * 5, out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.se = SEBlock(out_ch, reduction=16)

    def forward(self, x: Tensor) -> Tensor:
        height, width = x.shape[-2:]
        pooled = self.global_pool(x)
        pooled = F.interpolate(pooled, size=(height, width), mode="bilinear", align_corners=False)
        features = [branch(x) for branch in self.branches] + [pooled]
        return self.se(self.project(torch.cat(features, dim=1)))


class Decoder(nn.Module):
    """DeepLabV3+ decoder using the stride-4 EfficientNet feature map."""

    def __init__(self, in_ch_low: int, in_ch_aspp: int, out_ch: int) -> None:
        super().__init__()
        self.project = nn.Sequential(
            conv1x1(in_ch_low, 48),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.refine = nn.Sequential(
            depthwise_separable_conv(48 + in_ch_aspp, in_ch_aspp, dilation=1),
            depthwise_separable_conv(in_ch_aspp, in_ch_aspp, dilation=1),
        )
        self.final_conv = nn.Conv2d(in_ch_aspp, out_ch, kernel_size=1)

    def forward(self, x_low: Tensor, x_aspp: Tensor) -> Tensor:
        low = self.project(x_low)
        x_aspp = F.interpolate(x_aspp, size=x_low.shape[-2:], mode="bilinear", align_corners=False)
        return self.final_conv(self.refine(torch.cat([low, x_aspp], dim=1)))


class DeepLabV3Plus_EfficientNet(nn.Module):
    """DeepLabV3+ segmentation head on a TorchVision EfficientNet-B1 backbone."""

    def __init__(self, num_classes: int = 21, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.EfficientNet_B1_Weights.IMAGENET1K_V1 if pretrained else None
        features = models.efficientnet_b1(weights=weights).features

        self.backbone_low = features[:3]
        _apply_dilation_to_efficientnet(features, start=6, dilation=2)
        self.backbone_high = features[3:]

        self.aspp = ASPP(1280, 256, rates=(6, 12, 18))
        self.decoder = Decoder(24, 256, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        output_size = x.shape[-2:]
        low = self.backbone_low(x)
        high = self.backbone_high(low)
        logits = self.decoder(low, self.aspp(high))
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)


def deeplab_v3_efficientnet(
    num_classes: int = 21,
    pretrained: bool = True,
) -> DeepLabV3Plus_EfficientNet:
    return DeepLabV3Plus_EfficientNet(num_classes=num_classes, pretrained=pretrained)
