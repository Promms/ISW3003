from __future__ import annotations

from typing import List

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch
from torch import Tensor

from utils.blocks import SEBlock


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


def apply_output_stride_16(features: nn.Sequential, start: int, dilation: int = 2) -> None:
    """Replace the final stride-2 stage with dilation to keep output stride 16."""
    for i in range(start, len(features)):
        for layer in features[i].modules():
            if isinstance(layer, nn.Conv2d) and layer.kernel_size != (1, 1):
                if layer.stride == (2, 2):
                    layer.stride = (1, 1)
                layer.dilation = (dilation, dilation)
                kh, kw = layer.kernel_size
                layer.padding = ((kh - 1) * dilation // 2, (kw - 1) * dilation // 2)


def build_mobilenet_backbone(
    backbone: str,
    pretrained: bool,
) -> tuple[nn.Sequential, nn.Sequential, int, int]:
    """Return low/high feature modules plus their channel counts."""
    if backbone == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        features = models.mobilenet_v2(weights=weights).features
        low = features[:4]       # stride 4, 24 channels
        apply_output_stride_16(features, start=14)
        high = features[4:]      # stride 16 after dilation, 1280 channels
        return low, high, 24, 1280

    if backbone == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        features = models.mobilenet_v3_large(weights=weights).features
        low = features[:4]       # stride 4, 24 channels
        apply_output_stride_16(features, start=13)
        high = features[4:]      # stride 16 after dilation, 960 channels
        return low, high, 24, 960

    raise ValueError(
        f"Unknown backbone '{backbone}'. Expected 'mobilenet_v2' or 'mobilenet_v3_large'."
    )


class ASPP(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, rates: List[int]) -> None:
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
        h, w = x.shape[-2:]
        results = [branch(x) for branch in self.branches]
        pooled = self.global_pool(x)
        pooled = F.interpolate(pooled, size=(h, w), mode="bilinear", align_corners=False)
        results.append(pooled)
        return self.se(self.project(torch.cat(results, dim=1)))


class Decoder(nn.Module):
    def __init__(
        self,
        in_ch_low: int,
        in_ch_aspp: int,
        out_ch: int,
        low_project_ch: int,
    ) -> None:
        super().__init__()
        self.project = nn.Sequential(
            conv1x1(in_ch_low, low_project_ch),
            nn.BatchNorm2d(low_project_ch),
            nn.ReLU(inplace=True),
        )
        self.refine = nn.Sequential(
            depthwise_separable_conv(low_project_ch + in_ch_aspp, in_ch_aspp, dilation=1),
            depthwise_separable_conv(in_ch_aspp, in_ch_aspp, dilation=1),
        )
        self.final_conv = nn.Conv2d(in_ch_aspp, out_ch, kernel_size=1)

    def forward(self, x_low: Tensor, x_aspp: Tensor) -> Tensor:
        low = self.project(x_low)
        x_aspp = F.interpolate(x_aspp, size=x_low.shape[-2:], mode="bilinear", align_corners=False)
        return self.final_conv(self.refine(torch.cat([low, x_aspp], dim=1)))


class DeepLabV3Plus(nn.Module):
    def __init__(
        self,
        num_classes: int = 21,
        backbone: str = "mobilenet_v2",
        aspp_channels: int = 224,
        decoder_low_channels: int = 48,
        pretrained_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.aspp_channels = aspp_channels
        self.backbone_low, self.backbone_high, low_ch, high_ch = build_mobilenet_backbone(
            backbone=backbone,
            pretrained=pretrained_backbone,
        )
        self.aspp = ASPP(high_ch, aspp_channels, rates=[6, 12, 18])
        self.decoder = Decoder(low_ch, aspp_channels, num_classes, decoder_low_channels)

    def forward(self, x: Tensor) -> Tensor:
        out_size = x.shape[-2:]
        low_feat = self.backbone_low(x)
        high_feat = self.backbone_high(low_feat)
        aspp_feat = self.aspp(high_feat)
        out = self.decoder(low_feat, aspp_feat)
        return F.interpolate(out, size=out_size, mode="bilinear", align_corners=False)


def deeplab_v3(
    num_classes: int = 21,
    backbone: str = "mobilenet_v2",
    aspp_channels: int = 224,
    decoder_low_channels: int = 48,
    pretrained_backbone: bool = True,
) -> DeepLabV3Plus:
    return DeepLabV3Plus(
        num_classes=num_classes,
        backbone=backbone,
        aspp_channels=aspp_channels,
        decoder_low_channels=decoder_low_channels,
        pretrained_backbone=pretrained_backbone,
    )
