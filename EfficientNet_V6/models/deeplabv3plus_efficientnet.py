from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import torchvision.models as models

from utils.modules import SEBlock


def conv1x1(in_ch: int, out_ch: int) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)


def depthwise_conv3x3(in_ch: int, dilation: int) -> nn.Conv2d:
    return nn.Conv2d(
        in_ch, in_ch,
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
    """
    EfficientNet-B0 features[start:] 블록에 dilation을 적용하여 output stride를 유지합니다.

    EfficientNet-B0 기본 output stride:
      features[:3]  → stride 4  (24ch, low-level feature)
      features[3:6] → stride 16
      features[6:]  → stride 32 ← 여기서 stride=2가 한 번 더 발생

    features[6]의 stride=2를 stride=1 + dilation=2로 교체하면
    output stride가 16으로 유지됩니다.

    EfficientNet은 3x3과 5x5 depthwise conv를 혼용하므로 padding을 커널 크기에 맞게 계산.
    """
    for i in range(start, len(features)):
        for m in features[i].modules():
            if isinstance(m, nn.Conv2d) and m.kernel_size != (1, 1):
                if m.stride == (2, 2):
                    m.stride = (1, 1)
                m.dilation = (dilation, dilation)
                # padding = ((kernel - 1) * dilation) // 2  ← 커널 크기 고려
                kh, kw = m.kernel_size
                m.padding = ((kh - 1) * dilation // 2, (kw - 1) * dilation // 2)


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
        # ASPP 5개 branch를 합친 뒤 채널별 중요도 재보정
        self.se = SEBlock(out_ch, reduction=16)

    def forward(self, x: Tensor) -> Tensor:
        results = [branch(x) for branch in self.branches]

        h, w = x.shape[-2:]
        res_pool = self.global_pool(x)
        res_pool = F.interpolate(res_pool, size=(h, w), mode='bilinear', align_corners=False)
        results.append(res_pool)

        combined = torch.cat(results, dim=1)
        return self.se(self.project(combined))


class Decoder(nn.Module):
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
        low_level_feat = self.project(x_low)

        h_low, w_low = x_low.shape[-2:]
        aspp_upsampled = F.interpolate(x_aspp, size=(h_low, w_low), mode='bilinear', align_corners=False)

        combined = torch.cat([low_level_feat, aspp_upsampled], dim=1)
        out = self.refine(combined)
        return self.final_conv(out)


class DeepLabV3Plus_EfficientNet(nn.Module):
    """
    EfficientNet-B0 백본 기반 DeepLabV3+

    EfficientNet-B0 feature map 구조:
      features[0~2] → stride 4,  24ch  ← low-level feature (decoder skip connection)
      features[3~5] → stride 16, 112ch
      features[6~8] → stride 32, 1280ch ← dilation 적용으로 stride 16 유지
    """
    def __init__(self, num_classes: int = 21) -> None:
        super().__init__()

        enet = models.efficientnet_b0(weights='IMAGENET1K_V1').features

        # stride 4, 24ch — decoder skip connection용 low-level feature
        self.backbone_low = enet[:3]

        # features[6]의 stride=2를 dilation=2로 교체 → output stride 16 유지
        _apply_dilation_to_efficientnet(enet, start=6, dilation=2)
        self.backbone_high = enet[3:]

        # EfficientNet-B0 마지막 채널: 1280ch
        self.aspp = ASPP(1280, 256, rates=[6, 12, 18])

        # low-level: 24ch, aspp: 256ch
        self.decoder = Decoder(24, 256, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        h_ori, w_ori = x.shape[-2:]

        low_feat  = self.backbone_low(x)
        high_feat = self.backbone_high(low_feat)

        aspp_feat = self.aspp(high_feat)
        out = self.decoder(low_feat, aspp_feat)
        out = F.interpolate(out, size=(h_ori, w_ori), mode='bilinear', align_corners=False)
        return out


def deeplab_v3_efficientnet(num_classes: int = 21) -> DeepLabV3Plus_EfficientNet:
    return DeepLabV3Plus_EfficientNet(num_classes=num_classes)
