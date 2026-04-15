from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import torchvision.models as models


def conv1x1(in_ch: int, out_ch: int) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

def g_conv3x3(in_ch: int, out_ch: int, dilation: int) -> nn.Conv2d:
    return nn.Conv2d(
        in_ch,
        out_ch,
        kernel_size=3,
        padding=dilation,
        dilation=dilation,
        groups=in_ch,
        bias=False,
    )

def atrous_separable_conv(in_ch, out_ch, dilation):
    return nn.Sequential(
        g_conv3x3(in_ch, in_ch, dilation),
        nn.BatchNorm2d(in_ch),
        nn.ReLU(inplace=True),
        conv1x1(in_ch, out_ch),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class ASPP(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, rates: List[int]) -> None:
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(conv1x1(in_ch, out_ch), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)),
            atrous_separable_conv(in_ch, out_ch, dilation=rates[0]),
            atrous_separable_conv(in_ch, out_ch, dilation=rates[1]),
            atrous_separable_conv(in_ch, out_ch, dilation=rates[2]),
        ])
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            conv1x1(in_ch, out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.project = nn.Sequential(
            conv1x1(out_ch * 5, out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        results = [branch(x) for branch in self.branches]

        h, w = x.shape[-2:]
        res_pool = self.global_pool(x)
        res_pool = F.interpolate(res_pool, size=(h, w), mode='bilinear', align_corners=False)
        results.append(res_pool)

        combined = torch.cat(results, dim=1)
        out = self.project(combined)

        return out
    
class Decoder(nn.Module):
    def __init__(self, in_ch_low: int, in_ch_aspp: int, out_ch: int) -> None:
        super().__init__()

        self.project = nn.Sequential(
            conv1x1(in_ch_low, 48),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        self.refine = nn.Sequential(
            atrous_separable_conv(48 + in_ch_aspp, in_ch_aspp, dilation=1),
            atrous_separable_conv(in_ch_aspp, in_ch_aspp, dilation=1)
        )

        self.final_conv = nn.Conv2d(in_ch_aspp, out_ch, kernel_size=1)

    def forward(self, x_low: Tensor, x_aspp: Tensor) -> Tensor:

        low_level_feat = self.project(x_low)

        h_low, w_low = x_low.shape[-2:]
        aspp_upsampled = F.interpolate(x_aspp, size=(h_low, w_low), mode='bilinear', align_corners=False)

        combined = torch.cat([low_level_feat, aspp_upsampled], dim=1)

        out = self.refine(combined)
        out = self.final_conv(out)
        return out
    
class DeepLabV3PLUS_MobileNet(nn.Module):
    def __init__(self, num_classes: int = 21) -> None:
        super().__init__()
        # FLOPs 절감을 위해 backbone으로 mobilenet 구조를 사용해봄
        mnet = models.mobilenet_v2(weights='IMAGENET1K_V1').features
        self.backbone_low = mnet[:4]
        self.backbone_high = mnet[4:]

        self.aspp = ASPP(1280, 256, rates=[6, 12, 18])
        self.decoder = Decoder(24, 256, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        h_ori, w_ori = x.shape[-2:]

        low_feat = self.backbone_low(x)
        high_feat = self.backbone_high(low_feat)

        aspp_feat = self.aspp(high_feat)
        out = self.decoder(low_feat, aspp_feat)
        out = F.interpolate(out, size=(h_ori, w_ori), mode='bilinear', align_corners=False)
        return out


def deeplab_v3(num_classes: int = 21) -> DeepLabV3PLUS_MobileNet:
    return DeepLabV3PLUS_MobileNet(num_classes=num_classes)