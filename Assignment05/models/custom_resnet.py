import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models import resnet50, ResNet50_Weights


class ImagenetteBackbone(nn.Module):
    """
    Spatial dimensions for H, W ImageNette input:
      conv1 (stride 2) + maxpool (stride 2) -> [B, 64, H/4, W/4]
      layer1 (stride 1)                     -> [B, 256, H/4, W/4]
      layer2 (stride 2)                     -> [B, 512, H/8, W/8]
      layer3 (stride 2)                     -> [B, 1024, H/16, W/16]
    """

    def __init__(self) -> None:
        super().__init__()
        # 외부 parameter를 가져옴
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        named = dict(backbone.named_children())
        # Stem, layer1, layer2, layer3만 셋팅
        self.initial_conv = nn.Sequential(
            named["conv1"],
            named["bn1"],
            named["relu"],
            named["maxpool"],
        )
        self.layer1 = named["layer1"]
        self.layer2 = named["layer2"]
        self.layer3 = named["layer3"]
        # layer4, avgpool, fc are discarded

        self.more_layers = nn.Sequential(
            # Pointwise
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # Depthwise
            nn.Conv2d(512, 512, kernel_size=3, padding=1, groups=512),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # Pointwise
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.initial_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.more_layers(x)
        return x


class ImagenetteClassifier(nn.Module):
    """
    Imagenette backbone + global average pooling + classification head.

    Head uses only Dropout (no augmentations, no BatchNorm in head).
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.5) -> None:
        super().__init__()
        self.backbone = ImagenetteBackbone()
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)          # [B, 1024, H/16, W/16]
        x = x.mean(dim=[2, 3])        # [B, 1024] — global average pooling
        x = self.classifier(x)        # [B, num_classes]
        return x


def build_model(num_classes: int = 200, dropout: float = 0.5) -> ImagenetteClassifier:
    return ImagenetteClassifier(num_classes=num_classes, dropout=dropout)
