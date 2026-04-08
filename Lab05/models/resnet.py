import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models import resnet50, ResNet50_Weights


class TruncatedResNet50(nn.Module):
    """
    ResNet-50 backbone truncated after layer2.

    Spatial dimensions for 64x64 Tiny-ImageNet input:
      conv1 (stride 2) + maxpool (stride 2) -> [B, 64, 16, 16]
      layer1 (stride 1)                     -> [B, 256, 16, 16]
      layer2 (stride 2)                     -> [B, 512, 8, 8]
    """

    def __init__(self) -> None:
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        named = dict(backbone.named_children())

        self.initial_conv = nn.Sequential(
            named["conv1"],
            named["bn1"],
            named["relu"],
            named["maxpool"],
        )
        self.layer1 = named["layer1"]
        self.layer2 = named["layer2"]
        # layer3, layer4, avgpool, fc are discarded

    def forward(self, x: Tensor) -> Tensor:
        x = self.initial_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class TinyImageNetClassifier(nn.Module):
    """
    Truncated ResNet-50 backbone + global average pooling + classification head.

    Head uses only Dropout (no augmentations, no BatchNorm in head).
    """

    def __init__(self, num_classes: int = 200, dropout: float = 0.5) -> None:
        super().__init__()
        self.backbone = TruncatedResNet50()
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)          # [B, 512, 8, 8]
        x = x.mean(dim=[2, 3])        # [B, 512] — global average pooling
        x = self.classifier(x)        # [B, num_classes]
        return x


def build_model(num_classes: int = 200, dropout: float = 0.5) -> TinyImageNetClassifier:
    return TinyImageNetClassifier(num_classes=num_classes, dropout=dropout)
