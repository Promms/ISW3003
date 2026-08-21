import detectors  # noqa
import timm
from torch import nn


def load_model(pretrained: bool = True) -> nn.Module:
    model = timm.create_model("resnet50_cifar100", pretrained=pretrained)
    model.eval()
    return model
