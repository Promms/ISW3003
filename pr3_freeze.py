"""
pr3_freeze.py
Practice 3: Freezing parts of a ResNet34 model from torchvision.

Tasks:
  (1) Freeze the first 2 stages (layer1 and layer2) — the layers closest
      to the input — by setting requires_grad=False on their parameters.
  (2) Additionally freeze all 1-D parameters (e.g., BN weight/bias, Conv bias)
      across the entire model, then report the number of trainable parameters.
"""

import torch.nn as nn
from torchvision.models import resnet34


def load_model() -> nn.Module:
    """
    Load a ResNet34 model from torchvision without pretrained weights.

    Returns:
        model (nn.Module): ResNet34 model with random initialization.
    """
    model = resnet34(weights=None)
    return model


def freeze_first_two_stages(model: nn.Module, verbose: bool = True):
    """
    (1) Freeze the first 2 residual stages (layer1, layer2).

    In torchvision's ResNet34 the stages closest to the input are:
        conv1, bn1  — stem (not a "stage" per se)
        layer1      — stage 1
        layer2      — stage 2
        layer3      — stage 3
        layer4      — stage 4

    We freeze layer1 and layer2 by setting requires_grad=False on all
    their parameters, which prevents gradient computation and weight updates.

    Args:
        model (nn.Module): The model to modify in-place.
    """
    # Stages to freeze (closest to input first)
    stages_to_freeze = [model.layer1, model.layer2]

    for stage in stages_to_freeze:
        for param in stage.parameters():
            param.requires_grad = False  # freeze: detach from the computation graph

    print("=" * 60)
    print("[Task 1] Freezing layer1 and layer2 (first 2 stages)")
    print("=" * 60)

    if verbose:
        # Show requires_grad status per named parameter
        for name, param in model.named_parameters():
            if "layer1" in name or "layer2" in name:
                print(f"  {name:60s}  requires_grad={param.requires_grad}")
    print()


def freeze_1d_parameters(model: nn.Module, verbose: bool = True):
    """
    (2) Additionally freeze all 1-D parameters across the entire model.

    1-D parameters include:
        - BatchNorm weight (gamma) and bias (beta)
        - Conv2d / Linear bias vectors

    After freezing, count and report the number of still-trainable parameters.

    Args:
        model (nn.Module): The model to modify in-place.
    """
    pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Load ResNet34 without pretrained weights
    model = load_model()

    # (1) Freeze the first 2 stages (layer1, layer2)
    freeze_first_two_stages(model)

    # (2) Freeze all 1-D parameters and report trainable parameter count
    freeze_1d_parameters(model)