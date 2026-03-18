"""
pr2_weight.py
Practice 2: Weight inspection and pruning on ResNet34 from torchvision.

Tasks:
  (1) Among nn.Conv2d layers with kernel_size == 1, count their parameters
      and measure sparsity (fraction of zero-valued parameters).
  (2) For those same layers, zero out the bottom 10% of weights by absolute magnitude.
  (3) Measure the overall sparsity of the entire model after pruning.
"""

import torch.nn as nn
from torchvision.models import resnet34


def load_model() -> nn.Module:
    """
    Load a pretrained ResNet34 model from torchvision.
    Pretrained weights are used so the weight values are realistic.

    Returns:
        model (nn.Module): ResNet34 model with pretrained weights.
    """
    model = resnet34(weights=None)
    return model


def get_3x3_conv_layers(model: nn.Module):
    """
    Collect all nn.Conv2d submodules whose kernel_size is (3, 3).
    torchvision's ResNet34 uses 3x3 Conv2d in its downsample (shortcut) blocks.

    Args:
        model (nn.Module): The model to search.

    Returns:
        list[tuple[str, nn.Conv2d]]: (name, module) pairs for matching layers.
    """
    return None


def count_and_measure_sparsity(conv_layers: list, verbose: bool = True) -> int:
    """
    (1) Count the total number of parameters in the given 3x3 Conv2d layers
    and compute their sparsity (proportion of elements equal to zero).

    Args:
        conv_layers (list[tuple[str, nn.Conv2d]]): Layers to inspect.

    Returns:
        total_count (int): Total number of scalar parameters across all layers.
    """
    return 0


def prune_bottom_10_percent(conv_layers: list, verbose: bool = True):
    """
    (2) For each 3x3 Conv2d layer, zero out the 10% of weight elements
    with the smallest absolute magnitude (magnitude-based unstructured pruning).
    Bias parameters are left unchanged.

    Args:
        conv_layers (list[tuple[str, nn.Conv2d]]): Layers to prune.
    """
    pass


def measure_global_sparsity(model: nn.Module):
    """
    (3) Measure the sparsity of the entire model (all parameters combined).
    Sparsity = (number of zero elements) / (total number of elements).

    Args:
        model (nn.Module): The model to evaluate.
    """
    total_elements = 0
    total_zeros = 0

    for param in model.parameters():
        total_elements += param.numel()
        total_zeros += (param.data == 0).sum().item()

    sparsity = total_zeros / total_elements if total_elements > 0 else 0.0

    print("=" * 60)
    print("[Task 3] Global model sparsity after pruning")
    print("=" * 60)
    print(f"  Total parameters : {total_elements}")
    print(f"  Zero parameters  : {total_zeros}")
    print(f"  Global sparsity  : {sparsity:.4f}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Load pretrained ResNet34
    model = load_model()

    # Collect all 3x3 Conv2d layers
    conv3x3_layers = get_3x3_conv_layers(model)

    # (1) Count parameters and measure sparsity of 1x1 Conv2d layers
    _ = count_and_measure_sparsity(conv3x3_layers)

    # (2) Zero out the bottom 10% of weights by absolute magnitude
    prune_bottom_10_percent(conv3x3_layers)

    # (3) Measure global sparsity of the entire model
    measure_global_sparsity(model)