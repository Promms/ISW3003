"""
pr2_weight.py
Practice 2: Weight inspection and pruning on ResNet34 from torchvision.

Tasks:
  (1) Among nn.Conv2d layers with kernel_size == 1, count their parameters
      and measure sparsity (fraction of zero-valued parameters).
  (2) For those same layers, zero out the bottom 10% of weights by absolute magnitude.
  (3) Measure the overall sparsity of the entire model after pruning.
"""
import torch
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
    result = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # kernel_size is stored as a tuple, e.g., (1, 1) or (3, 3)
            if module.kernel_size == (3, 3):
                result.append((name, module))
    return result


def count_and_measure_sparsity(conv_layers: list, verbose: bool = True) -> int:
    """
    (1) Count the total number of parameters in the given 3x3 Conv2d layers
    and compute their sparsity (proportion of elements equal to zero).

    Args:
        conv_layers (list[tuple[str, nn.Conv2d]]): Layers to inspect.

    Returns:
        total_count (int): Total number of scalar parameters across all layers.
    """
    total_count = 0
    total_zeros = 0

    print("=" * 60)
    print("[Task 1] 3x3 Conv2d layers — parameter count & sparsity")
    print("=" * 60)

    for name, module in conv_layers:
        # Iterate over all parameters in this Conv2d (weight + bias if present)
        for param_name, param in module.named_parameters():
            num_elements = param.numel()
            num_zeros = (param.data == 0).sum().item()

            total_count += num_elements
            total_zeros += num_zeros

            sparsity = num_zeros / num_elements
            if verbose:
                print(f"  {name}.{param_name:10s}  "
                    f"params={num_elements:8d}  sparsity={sparsity:.4f}")

    overall_sparsity = total_zeros / total_count if total_count > 0 else 0.0
    print(f"\n  Total params in 3x3 Conv2d : {total_count}")
    print(f"  Overall sparsity           : {overall_sparsity:.4f}")
    print()

    return total_count


def prune_bottom_10_percent(conv_layers: list, verbose: bool = True):
    """
    (2) For each 3x3 Conv2d layer, zero out the 10% of weight elements
    with the smallest absolute magnitude (magnitude-based unstructured pruning).
    Bias parameters are left unchanged.

    Args:
        conv_layers (list[tuple[str, nn.Conv2d]]): Layers to prune.
    """
    print("=" * 60)
    print("[Task 2] Zeroing out bottom 10% by absolute magnitude (weights only)")
    for name, layer in conv_layers:
        weight = layer.weight
        threshold = torch.quantile(weight.abs(), 0.1)
        mask = weight < threshold
        weight.data[mask] = 0
    print("=" * 60)



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
    count_and_measure_sparsity(conv3x3_layers)

    # (2) Zero out the bottom 10% of weights by absolute magnitude
    prune_bottom_10_percent(conv3x3_layers)

    # (3) Measure global sparsity of the entire model
    measure_global_sparsity(model)