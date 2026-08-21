"""Magnitude-based unstructured pruning across stride-1 conv2 weights.

----
    1. Collect every targeted conv2.weight tensor.
    2. Concatenate |w| over all of them and find the global magnitude threshold
       at the s-th percentile.
    3. Zero out every entry whose magnitude is at or below that threshold.
"""

import torch
from timm.models.resnet import Bottleneck
from torch import nn


def _collect_conv2(model: nn.Module) -> list[nn.Conv2d]:
    """Return all stride-1 conv2 layers from Bottleneck blocks (helper, provided)."""
    convs: list[nn.Conv2d] = []
    for block in model.modules():
        if isinstance(block, Bottleneck) and block.conv2.stride in ((1, 1), 1):
            convs.append(block.conv2)
    return convs


def unstructured_prune_conv2(model: nn.Module, sparsity: float) -> nn.Module:
    """Zero out the smallest-magnitude entries across all targeted conv2 weights.

    Steps to implement:
        1. convs = _collect_conv2(model). If empty or sparsity == 0, return model.
        2. all_mags = torch.cat([c.weight.detach().abs().flatten() for c in convs]).
        3. k = int(sparsity * all_mags.numel());  if k == 0, return model.
        4. threshold = torch.kthvalue(all_mags, k).values
              (k-th smallest magnitude across the global tensor)
        5. For each conv in convs (under torch.no_grad()):
                mask = c.weight.abs() > threshold        # bool tensor, same shape
                c.weight.mul_(mask)                      # in-place zeroing
        6. Return model.
    """
    if not 0 <= sparsity < 1.0:
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")
    # TODO: implement global magnitude pruning. Do NOT use torch.nn.utils.prune.
    print("[unstructured_prune_conv2] not implemented yet — returning model unchanged.")
    return model
