"""FLOPs computation utilities using forward hooks."""

from typing import Dict, Tuple

import torch
import torch.nn as nn


def _conv2d_flops(layer: nn.Conv2d, output: torch.Tensor) -> int:
    # Output shape: (N, Cout, Hout, Wout)
    n, cout, hout, wout = output.shape
    kernel_ops = layer.kernel_size[0] * layer.kernel_size[1] * (layer.in_channels // layer.groups)
    # Per output element: kernel_ops multiplications and additions ~ kernel_ops.
    # We count multiply-add as 1 FLOP for simplicity.
    return int(n * cout * hout * wout * kernel_ops)


def _linear_flops(layer: nn.Linear, output: torch.Tensor) -> int:
    # Output shape: (N, out_features)
    n = output.shape[0]
    return int(n * layer.in_features * layer.out_features)


def compute_flops(
    model: nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
    device: str = "cpu",
) -> int:
    """Compute FLOPs for Conv2d and Linear layers only.

    Bias terms are ignored by design.

    Args:
        model: PyTorch module.
        input_size: Input tensor shape (N, C, H, W).
        device: Device for dummy forward.

    Returns:
        Total FLOPs.
    """

    flops: Dict[int, int] = {}
    hooks = []

    def conv_hook(layer: nn.Module, _inp, out):
        flops[id(layer)] = _conv2d_flops(layer, out)

    def linear_hook(layer: nn.Module, _inp, out):
        flops[id(layer)] = _linear_flops(layer, out)

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    was_training = model.training
    model.eval()

    with torch.no_grad():
        dummy = torch.zeros(input_size, device=device)
        model.to(device)(dummy)

    for hook in hooks:
        hook.remove()

    if was_training:
        model.train()

    return int(sum(flops.values()))
