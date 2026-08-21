import logging

import torch
from fvcore.nn import FlopCountAnalysis
from torch import nn


def compute_flops(model: nn.Module, input_size: tuple[int, int, int, int] = (1, 3, 32, 32)) -> int:
    """Return MACs for a single forward pass. fvcore reports multiply-accumulates."""
    logging.getLogger("fvcore").setLevel(logging.ERROR)
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    dummy = torch.randn(*input_size, device=device)
    with torch.no_grad():
        flops = FlopCountAnalysis(model, dummy)
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        total = flops.total()
    if was_training:
        model.train()
    return int(total)
