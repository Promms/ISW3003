"""Utilities for counting model parameters."""

import torch.nn as nn


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Count parameters in a model.

    Args:
        model: PyTorch module.
        trainable_only: If True, count only parameters with requires_grad.

    Returns:
        Total number of parameters.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())
