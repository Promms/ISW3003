from __future__ import annotations

from collections import OrderedDict

import torch.nn as nn


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Count total, trainable, and top-level module parameters."""
    stats: dict[str, int] = OrderedDict()
    stats["total"] = sum(param.numel() for param in model.parameters())
    stats["trainable"] = sum(param.numel() for param in model.parameters() if param.requires_grad)

    for name, module in model.named_children():
        stats[f"layer/{name}"] = sum(param.numel() for param in module.parameters())
    return stats


def log_parameter_counts(model: nn.Module) -> dict[str, int]:
    stats = count_parameters(model)
    print("=== Parameter Counts ===")
    for key, value in stats.items():
        print(f"  {key:35s}: {value:>12,d}")
    return stats
