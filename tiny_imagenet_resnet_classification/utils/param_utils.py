from collections import OrderedDict
import torch.nn as nn


def count_parameters(model: nn.Module) -> dict[str, int]:
    """
    Count parameters for the model and each top-level child module.

    Returns an OrderedDict with keys:
      "total"           — total parameter count
      "trainable"       — trainable-only count
      "layer/<name>"    — total count per top-level named child
    """
    stats: dict[str, int] = OrderedDict()
    stats["total"]     = sum(p.numel() for p in model.parameters())
    stats["trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)

    for name, module in model.named_children():
        stats[f"layer/{name}"] = sum(p.numel() for p in module.parameters())

    return stats


def log_parameter_counts(model: nn.Module) -> dict[str, int]:
    """Print and return parameter counts. Call once at training start."""
    stats = count_parameters(model)
    print("=== Parameter Counts ===")
    for key, val in stats.items():
        print(f"  {key:35s}: {val:>12,d}")
    return stats
