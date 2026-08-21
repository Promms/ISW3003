import io

import torch
from torch import nn


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_zeros(model: nn.Module) -> int:
    return sum((p == 0).sum().item() for p in model.parameters())


def model_size_mb(model: nn.Module) -> float:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / (1024 * 1024)
