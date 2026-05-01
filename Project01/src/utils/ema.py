from __future__ import annotations

import copy

import torch
import torch.nn as nn


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.ema_model = copy.deepcopy(model).eval()
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        decay = self.decay

        for ema_param, param in zip(self.ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)

        for ema_buffer, buffer in zip(self.ema_model.buffers(), model.buffers()):
            if buffer.dtype.is_floating_point:
                ema_buffer.data.mul_(decay).add_(buffer.data, alpha=1 - decay)
            else:
                ema_buffer.data.copy_(buffer.data)

    def state_dict(self) -> dict:
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self.ema_model.load_state_dict(state_dict)

    def to(self, device: torch.device) -> "ModelEMA":
        self.ema_model.to(device)
        return self
