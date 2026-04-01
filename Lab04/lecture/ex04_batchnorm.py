import torch
import torch.nn as nn
from torch import Tensor


class MyBatchNorm2d(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps, self.momentum = eps, momentum
        self.weight = nn.Parameter(torch.ones(num_features))  # intialize with 1
        self.bias = nn.Parameter(torch.zeros(num_features))  # initialize with 0
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, H, W)
        if self.training:
            mean = x.mean(dim=[0, 2, 3])  # (C,)
            var = x.var(dim=[0, 2, 3], unbiased=False)  # (C,)
            self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean)
            self.running_var.mul_(1 - self.momentum).add_(self.momentum * var)
        else:
            mean, var = self.running_mean, self.running_var

        mean = mean.view(1, -1, 1, 1)  # (1, C, 1, 1)
        var = var.view(1, -1, 1, 1)  # (1, C, 1, 1)
        inv_std = torch.rsqrt(var + self.eps)  # (1, C, 1, 1)
        x_hat = (x - mean) * inv_std

        y = self.weight.view(1, -1, 1, 1) * x_hat + self.bias.view(1, -1, 1, 1)
        return y
