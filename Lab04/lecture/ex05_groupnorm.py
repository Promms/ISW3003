import torch
import torch.nn as nn
from torch import Tensor


class MyGroupNorm(nn.Module):
    def __init__(self, num_groups: int, num_channels: int, eps=1e-5):
        super().__init__()
        self.num_groups, self.eps = num_groups, eps
        self.weight = nn.Parameter(torch.ones(num_channels))  # initialize to 1
        self.bias = nn.Parameter(torch.zeros(num_channels))  # initialize to 0

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        G = self.num_groups
        xg = x.view(B, G, C // G, H, W)  # (B, G, C/G, H, W)

        mean = xg.mean(dim=[2, 3, 4], keepdim=True)  # (B, G, 1, 1, 1)
        var = xg.var(dim=[2, 3, 4], unbiased=False, keepdim=True)  # (B, G, 1, 1, 1)
        x_hat = (xg - mean) * torch.rsqrt(var + self.eps)  # (B, G, C/G, H, W)
        x_hat = x_hat.view(B, C, H, W)

        y = self.weight.view(1, C, 1, 1) * x_hat + self.bias.view(1, C, 1, 1)
        return y
