import torch
import torch.nn as nn
from torch import Tensor


class MyLayerNorm(nn.Module):
    def __init__(self, feature_dim: int, eps=1e-5):
        super().__init__()
        self.feature_dim, self.eps = feature_dim, eps
        self.weight = nn.Parameter(torch.ones(feature_dim))
        self.bias = nn.Parameter(torch.zeros(feature_dim))

    def forward(self, x: Tensor) -> Tensor:
        # x: (..., D)
        var, mean = torch.var_mean(x, dim=-1, unbiased=False, keepdim=True)  # (..., 1)
        x_hat = (x - mean) * torch.rsqrt(var + self.eps)

        y = self.weight * x_hat + self.bias  # (..., 1) x (D,) + (D,) = (..., D)
        return y


class MyRMSNorm(nn.Module):
    def __init__(self, feature_dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(feature_dim))

    def forward(self, x: Tensor) -> Tensor:
        # x: (..., D)
        inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x_hat = x * inv_rms

        y = self.weight * x_hat
        return y
