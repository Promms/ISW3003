import torch
import torch.nn as nn
from torch import Tensor


class MyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))  # (Dout, Din)
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None  # (Dout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)  # initialize weight
        if self.bias is not None:  # initialize bias if exists
            nn.init.uniform_(self.bias, -0.05, 0.05)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, ..., Din)
        y = x @ self.weight.T  # (B, ..., Din) x (Din, Dout) = (B, ..., Dout)

        if self.bias is not None:
            y = y + self.bias  # (B, ..., Dout) + (Dout,)

        return y
