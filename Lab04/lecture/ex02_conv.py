import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_pair(val: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(val, int):
        return (val, val)
    return val


class MyConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],  # (height, width)
        stride: int | tuple[int, int],  # (height, width)
        padding: int | tuple[int, int],  # (up-down pad, left-right pad)
        dilation: int | tuple[int, int],  # (height, width)
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        kh, kw = _to_pair(kernel_size)
        self.stride, self.padding = _to_pair(stride), _to_pair(padding)
        self.dilation, self.groups = _to_pair(dilation), groups

        # weight: (Cout, Cin/g, kernel_h, kernel_w)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kh, kw))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None  # (Cout,)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)  # initialize weight
        if self.bias is not None:  # initialize bias if exists
            nn.init.uniform_(self.bias, -0.05, 0.05)

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
