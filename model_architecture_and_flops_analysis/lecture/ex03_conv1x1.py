import torch
import torch.nn as nn


conv1x1_layer = nn.Conv2d(16, 32, kernel_size=1, stride=1)
linear_layer = nn.Linear(16, 32)
with torch.no_grad():
    # (Dout, Din) <- (Cout, Cin, 1, 1)
    linear_layer.weight.data.copy_(conv1x1_layer.weight.data.squeeze())
    linear_layer.bias.data.copy_(conv1x1_layer.bias.data)

x_in = torch.ones(4, 16, 64, 64)  # (B, C, H, W)
x_out_conv1x1 = conv1x1_layer(x_in)
print(x_out_conv1x1.shape)  # (B, C, H, W) = (4, 32, 64, 64)

x_in_reshape = x_in.permute(0, 2, 3, 1)  # (B, H, W, C)
x_out_linear = linear_layer(x_in_reshape.contiguous())
print(x_out_linear.shape)  # (B, H, W, C) = (4, 64, 64, 32)
x_out_reshape = x_out_linear.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

diff = (x_out_conv1x1 - x_out_reshape).abs().sum()
print(f"Difference: {diff}")  # 0.0
