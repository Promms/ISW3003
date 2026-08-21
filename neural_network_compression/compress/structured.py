"""Structured pruning of stride-1 conv2 input/output channels.

----
For each Bottleneck block whose conv2 has stride == 1:

    1. Score conv2's input channels by absolute weight (L1 norm summed over (out, kh, kw)).
    2. Score conv2's output channels by absolut weight (L1 norm summed over (in,  kh, kw)).
    3. Keep the top (1 - ratio) fraction on each side.
    4. Surgically shrink:
         block.conv1 : keep only the selected OUTPUT channels (= conv2's input)
         block.bn1   : slice channels by keep_in
         block.conv2 : slice BOTH input and output channels
         block.bn2   : slice channels by keep_out
         block.conv3 : keep only the selected INPUT channels (= conv2's output)
       Leave block.conv3.out, block.bn3, and block.downsample untouched.

Helpers (_topk, _slice_conv, _slice_bn) are provided. Your task is to wire them
together inside _prune_block and then iterate over the model.
"""

import torch
from timm.models.resnet import Bottleneck
from torch import nn


def _topk(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Return sorted indices of the top-k entries of `scores`. (helper, provided)"""
    k = max(1, min(k, scores.numel()))
    return torch.topk(scores, k).indices.sort().values


def _slice_conv(
    conv: nn.Conv2d,
    keep_out: torch.Tensor | None,
    keep_in: torch.Tensor | None,
) -> nn.Conv2d:
    """Return a fresh Conv2d whose rows/cols of weight are restricted to keep_out/keep_in.
    (helper, provided)"""
    w = conv.weight.detach()
    if keep_out is not None:
        w = w.index_select(0, keep_out)
    if keep_in is not None:
        w = w.index_select(1, keep_in)
    new_out, new_in, kh, kw = w.shape
    new = nn.Conv2d(
        in_channels=new_in,
        out_channels=new_out,
        kernel_size=(kh, kw),
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
    )
    with torch.no_grad():
        new.weight.copy_(w)
        if conv.bias is not None:
            b = conv.bias.detach()
            if keep_out is not None:
                b = b.index_select(0, keep_out)
            new.bias.copy_(b)
    return new


def _slice_bn(bn: nn.BatchNorm2d, keep: torch.Tensor) -> nn.BatchNorm2d:
    """Return a fresh BatchNorm2d whose channel-wise buffers are restricted to keep.
    (helper, provided)"""
    new = nn.BatchNorm2d(
        num_features=keep.numel(),
        eps=bn.eps,
        momentum=bn.momentum,
        affine=bn.affine,
        track_running_stats=bn.track_running_stats,
    )
    with torch.no_grad():
        if bn.affine:
            new.weight.copy_(bn.weight.detach().index_select(0, keep))
            new.bias.copy_(bn.bias.detach().index_select(0, keep))
        if bn.track_running_stats:
            new.running_mean.copy_(bn.running_mean.detach().index_select(0, keep))
            new.running_var.copy_(bn.running_var.detach().index_select(0, keep))
            new.num_batches_tracked.copy_(bn.num_batches_tracked.detach())
    return new


def _prune_block(block: Bottleneck, ratio: float) -> None:
    """Slice conv1.out / bn1 / conv2 / bn2 / conv3.in inside a single Bottleneck.

    Steps to implement:
        1. conv2 = block.conv2.   Capture out_ch = conv2.out_channels, in_ch = conv2.in_channels.
        2. Compute scores:
               in_scores  = conv2.weight.detach().abs().sum(dim=(0, 2, 3))   # shape (in_ch,)
               out_scores = conv2.weight.detach().abs().sum(dim=(1, 2, 3))   # shape (out_ch,)
        3. Pick the kept indices:
               keep_in  = _topk(in_scores,  round(in_ch  * (1 - ratio)))
               keep_out = _topk(out_scores, round(out_ch * (1 - ratio)))
        4. Replace four modules (use the helpers above):
               block.conv1 = _slice_conv(block.conv1, keep_out=keep_in, keep_in=None)
               block.bn1   = _slice_bn(block.bn1, keep_in)
               block.conv2 = _slice_conv(block.conv2, keep_out=keep_out, keep_in=keep_in)
               block.bn2   = _slice_bn(block.bn2, keep_out)
               block.conv3 = _slice_conv(block.conv3, keep_out=None, keep_in=keep_out)
        5. Do NOT touch block.conv3.out, block.bn3, or block.downsample.
    """
    # TODO: implement steps 1-5 above.
    raise NotImplementedError("compress/structured.py: implement _prune_block")


def structured_prune_conv2(model: nn.Module, ratio: float) -> nn.Module:
    """Apply _prune_block to every Bottleneck whose conv2 has stride == 1."""
    if not 0 <= ratio < 1.0:
        raise ValueError(f"ratio must be in [0, 1), got {ratio}")
    # TODO: iterate model.modules(), find Bottleneck blocks with stride-1 conv2,
    # and call _prune_block(block, ratio) on each.
    print("[structured_prune_conv2] not implemented yet — returning model unchanged.")
    return model
