"""SVD-based low-rank factorization of 3x3 conv2 layers in stride-1 Bottlenecks.

----
For W in R^(out x in x k x k), reshape to (out, in*k*k), then take a truncated
SVD with rank r:

    W ~= U_r @ diag(s_r) @ V_r^T = A @ B
        A in R^(out x r)
        B in R^(r, in*k*k)

The conv is then factored into two sequential convolutions:
    conv_a: in -> r,   kxk, original stride/padding   (carries the spatial op)
    conv_b: r  -> out, 1x1                            (recombines channels)

Parameter cost shrinks from in*out*k*k to in*r*k*k + r*out.
"""

import torch
from timm.models.resnet import Bottleneck
from torch import nn


def _decompose_conv(conv: nn.Conv2d, rank: int) -> nn.Sequential:
    """Return an nn.Sequential of two convs that approximates `conv` with the given rank.

    Steps to implement:
        1. Read out the weight shape (out_ch, in_ch, kh, kw).
        2. Reshape conv.weight to a 2-D matrix W of shape (out_ch, in_ch * kh * kw).
        3. Run torch.linalg.svd(W, full_matrices=False) -> U, S, Vh.
        4. Clamp `rank` into a valid range and take the top-`rank` components:
               S = S[:rank]
               A = U[:, :rank]             shape (out_ch, rank)
               B = S[:, None] * Vh[:rank]  shape (rank,   in_ch*kh*kw)
        5. Build two nn.Conv2d layers:
               conv_a : Conv2d(in_ch, rank, (kh, kw),
                               stride=conv.stride, padding=conv.padding,
                               dilation=conv.dilation, groups=conv.groups, bias=False)
               conv_b : Conv2d(rank, out_ch, 1, bias=(conv.bias is not None))
        6. Copy the reshaped tensors into their .weight (and bias) under torch.no_grad().
               conv_a.weight  <- B.reshape(rank, in_ch, kh, kw)
               conv_b.weight  <- A.reshape(out_ch, rank, 1, 1)
               conv_b.bias    <- conv.bias        (if conv.bias is not None)
        7. Return nn.Sequential(conv_a, conv_b).
    """
    # TODO: implement steps 1-7 above.
    raise NotImplementedError("compress/factorization.py: implement _decompose_conv")


def factorize_conv2(model: nn.Module, rank_ratio: float) -> nn.Module:
    """Replace every stride-1 conv2 inside Bottleneck blocks with a low-rank factorization.

    Steps to implement:
        1. Iterate over model.modules() and pick out timm Bottleneck blocks
           whose conv2.stride is (1, 1) or 1.
        2. Determine target rank from rank_ratio:
               full_rank = min(out_ch, in_ch * kh * kw)
               rank      = max(1, round(full_rank * rank_ratio))
        3. block.conv2 = _decompose_conv(block.conv2, rank)
        4. Return the (mutated) model.
    """
    if not 0 < rank_ratio <= 1.0:
        raise ValueError(f"rank_ratio must be in (0, 1], got {rank_ratio}")
    # TODO: walk the model, locate Bottleneck blocks with stride-1 conv2,
    # and replace block.conv2 with its decomposed Sequential.
    print("[factorize_conv2] not implemented yet — returning model unchanged.")
    return model
