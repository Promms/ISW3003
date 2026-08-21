"""Adversarial losses and exact R1/R2 gradient penalty backend.

The discriminator is piecewise-linear (only BiasLeakyReLU activations; ConvLayer
has no bias). For a fixed activation mask:

    D(x) = w(theta, mask) . x + c(theta, mask)

so the input-gradient g = grad_x D(x) depends on (theta, mask, batch index) only.
The GP value `||g||^2` is then a scalar function of theta alone (with mask
detached), and grad_theta GP can be obtained by a *single* autograd backward
through the manually written transposed network, with no double-backward.

Public entry point:

    exact_gp_value_and_param_grads(discriminator, images, resolution, alpha, stage_mode)
        -> (gp_value: Tensor (scalar, detached), param_grads: dict[str, Tensor|None])
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from project02.models.discriminator import (
    ProgressiveResidualDiscriminator,
    _DiscriminatorDownStage,
    _DiscriminatorTail,
)
from project02.models.layers import (
    BiasLeakyReLU,
    ConvLayer,
    DiscriminativeBasis,
    FixedFilterDownsample,
    R3ResidualBlock,
)


def discriminator_adversarial_loss(real_scores: Tensor, fake_scores: Tensor) -> Tensor:
    """R3GAN relativistic softplus discriminator loss."""

    return F.softplus(-(real_scores - fake_scores)).mean()


def generator_adversarial_loss(real_scores: Tensor, fake_scores: Tensor) -> Tensor:
    """R3GAN relativistic softplus generator loss."""

    return F.softplus(-(fake_scores - real_scores)).mean()


def inject_gp_param_grads(
    discriminator: ProgressiveResidualDiscriminator,
    param_grads: dict[str, Tensor | None],
    scale: float,
) -> None:
    """Accumulate exact GP parameter gradients into param.grad.

    This is called after adversarial backward. The scale already includes the
    gamma/2 factor and any gradient-accumulation correction for the step.
    """

    for name, param in discriminator.named_parameters():
        grad = param_grads.get(name)
        if grad is None:
            continue
        grad = grad.to(dtype=param.dtype, device=param.device)
        if param.grad is None:
            param.grad = grad.mul(scale)
        else:
            param.grad.add_(grad, alpha=scale)


# ---------------------------------------------------------------------------
# Per-primitive helpers
#   "forward" helpers run with no_grad and only exist to capture activation
#   masks for BiasLeakyReLU. The compound trace functions below stitch them
#   together exactly as the forward of ProgressiveResidualDiscriminator does.
#
#   "transposed" helpers compute the input-side cotangent given the
#   output-side cotangent. They are autograd-tracked w.r.t. layer weights so
#   that we can later call autograd.grad(gp_value, params) for a single
#   backward.
# ---------------------------------------------------------------------------


def _conv_layer_forward(layer: ConvLayer, x: Tensor) -> Tensor:
    weight = layer.conv.weight.to(dtype=x.dtype)
    return F.conv2d(
        x,
        weight,
        bias=None,
        stride=layer.conv.stride,
        padding=layer.conv.padding,
        dilation=layer.conv.dilation,
        groups=layer.conv.groups,
    )


def _conv_layer_transposed(layer: ConvLayer, g: Tensor) -> Tensor:
    """Transpose of ConvLayer. stride is always (1,1) and padding is (k-1)//2,
    so output_padding=0 yields the same spatial size as the original input.
    """
    weight = layer.conv.weight.to(dtype=g.dtype)
    return F.conv_transpose2d(
        g,
        weight,
        bias=None,
        stride=layer.conv.stride,
        padding=layer.conv.padding,
        output_padding=0,
        groups=layer.conv.groups,
        dilation=layer.conv.dilation,
    )


def _bias_leaky_mask(act: BiasLeakyReLU, pre_act: Tensor) -> Tensor:
    """Compute the leaky-relu activation mask at the pre-activation tensor.

    The mask has the same shape as `pre_act`, is in pre_act.dtype, holds 1.0
    where (x + bias) > 0 and `slope` elsewhere, and is detached from autograd
    (we run inside no_grad when calling this).
    """
    bias = act.bias.to(dtype=pre_act.dtype).view(1, -1, 1, 1)
    positive = (pre_act + bias) > 0
    mask = torch.where(
        positive,
        torch.ones_like(pre_act),
        torch.full_like(pre_act, float(act.slope)),
    )
    return mask


def _fixed_filter_down_forward(layer: FixedFilterDownsample, x: Tensor) -> Tensor:
    kernel = layer.kernel.to(dtype=x.dtype).expand(x.size(1), -1, -1, -1)
    return F.conv2d(x, kernel, stride=2, padding=1, groups=x.size(1))


def _fixed_filter_down_transposed(layer: FixedFilterDownsample, g: Tensor) -> Tensor:
    """Transpose of FixedFilterDownsample (stride 2 depthwise conv).

    The downsample input has even spatial size (resolution is a power of two),
    so output_padding=1 recovers the original spatial size.
    """
    kernel = layer.kernel.to(dtype=g.dtype).expand(g.size(1), -1, -1, -1)
    return F.conv_transpose2d(
        g,
        kernel,
        stride=2,
        padding=1,
        output_padding=1,
        groups=g.size(1),
    )


def _discriminative_basis_transposed(basis: DiscriminativeBasis, logit_grad: Tensor) -> Tensor:
    """Transpose the score head.

    Forward: x4 = depthwise_conv(x, kernel=4, padding=0); flat = x4.flatten(1);
    out = linear(flat).squeeze(1)
    Reverse: out_grad -> flat_grad (outer with linear weight) -> reshape to
    (B, C, 1, 1) -> conv_transpose with depthwise weight -> (B, C, 4, 4).
    """
    lin_w = basis.linear.weight.to(dtype=logit_grad.dtype)  # (1, in_ch)
    flat_grad = logit_grad.unsqueeze(1) * lin_w  # (B, in_ch)
    in_ch = flat_grad.size(1)
    x4_grad = flat_grad.view(flat_grad.size(0), in_ch, 1, 1)
    dw_w = basis.depthwise.weight.to(dtype=logit_grad.dtype)
    return F.conv_transpose2d(
        x4_grad,
        dw_w,
        bias=None,
        stride=basis.depthwise.stride,
        padding=basis.depthwise.padding,
        output_padding=0,
        groups=basis.depthwise.groups,
        dilation=basis.depthwise.dilation,
    )


# ---------------------------------------------------------------------------
# Trace functions
#   Each compound module gets a no_grad forward pass that captures only the
#   BiasLeakyReLU masks. ConvLayer/FixedFilter ops are linear so no per-layer
#   state is needed for the reverse beyond the masks.
# ---------------------------------------------------------------------------


def _r3block_trace(block: R3ResidualBlock, x: Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
    """Run R3ResidualBlock forward in no_grad and return (output, (mask1, mask2))."""
    with torch.no_grad():
        h = _conv_layer_forward(block.conv1, x)
        mask1 = _bias_leaky_mask(block.act1, h)
        h_act1 = block.act1(h)
        h2 = _conv_layer_forward(block.conv2, h_act1)
        mask2 = _bias_leaky_mask(block.act2, h2)
        h_act2 = block.act2(h2)
        h3 = _conv_layer_forward(block.conv3, h_act2)
        y = x + h3
    return y, (mask1, mask2)


def _r3block_reverse(
    block: R3ResidualBlock, g_out: Tensor, masks: Tuple[Tensor, Tensor]
) -> Tensor:
    """Transposed forward through a residual block.

    The forward branch is conv1 -> act1 -> conv2 -> act2 -> conv3 -> add. The
    transposed branch reverses each op (conv -> conv_transpose; act -> mask
    multiply), and the residual add becomes "pass-through plus branch
    contribution" on the reverse side.
    """
    mask1, mask2 = masks
    g_a2 = _conv_layer_transposed(block.conv3, g_out)
    g_h2 = g_a2 * mask2
    g_a1 = _conv_layer_transposed(block.conv2, g_h2)
    g_h1 = g_a1 * mask1
    g_branch = _conv_layer_transposed(block.conv1, g_h1)
    return g_out + g_branch


def _downstage_trace(
    stage: _DiscriminatorDownStage, x: Tensor
) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
    masks: List[Tuple[Tensor, Tensor]] = []
    h = x
    for block in stage.blocks:
        h, block_masks = _r3block_trace(block, h)
        masks.append(block_masks)
    with torch.no_grad():
        h = _fixed_filter_down_forward(stage.downsample, h)
        if isinstance(stage.proj, ConvLayer):
            h = _conv_layer_forward(stage.proj, h)
        # Identity case: leave h as-is.
    return h, masks


def _downstage_reverse(
    stage: _DiscriminatorDownStage, g_out: Tensor, block_masks: List[Tuple[Tensor, Tensor]]
) -> Tensor:
    g = g_out
    if isinstance(stage.proj, ConvLayer):
        g = _conv_layer_transposed(stage.proj, g)
    g = _fixed_filter_down_transposed(stage.downsample, g)
    for block, masks in zip(reversed(stage.blocks), reversed(block_masks)):
        g = _r3block_reverse(block, g, masks)
    return g


def _tail_trace(
    tail: _DiscriminatorTail, x: Tensor
) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
    masks: List[Tuple[Tensor, Tensor]] = []
    h = x
    for block in tail.blocks:
        h, block_masks = _r3block_trace(block, h)
        masks.append(block_masks)
    with torch.no_grad():
        logits = tail.score(h)
    return logits, masks


def _tail_reverse_from_logit(
    tail: _DiscriminatorTail, logit_grad: Tensor, block_masks: List[Tuple[Tensor, Tensor]]
) -> Tensor:
    g = _discriminative_basis_transposed(tail.score, logit_grad)
    for block, masks in zip(reversed(tail.blocks), reversed(block_masks)):
        g = _r3block_reverse(block, g, masks)
    return g


# ---------------------------------------------------------------------------
# Whole-discriminator trace and transposed forward
# ---------------------------------------------------------------------------


def _check_stage(discriminator: ProgressiveResidualDiscriminator, resolution: int, stage_mode: str) -> None:
    if stage_mode not in {"stabilize", "transition"}:
        raise ValueError(
            f"etmann backend stage_mode must be 'stabilize' or 'transition', got {stage_mode!r}."
        )
    if int(resolution) not in tuple(getattr(discriminator, "supported_resolutions", ())):
        raise ValueError(
            "etmann backend resolution is not active for this discriminator: "
            f"resolution={resolution}, supported={getattr(discriminator, 'supported_resolutions', ())}"
        )
    if stage_mode == "transition" and resolution == 64:
        raise ValueError("64 stage cannot run in transition mode.")


def _trace_discriminator(
    discriminator: ProgressiveResidualDiscriminator,
    x: Tensor,
    resolution: int,
    alpha: float,
    stage_mode: str,
) -> Tuple[Tensor, dict]:
    """Run a no_grad forward that mirrors ProgressiveResidualDiscriminator.forward,
    capturing all activation masks and remembering the per-step structure.
    """
    _check_stage(discriminator, resolution, stage_mode)
    info = {
        "resolution": int(resolution),
        "stage_mode": stage_mode,
        "alpha": float(alpha),
        "down_chain_masks": [],  # list of (cur_resolution, downstage_masks)
        "high_downstage_masks": None,  # only for transition
        "tail_masks": None,
        "score_resolution": None,
    }

    if stage_mode == "transition":
        prev_resolution = resolution // 2
        with torch.no_grad():
            low_x = _fixed_filter_down_forward(discriminator.downsample_rgb, x)
            low_features = _conv_layer_forward(discriminator.from_rgb[str(prev_resolution)], low_x)
            high_features = _conv_layer_forward(discriminator.from_rgb[str(resolution)], x)
        high_features, info["high_downstage_masks"] = _downstage_trace(
            discriminator.down_stages[str(resolution)], high_features
        )
        with torch.no_grad():
            alpha_t = torch.as_tensor(
                float(alpha), dtype=high_features.dtype, device=high_features.device
            )
            features = torch.lerp(low_features, high_features, alpha_t)
        score_resolution = prev_resolution
    else:
        with torch.no_grad():
            features = _conv_layer_forward(discriminator.from_rgb[str(resolution)], x)
        score_resolution = resolution
    info["score_resolution"] = int(score_resolution)

    # Walk down to 4x4 through the remaining down stages.
    cur = score_resolution
    h = features
    while cur > 4:
        h, masks = _downstage_trace(discriminator.down_stages[str(cur)], h)
        info["down_chain_masks"].append((cur, masks))
        cur //= 2

    logits, tail_masks = _tail_trace(discriminator.tail, h)
    info["tail_masks"] = tail_masks
    return logits, info


def _reverse_to_input(
    discriminator: ProgressiveResidualDiscriminator,
    logit_grad: Tensor,
    info: dict,
) -> Tensor:
    """Transposed-network forward returning the input cotangent g = grad_x D(x).

    Tracked by autograd w.r.t. discriminator parameters; not tracked w.r.t. x
    (we never use x here, only masks and weights).
    """
    g = _tail_reverse_from_logit(discriminator.tail, logit_grad, info["tail_masks"])
    for cur, masks in reversed(info["down_chain_masks"]):
        g = _downstage_reverse(discriminator.down_stages[str(cur)], g, masks)

    if info["stage_mode"] == "transition":
        alpha = info["alpha"]
        resolution = info["resolution"]
        prev_resolution = info["score_resolution"]
        # lerp(low, high, alpha) reverse: g_low = (1-alpha) * g; g_high = alpha * g
        g_low = (1.0 - alpha) * g
        g_high = alpha * g
        g_low = _conv_layer_transposed(
            discriminator.from_rgb[str(prev_resolution)], g_low
        )
        g_low = _fixed_filter_down_transposed(discriminator.downsample_rgb, g_low)
        g_high = _downstage_reverse(
            discriminator.down_stages[str(resolution)],
            g_high,
            info["high_downstage_masks"],
        )
        g_high = _conv_layer_transposed(
            discriminator.from_rgb[str(resolution)], g_high
        )
        return g_low + g_high

    g = _conv_layer_transposed(discriminator.from_rgb[str(info["resolution"])], g)
    return g


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_input_gradient(
    discriminator: ProgressiveResidualDiscriminator,
    x: Tensor,
    resolution: int,
    alpha: float,
    stage_mode: str,
) -> Tuple[Tensor, Tensor]:
    """Compute g = grad_x D(x) without autograd double-backward.

    Returns (logits_detached, g) where logits is the no_grad forward output
    and g has shape matching x. g is autograd-tracked w.r.t. discriminator
    parameters (so it can be used to build a scalar that backpropagates into
    parameters with a single backward), but not w.r.t. x.
    """
    logits, info = _trace_discriminator(discriminator, x, resolution, alpha, stage_mode)
    cotangent = torch.ones_like(logits)
    g = _reverse_to_input(discriminator, cotangent, info)
    return logits, g


def exact_gp_value(
    discriminator: ProgressiveResidualDiscriminator,
    x: Tensor,
    resolution: int,
    alpha: float,
    stage_mode: str,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute (logits, g, gp_value).

    gp_value = mean over the batch of `||grad_x D(x)||^2`, i.e. the standard
    zero-centered gradient penalty quantity. It is a scalar tensor that depends
    on discriminator parameters via the transposed-network forward; calling
    .backward() or autograd.grad on it accumulates the GP parameter gradient in
    a single backward pass.
    """
    logits, g = compute_input_gradient(discriminator, x, resolution, alpha, stage_mode)
    gp_value = g.pow(2).flatten(1).sum(1).mean()
    return logits, g, gp_value


def exact_gp_value_and_param_grads(
    discriminator: ProgressiveResidualDiscriminator,
    x: Tensor,
    resolution: int,
    alpha: float,
    stage_mode: str,
) -> Tuple[Tensor, "dict[str, Optional[Tensor]]"]:
    """Compute (gp_value_detached, param_grads).

    `param_grads` is a dict from parameter name to its GP-gradient tensor
    (same shape/dtype as the parameter), or None if the parameter does not
    appear in the transposed network. Bias parameters of BiasLeakyReLU do not
    appear in the transposed network (mask is detached), so their entries are
    None.

    Caller is responsible for any scale (e.g. gamma * 0.5) and for accumulating
    into param.grad.
    """
    _, _, gp_value = exact_gp_value(discriminator, x, resolution, alpha, stage_mode)
    names = [name for name, _ in discriminator.named_parameters()]
    params = [param for _, param in discriminator.named_parameters()]
    grads = torch.autograd.grad(
        gp_value,
        params,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )
    return gp_value.detach(), {name: grad for name, grad in zip(names, grads)}
