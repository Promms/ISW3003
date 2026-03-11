"""Open Source AI Practice"""
import torch
from torch import Tensor
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Task 1 : Pairwise Add
# ---------------------------------------------------------------------------
def pairwise_add(a: Tensor, b: Tensor) -> Tensor:
    """Pairwise element-wise addition across the D dimension.

    Given two tensors of shape (B, D), produce a tensor of shape (B, D, D)
    where output[b, i, j] = a[b, i] + b[b, j].

    Args:
        a: Tensor of shape (B, D)
        b: Tensor of shape (B, D)

    Returns:
        Tensor of shape (B, D, D)

    Example:
        >>> a = torch.tensor([[1., 2.], [3., 4.]])  # (2, 2)
        >>> b = torch.tensor([[10., 20.], [30., 40.]])  # (2, 2)
        >>> pairwise_add(a, b).shape
        torch.Size([2, 2, 2])
        >>> pairwise_add(a, b)[0]
        tensor([[11., 21.],
                [12., 22.]])
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Task 2 : Pairwise Dot-Product
# ---------------------------------------------------------------------------
def pairwise_dot(x: Tensor) -> Tensor:
    """Compute all pairwise dot-products between rows of x.

    Given a tensor of shape (B, D), produce a tensor of shape (B, B)
    where output[i, j] = dot(x[i], x[j]).

    Args:
        x: Tensor of shape (B, D)

    Returns:
        Tensor of shape (B, B)

    Example:
        >>> x = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])  # (3, 2)
        >>> pairwise_dot(x)
        tensor([[1., 0., 1.],
                [0., 1., 1.],
                [1., 1., 2.]])
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Task 3 : Channel-Wise Affine Transform
# ---------------------------------------------------------------------------
def channel_affine(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    """Apply a per-channel affine transform to a 4-D feature map.

    output[b, c, h, w] = weight[c] * x[b, c, h, w] + bias[c]

    Args:
        x:      Tensor of shape (B, C, H, W)
        weight: Tensor of shape (C,)
        bias:   Tensor of shape (C,)

    Returns:
        Tensor of shape (B, C, H, W)

    Example:
        >>> x = torch.ones(2, 3, 4, 4)
        >>> w = torch.tensor([1., 2., 3.])
        >>> b = torch.tensor([0., 1., -1.])
        >>> channel_affine(x, w, b)[0, :, 0, 0]
        tensor([ 1.,  3.,  2.])
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Task 4 : Patchify / Unpatchify  (Swin-style)
# ---------------------------------------------------------------------------
def patchify(x: Tensor, patch_size: int) -> Tensor:
    """Partition an image into non-overlapping patches (as in Swin Transformer).

    Args:
        x:          Tensor of shape (B, H, W, C)
        patch_size: Integer P; assume H and W are divisible by P.

    Returns:
        Tensor of shape (B, num_patches_h, num_patches_w, P, P, C)
        where num_patches_h = H // P, num_patches_w = W // P.

    Example:
        >>> x = torch.arange(1*8*8*1, dtype=torch.float).reshape(1, 8, 8, 1)
        >>> patchify(x, 4).shape
        torch.Size([1, 2, 2, 4, 4, 1])
    """
    # do not use loop.
    raise NotImplementedError


def unpatchify(patches: Tensor) -> Tensor:
    """Reconstruct an image from its patches (inverse of patchify).

    Args:
        patches: Tensor of shape (B, num_patches_h, num_patches_w, P, P, C)

    Returns:
        Tensor of shape (B, H, W, C)
        where H = num_patches_h * P, W = num_patches_w * P.

    Example:
        >>> x = torch.arange(1*8*8*1, dtype=torch.float).reshape(1, 8, 8, 1)
        >>> patches = patchify(x, 4)
        >>> torch.allclose(unpatchify(patches), x)
        True
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Task 5 : L2 Normalization
# ---------------------------------------------------------------------------
def l2_normalize(x: Tensor, eps: float = 1e-8) -> Tensor:
    """Normalise each row of x to unit L2 norm.

    Args:
        x:   Tensor of shape (B, D)
        eps: Small constant for numerical stability.

    Returns:
        Tensor of shape (B, D), where each row has L2 norm ≈ 1.

    Example:
        >>> x = torch.tensor([[3., 4.], [0., 0.]])
        >>> l2_normalize(x)[0]
        tensor([0.6000, 0.8000])
    """
    # do not use torch.nn.functional.normalize
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Task 6 : Channel-Wise Normalization
# ---------------------------------------------------------------------------
def channel_normalize(x: Tensor, eps: float = 1e-8) -> Tensor:
    """Normalise each channel across (H, W) to zero mean and unit variance.

    For each (b, c) pair, subtract the mean and divide by the std computed
    over the spatial dimensions H and W.

    Args:
        x:   Tensor of shape (B, C, H, W)
        eps: Small constant for numerical stability.

    Returns:
        Tensor of shape (B, C, H, W)
    """
    # do not use torch.nn.functional.normalize
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Task 7 : In-Place Moving Average Update
# ---------------------------------------------------------------------------
def moving_average_update_(running: Tensor, new_val: Tensor, momentum: float) -> None:
    """Update a running statistic in-place using an exponential moving average.

    running = (1 - momentum) * running + momentum * new_val

    The function must modify `running` in-place and return None.

    Args:
        running:  Tensor of any shape — the running statistic to update.
        new_val:  Tensor of the same shape as `running`.
        momentum: Scalar in (0, 1).

    Example:
        >>> r = torch.zeros(3)
        >>> moving_average_update_(r, torch.ones(3), 0.1)
        >>> r
        tensor([0.1000, 0.1000, 0.1000])
    """
    # use torch inplace ops
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Task 8 : Masked Average
# ---------------------------------------------------------------------------
def masked_average(x: Tensor, mask: Tensor) -> Tensor:
    """Compute the mean over valid (unmasked) time steps for each sequence.

    Args:
        x:    Tensor of shape (B, T, D)
        mask: Boolean (or 0/1 float) Tensor of shape (B, T)
              — True / 1 means the position is valid, False / 0 means padding.

    Returns:
        Tensor of shape (B, D) — the average of valid positions per sample.

    Example:
        >>> x = torch.tensor([[[1., 1.], [2., 2.], [3., 3.]]])  # (1, 3, 2)
        >>> mask = torch.tensor([[True, True, False]])
        >>> masked_average(x, mask)
        tensor([[1.5000, 1.5000]])
    """
    # do not use loop
    # there exist many edge cases to handle
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Task 9 : Entropy Computation
# ---------------------------------------------------------------------------
def entropy(probs: Tensor, eps: float = 1e-8) -> Tensor:
    """Compute Shannon entropy H = -sum(p * log(p)) along the vocabulary axis.

    Args:
        probs: Tensor of shape (B, T, V) — values are probabilities (≥ 0,
               summing to 1 along the last dimension).
        eps:   Small constant added inside log to avoid log(0).

    Returns:
        Tensor of shape (B, T) containing the entropy for each position.
    """
    # do not use loop
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Task 10 : Top-K Extraction
# ---------------------------------------------------------------------------
def topk_extract(logits: Tensor, k: int) -> Tuple[Tensor, Tensor]:
    """Extract the top-k values and their indices along the vocabulary axis.

    Args:
        logits: Tensor of shape (B, T, V)
        k:      Number of top elements to extract.

    Returns:
        values:  Tensor of shape (B, T, k) — top-k values (sorted descending).
        indices: Tensor of shape (B, T, k) — corresponding original indices.

    Example:
        >>> logits = torch.tensor([[[3., 1., 4., 1., 5., 9., 2., 6.]]])
        >>> vals, idx = topk_extract(logits, 3)
        >>> vals
        tensor([[[9., 6., 5.]]])
        >>> idx
        tensor([[[5, 7, 4]]])
    """
        # do not use torch.topk
    # use torch.sort or torch.argsort.
    # do not use loop
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Task 11 : Relative Position Indices
# ---------------------------------------------------------------------------
def relative_position_indices(T: int) -> Tensor:
    """Compute a matrix of relative position indices for a 1-D sequence.

    output[i, j] = j - i, shifted so that the minimum value is 0.

    The resulting values lie in [0, 2*(T-1)].

    Args:
        T: Sequence length.

    Returns:
        LongTensor of shape (T, T).

    Example:
        >>> relative_position_indices(3)
        tensor([[2, 3, 4],
                [1, 2, 3],
                [0, 1, 2]])
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Task 12 : Pad and Stack Variable-Length Sequences
# ---------------------------------------------------------------------------
def pad_and_stack(sequences: List[Tensor]) -> Tuple[Tensor, Tensor]:
    """Pad a list of variable-length sequences and stack into a batch.

    Args:
        sequences: List of N tensors, each of shape (T_i, D).
                   The T_i may differ; D must be the same for all.

    Returns:
        padded: Float Tensor of shape (N, max_T, D) — sequences padded with
                zeros on the right to the maximum length.
        mask:   Boolean Tensor of shape (N, max_T) — True where the position
                is a real token, False where it is padding.

    Example:
        >>> seqs = [torch.ones(2, 4), torch.ones(5, 4)]
        >>> padded, mask = pad_and_stack(seqs)
        >>> padded.shape
        torch.Size([2, 5, 4])
        >>> mask
        tensor([[ True,  True, False, False, False],
                [ True,  True,  True,  True,  True]])
    """
    raise NotImplementedError


# ===========================================================================
# Tests
# ===========================================================================
def _test_all():
    print("Running tests...\n")
    passed = 0
    failed = 0

    def check(name, cond):
        nonlocal passed, failed
        if cond:
            print(f"[PASS] {name}")
            passed += 1
        else:
            print(f"[FAIL] {name}")
            failed += 1

    # Task 1
    try:
        a = torch.tensor([[1., 2.], [3., 4.]])
        b = torch.tensor([[10., 20.], [30., 40.]])
        out = pairwise_add(a, b)
        check("Task 1: shape",      out.shape == torch.Size([2, 2, 2]))
        check("Task 1: values",     torch.allclose(out[0], torch.tensor([[11., 21.], [12., 22.]])))
    except NotImplementedError:
        print("[SKIP] Task 1: not implemented")

    # Task 2
    # TODO add a test case

    # Task 3
    try:
        x = torch.ones(2, 3, 4, 4)
        w = torch.tensor([1., 2., 3.])
        b = torch.tensor([0., 1., -1.])
        out = channel_affine(x, w, b)
        check("Task 3: shape",  out.shape == torch.Size([2, 3, 4, 4]))
        check("Task 3: values", torch.allclose(out[0, :, 0, 0], torch.tensor([1., 3., 2.])))
    except NotImplementedError:
        print("[SKIP] Task 3: not implemented")

    # Task 4
    # TODO add a test case

    # Task 5
    try:
        x = torch.tensor([[3., 4.], [1., 0.]])
        out = l2_normalize(x)
        check("Task 5: direction", torch.allclose(out[0], torch.tensor([0.6, 0.8])))
    except NotImplementedError:
        print("[SKIP] Task 5: not implemented")

    # Task 6
    # TODO add a test case

    # Task 7
    try:
        r = torch.zeros(3)
        moving_average_update_(r, torch.ones(3), 0.1)
        check("Task 7: in-place EMA", torch.allclose(r, torch.full((3,), 0.1)))
    except NotImplementedError:
        print("[SKIP] Task 7: not implemented")

    # Task 8
    # TODO add a test case

    # Task 9
    try:
        p = torch.full((1, 1, 4), 0.25)
        out = entropy(p)
        check("Task 9: shape",  out.shape == torch.Size([1, 1]))
        check("Task 9: value",  torch.isclose(out, torch.tensor([[torch.log(torch.tensor(4.))]])).all())
    except NotImplementedError:
        print("[SKIP] Task 9: not implemented")

    # Task 10
    # TODO add a test case

    # Task 11
    try:
        out = relative_position_indices(3)
        ref = torch.tensor([[2, 3, 4], [1, 2, 3], [0, 1, 2]])
        check("Task 11: shape",  out.shape == torch.Size([3, 3]))
        check("Task 11: values", torch.equal(out, ref))
    except NotImplementedError:
        print("[SKIP] Task 11: not implemented")

    # Task 12
    # TODO add a test case

    # Summary
    print(f"\nResults: {passed} passed, {failed} failed.")


if __name__ == "__main__":
    _test_all()