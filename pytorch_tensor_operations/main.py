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
    a = a.unsqueeze(2)  # (B, D) -> (B, D, 1) 차원 확장 
    b = b.unsqueeze(1)  # (B, D) -> (B, 1, D) 차원 확장
    # 크기가 1인 차원을 생성하여 브로드캐스팅 기능을 활용하여 차원을 (B, D, D)로 맞춰 pairwise
    c = a + b           # (B, D, D)

    return c


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

                [1 0] [1 0 1]
                [0 1] [0 1 1]
                [1 1]
    """

    pairwise_dot = x @ x.transpose(0, 1)

    return pairwise_dot


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
    # 4차원 텐서와 1차원 weight, bias를 계산하기 위해 view를 활용해 차원을 맞춘다.
    weight = weight.view(1, -1, 1, 1)   # (C,) -> (1, C, 1, 1)
    bias = bias.view(1, -1, 1, 1)       # (C,) -> (1, C, 1, 1)

    y = x * weight + bias
    return y


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
    num_patches_h = x.size(1) // patch_size
    num_patches_w = x.size(2) // patch_size
    # element들의 순서를 고려하면서 자르기 위해 자른 뒤 permute를 활용하여 순서를 변경한다.
    x2 = x.view(x.size(0), num_patches_h, patch_size, num_patches_w, patch_size, x.size(3))
    x2 = x2.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x2


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
    original_h = patches.size(1) * patches.size(3)
    original_w = patches.size(2) * patches.size(4)

    # patch할 때 순서를 변경했기 때문에 원래 순서로 변경한 뒤 합쳐야 한다
    patches = patches.permute(0, 1, 3, 2, 4, 5).contiguous()
    x2 = patches.reshape(patches.size(0), original_h, original_w, patches.size(5))
    return x2


# ---------------------------------------------------------------------------
# Task 5 : L2 Normalization
# ---------------------------------------------------------------------------
def l2_normalize(x: Tensor, eps: float = 1e-8) -> Tensor:
    """Normalise each row of x to unit L2 norm.

    L2 norm을 구해서 길이가 1이 되도록 나눔

    Args:
        x:   Tensor of shape (B, D) -> input
        eps: Small constant for numerical stability.

    Returns:
        Tensor of shape (B, D), where each row has L2 norm ≈ 1.

    Example:
        >>> x = torch.tensor([[3., 4.], [0., 0.]])
        >>> l2_normalize(x)[0]
        tensor([0.6000, 0.8000])
    """
    # do not use torch.nn.functional.normalize

    # L2 norm = sqrt(sum(x_i^2))
    norm = torch.sum(x*x, dim=1, keepdim=True)      # 1번 축으로 더해야함 (B, 1)
    norm = torch.sqrt(norm + eps)

    x = x / norm                                    # (B, D) / (B, 1) = (B, D)
    return x


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

    x_mean = x.mean(dim=(2,3), keepdim=True)
    x_std = x.std(dim=(2,3), keepdim=True)

    x_norm = x.subtract(x_mean).divide(x_std + eps)
    return x_norm


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
    # in-place로 계산하기 위해 _를 붙여서 계산한다.
    running.mul_(1 - momentum).add_(momentum * new_val)
    # running  = (1 - momentum)

    return running


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

    mask = mask.unsqueeze(-1).float()           # (B, T) -> (B, T, 1)
    x = x * mask                                # mask가 (B, T, D)로 브로드캐스팅 되어 계산된다.
    num_true = mask.sum(dim=1, keepdim=True)    # shape (B, 1, 1)

    x_sum = x.sum(dim=1)                        # sum over T: (B, D)
    eps = float(1e-8)                           # 분모가 0이 되는 상황을 방지하기 위해 위에서 쓴 방식 차용 
    return x_sum / (num_true.squeeze(-1) + eps) # (B, D) / (B, 1) = (B, D)


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
    entropy = -probs * torch.log(probs + eps)
    entropy = torch.sum(entropy, dim=2)         #(B, T, V) -> (B, T)
    return entropy

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

    sorted_values, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    # 슬라이싱을 통해 (B, T, k) 크기로 잘라냄
    return sorted_values[:,:,:k], sorted_indices[:,:,:k]


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
    indices = torch.arange(0, T, step=1)  # (T, )
    distance = indices.unsqueeze(0) - indices.unsqueeze(1)  # (1, T) - (T, 1) = (T, T)
    distance += (T-1)
    return distance

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
    # max_T 값을 찾는다
    max_T = max(seq.size(0) for seq in sequences)
    
    # mask는 (시퀀스의 크기, max_T)로 생성함
    N = len(sequences)
    D = sequences[0].size(1)
    mask = torch.zeros((N, max_T), dtype=torch.bool)
    
    # 시퀀스를 루프로 돌면서 패드를 수행
    padded_list = []
    for i, seq in enumerate(sequences):
        T = seq.size(0)
        mask[i, :T] = True
        
        # torch.zeros(max_T - T, D)으로 텐서를 만들고 합침 -> padded
        if T < max_T:
            pad_tensor = torch.zeros((max_T - T, D), dtype=seq.dtype)
            padded_seq = torch.cat([seq, pad_tensor], dim=0)
        else:
            padded_seq = seq
        
        padded_list.append(padded_seq)
    
    # 그렇게 padded된 모든 텐서를 합침
    padded = torch.stack(padded_list, dim=0)
    
    return padded, mask


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
    try:
        a = torch.tensor([[1., 2., 3.], 
                          [4., 5., 6.]])
        out = pairwise_dot(a)
        check("Task 2: shape", out.shape == torch.Size([2, 2]))
        expected = torch.tensor([[14., 32.], [32., 77.]])
        check("Task 2: values", torch.allclose(out, expected))
    except NotImplementedError:
        print("[SKIP] Task 2: not implemented")

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
    try:
        x = torch.arange(1*8*8*1, dtype=torch.float).reshape(1, 8, 8, 1)
        p = 4
        out = patchify(x, p)
        check("Task 4: patched shape",  out.shape == torch.Size([1, 2, 2, 4, 4, 1]))
        out2 = unpatchify(out)
        check("Task 4: unpatched shape",  out2.shape == torch.Size([1, 8, 8, 1]))
    except NotImplementedError:
        print("[SKIP] Task 4: not implemented")

    # Task 5
    try:
        x = torch.tensor([[3., 4.], [1., 0.]])
        out = l2_normalize(x)
        check("Task 5: direction", torch.allclose(out[0], torch.tensor([0.6, 0.8])))
    except NotImplementedError:
        print("[SKIP] Task 5: not implemented")

    # Task 6
    try:
        x = torch.randn(1, 2, 4, 4)
        out = channel_normalize(x)
        check("Task 6: shape", out.shape == torch.Size([1, 2, 4, 4]))
        mean = out.mean(dim=(2, 3))
        std = out.std(dim=(2, 3))
        check("Task 6: mean is 0", torch.allclose(mean, torch.zeros_like(mean), atol=1e-6))
        check("Task 6: std is 1", torch.allclose(std, torch.ones_like(std), atol=1e-6))
    except NotImplementedError:
        print("[SKIP] Task 6: not implemented")

    # Task 7
    try:
        r = torch.zeros(3)
        moving_average_update_(r, torch.ones(3), 0.1)
        check("Task 7: in-place EMA", torch.allclose(r, torch.full((3,), 0.1)))
    except NotImplementedError:
        print("[SKIP] Task 7: not implemented")

    # Task 8
    try:
        x = torch.tensor([[[1., 1.], [2., 2.], [3., 3.]]], dtype=torch.float)
        mask = torch.tensor([[True, True, False]], dtype=torch.bool)
        out = masked_average(x, mask)
        check("Task 8: shape", out.shape == torch.Size([1, 2]))
        expected = torch.tensor([[1.5000, 1.5000]])
        check("Task 8: values", torch.allclose(out, expected))
        
    except NotImplementedError:
        print("[SKIP] Task 8: not implemented")

    # Task 9
    try:
        p = torch.full((1, 1, 4), 0.25)
        out = entropy(p)
        check("Task 9: shape",  out.shape == torch.Size([1, 1]))
        check("Task 9: value",  torch.isclose(out, torch.tensor([[torch.log(torch.tensor(4.))]])).all())
    except NotImplementedError:
        print("[SKIP] Task 9: not implemented")

    # Task 10
    try:
        logits = torch.tensor([[[3., 1., 4., 1., 5., 9., 2., 6.]]])
        k = 3
        vals, idx = topk_extract(logits, k)
        check("Task 10: values shape", vals.shape == torch.Size([1, 1, 3]))
        check("Task 10: indices shape", idx.shape == torch.Size([1, 1, 3]))
        check("Task 10: values", torch.allclose(vals, torch.tensor([[[9., 6., 5.]]])))
        check("Task 10: indices", torch.equal(idx, torch.tensor([[[5, 7, 4]]])))
    except NotImplementedError:
        print("[SKIP] Task 10: not implemented")

    # Task 11
    try:
        out = relative_position_indices(3)
        ref = torch.tensor([[2, 3, 4], [1, 2, 3], [0, 1, 2]])
        check("Task 11: shape",  out.shape == torch.Size([3, 3]))
        check("Task 11: values", torch.equal(out, ref))
    except NotImplementedError:
        print("[SKIP] Task 11: not implemented")

    # Task 12
    try:
        seqs = [torch.ones(2, 4), torch.ones(5, 4)]
        padded, mask = pad_and_stack(seqs)
        check("Task 12: padded shape", padded.shape == torch.Size([2, 5, 4]))
        check("Task 12: mask shape", mask.shape == torch.Size([2, 5]))
        expected_mask = torch.tensor([[True, True, False, False, False],
                                      [True, True, True, True, True]])
        check("Task 12: mask values", torch.equal(mask, expected_mask))
    except NotImplementedError:
        print("[SKIP] Task 12: not implemented")

    # Summary
    print(f"\nResults: {passed} passed, {failed} failed.")


if __name__ == "__main__":
    _test_all()