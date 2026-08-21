from __future__ import annotations

import torch


def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Gradient of the Lovasz extension with respect to sorted errors."""
    num_pixels = len(gt_sorted)
    gt_sum = gt_sorted.sum()
    intersection = gt_sum - gt_sorted.float().cumsum(0)
    union = gt_sum + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if num_pixels > 1:
        jaccard[1:num_pixels] = jaccard[1:num_pixels] - jaccard[:num_pixels - 1]
    return jaccard


def lovasz_softmax(
    probas: torch.Tensor,
    labels: torch.Tensor,
    classes: str | list[int] = "present",
    per_image: bool = False,
    ignore: int | None = None,
) -> torch.Tensor:
    """Multi-class Lovasz-Softmax loss from Berman et al. (2018)."""
    if per_image:
        losses = (
            lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
            for prob, lab in zip(probas, labels)
        )
        return mean(losses)
    return lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)


def lovasz_softmax_flat(
    probas: torch.Tensor,
    labels: torch.Tensor,
    classes: str | list[int] = "present",
) -> torch.Tensor:
    if probas.numel() == 0:
        return probas.sum() * 0.0

    num_classes = probas.size(1)
    class_to_sum = list(range(num_classes)) if classes in ("all", "present") else classes
    losses = []

    for class_idx in class_to_sum:
        foreground = (labels == class_idx).float()
        if classes == "present" and foreground.sum() == 0:
            continue

        class_pred = probas[:, 0] if num_classes == 1 else probas[:, class_idx]
        errors = (foreground - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        foreground_sorted = foreground[perm]
        losses.append(torch.dot(errors_sorted, lovasz_grad(foreground_sorted)))

    return mean(losses)


def flatten_probas(
    probas: torch.Tensor,
    labels: torch.Tensor,
    ignore: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if probas.dim() == 3:
        batch, height, width = probas.size()
        probas = probas.view(batch, 1, height, width)

    batch, channels, height, width = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(batch * height * width, channels)
    labels = labels.view(-1)

    if ignore is None:
        return probas, labels

    valid = labels != ignore
    return probas[valid], labels[valid]


def mean(values, empty=0):
    values = iter(values)
    try:
        count = 1
        acc = next(values)
    except StopIteration:
        return empty

    for count, value in enumerate(values, 2):
        acc += value
    return acc if count == 1 else acc / count
