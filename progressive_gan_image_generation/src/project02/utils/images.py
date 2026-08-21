from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from torch import Tensor
from torchvision.utils import save_image


def denormalize(images: Tensor) -> Tensor:
    return images.add(1.0).div(2.0).clamp(0.0, 1.0)


def save_sample_grid(images: Tensor, output_path: str | Path, nrow: int = 4) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_image(denormalize(images.detach().cpu()), output, nrow=nrow)


def save_image_batch(images: Tensor, output_dir: str | Path, start_index: int = 0) -> int:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    batch = denormalize(images.detach().float().cpu())
    for offset, image in enumerate(batch):
        save_image(image, output / f"{int(start_index) + offset:06d}.png")
    return int(batch.size(0))


def export_generated_images(
    generator: torch.nn.Module,
    sample_z_fn: Callable[[int, torch.device], Tensor],
    output_dir: str | Path,
    num_images: int,
    batch_size: int,
    device: torch.device,
    call_generator: Callable[[torch.nn.Module, Tensor], Tensor],
) -> int:
    generator_was_training = generator.training
    generator.eval()
    exported = 0
    try:
        with torch.no_grad():
            while exported < int(num_images):
                current = min(int(batch_size), int(num_images) - exported)
                z = sample_z_fn(current, device)
                images = call_generator(generator, z)
                exported += save_image_batch(images, output_dir, exported)
    finally:
        generator.train(generator_was_training)
    return exported
