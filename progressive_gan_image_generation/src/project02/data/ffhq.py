from __future__ import annotations

from pathlib import Path

from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class FFHQDataset(Dataset[Tensor]):
    def __init__(
        self,
        root: str | Path,
        image_size: int,
        horizontal_flip: bool = False,
        cache_mode: str | None = None,
        return_uint8: bool = False,
    ) -> None:
        self.root = Path(root)
        self.image_size = image_size
        self.horizontal_flip = horizontal_flip
        self.cache_mode = cache_mode or "none"
        self.return_uint8 = return_uint8
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")

        self.paths = sorted(
            path
            for path in self.root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise FileNotFoundError(f"No image files found under: {self.root}")

        if self.cache_mode not in {"none", "decoded_uint8"}:
            raise ValueError(f"Unknown FFHQ cache mode: {self.cache_mode}")

        self.cached_images: list[Tensor] | None = None
        if self.cache_mode == "decoded_uint8":
            self.cached_images = [self._load_uint8(path) for path in self.paths]
            total_gib = sum(image.numel() for image in self.cached_images) / (1024**3)
            print(f"[data] Cached {len(self.cached_images):,} decoded uint8 images ({total_gib:.2f} GiB)")

        transform_steps: list[object] = [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
        ]
        if horizontal_flip:
            transform_steps.append(transforms.RandomHorizontalFlip(p=0.5))
        transform_steps.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )
        self.transform = transforms.Compose(transform_steps)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tensor:
        if self.cached_images is not None:
            image = self.cached_images[index]
            if self.horizontal_flip and bool(torch.rand(()) < 0.5):
                image = torch.flip(image, dims=[2])
            if self.return_uint8:
                return image
            return image.float().div(127.5).sub(1.0)

        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))

    def _load_uint8(self, path: Path) -> Tensor:
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = TF.resize(image, self.image_size, interpolation=transforms.InterpolationMode.BICUBIC)
            image = TF.center_crop(image, self.image_size)
            data = torch.frombuffer(image.tobytes(), dtype=torch.uint8).clone()
        return data.view(self.image_size, self.image_size, 3).permute(2, 0, 1).contiguous()
