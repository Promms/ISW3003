import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def pil_to_tensor(img) -> Tensor:
    """Convert a PIL Image (HWC uint8) to a float32 CHW tensor in [0, 1]."""
    arr = np.array(img)               # HWC, uint8
    arr = arr.transpose(2, 0, 1)      # CHW
    return torch.from_numpy(arr).float().div(255.0)


def normalize(tensor: Tensor, mean: tuple, std: tuple) -> Tensor:
    """Normalize a CHW float tensor with given per-channel mean and std."""
    mean_t = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std_t  = torch.tensor(std,  dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean_t) / std_t


class TinyImageNetDataset(Dataset):
    """
    PyTorch Dataset wrapping a HuggingFace Tiny-ImageNet split.

    Transforms are applied in __getitem__:
      PIL Image -> float32 CHW tensor in [0,1] -> ImageNet-normalized
    """

    def __init__(self, hf_split) -> None:
        self.data = hf_split

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        sample = self.data[idx]
        image  = normalize(pil_to_tensor(sample["image"].convert("RGB")), IMAGENET_MEAN, IMAGENET_STD)
        label  = sample["label"]
        return image, label


def get_tiny_imagenet_dataloaders(
    dataset_name: str = "zh-plus/tiny-imagenet",
    batch_size: int = 128,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """
    Load Tiny-ImageNet from HuggingFace, wrap in TinyImageNetDataset,
    and return (train_loader, val_loader).
    """
    raw = load_dataset(dataset_name)

    train_ds = TinyImageNetDataset(raw["train"])
    val_ds   = TinyImageNetDataset(raw["valid"])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader
