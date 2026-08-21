import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10


class CIFAR10ForGAN(Dataset):
    """
    CIFAR-10 train split, returning images normalized to [-1, 1] (matching tanh output).
    Labels are dropped — GAN training is unconditional.
    """

    def __init__(self, root: str, download: bool = True) -> None:
        self.data = CIFAR10(root=root, train=True, download=download)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tensor:
        img, _ = self.data[idx]
        arr = np.array(img)  # HWC uint8
        t = torch.from_numpy(arr).permute(2, 0, 1).float()  # CHW float32 in [0, 255]
        return t.div(127.5).sub(1.0)  # [-1, 1]


def get_cifar10_loader(
    root: str,
    batch_size: int,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> DataLoader:
    dataset = CIFAR10ForGAN(root=root, download=True)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
