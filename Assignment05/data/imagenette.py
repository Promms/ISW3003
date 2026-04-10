import torch
import torchvision.transforms as T

from torch.utils.data import DataLoader
from torchvision.datasets import Imagenette

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
    
def get_imagenette_dataloaders(
    dataset_name: str = "imagenette",
    image_size: int = 320,
    batch_size: int = 128,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader]:

    transform = T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_ds = Imagenette(root="./data", split="train", size=f"{image_size}px", transform=transform, download=True)
    val_ds   = Imagenette(root="./data", split="val",   size=f"{image_size}px", transform=transform, download=True)

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