from pathlib import Path

import torch
import torchvision
import torchvision.transforms as T  # noqa

CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)

DATA_ROOT = Path(__file__).parent / ".data"


def get_test_loader(batch_size: int = 256, num_workers: int = 2) -> torch.utils.data.DataLoader:
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    dataset = torchvision.datasets.CIFAR100(
        root=str(DATA_ROOT), train=False, download=True, transform=transform
    )
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
