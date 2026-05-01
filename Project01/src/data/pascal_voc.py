from __future__ import annotations

import random

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.datasets import VOCSegmentation

from data.transforms import (
    ERASING_RATIO,
    ERASING_SCALE,
    IGNORE_INDEX,
    IMAGENET_MEAN,
    IMAGENET_STD,
    augment_src_for_copy_paste,
    copy_paste,
    train_augment,
    val_transform,
)

VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "potted plant", "sheep", "sofa", "train", "tv/monitor",
]

VOC_COLORMAP = [
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
    [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
    [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
    [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
    [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128],
]

NUM_CLASSES = 21


class VOCSegDataset(VOCSegmentation):
    """VOC segmentation dataset with augmentation."""

    def __init__(
        self,
        root,
        year="2012",
        image_set="train",
        crop_size=320,
        augment=False,
        download=False,
        copy_paste_prob=0.1,
        random_erasing_prob=0.25,
    ):
        super().__init__(root=root, year=year, image_set=image_set, download=download)
        self.crop_size = crop_size
        self.augment = augment
        self.normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        self.copy_paste_prob = copy_paste_prob if augment else 0.0
        self.random_erasing_prob = random_erasing_prob if augment else 0.0

    def _load(self, idx):
        image = cv2.imread(self.images[idx], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = np.array(Image.open(self.masks[idx]))
        return image, mask

    def __getitem__(self, idx):
        image, mask = self._load(idx)

        if self.augment:
            image, mask = train_augment(image, mask, self.crop_size)

            if random.random() < self.copy_paste_prob:
                src_idx = random.randrange(len(self))
                if src_idx == idx:
                    src_idx = (src_idx + 1) % len(self)
                image_src, mask_src = self._load(src_idx)
                image_src, mask_src = augment_src_for_copy_paste(
                    image_src, mask_src, self.crop_size
                )
                image, mask = copy_paste(image, mask, image_src, mask_src)
        else:
            image, mask = val_transform(image, mask)

        image = torch.from_numpy(image.transpose(2, 0, 1).copy()).float().div_(255.0)
        image = self.normalize(image)
        mask = torch.from_numpy(mask).long()

        if self.augment and random.random() < self.random_erasing_prob:
            i, j, h, w, v = T.RandomErasing.get_params(
                image, scale=ERASING_SCALE, ratio=ERASING_RATIO, value=[0.0]
            )
            image[:, i:i + h, j:j + w] = v
            mask[i:i + h, j:j + w] = IGNORE_INDEX

        return image, mask


def build_voc_datasets(
    root,
    years=["2007", "2012"],
    image_set="train",
    crop_size=320,
    augment=None,
    download=False,
) -> list[VOCSegDataset]:
    if augment is None:
        augment = (image_set == "train")
    return [
        VOCSegDataset(
            root=root,
            year=year,
            image_set=image_set,
            crop_size=crop_size,
            augment=augment,
            download=download,
        )
        for year in years
    ]


def get_loader(
    root,
    years=["2007", "2012"],
    image_set="train",
    crop_size=320,
    batch_size=8,
    num_workers=2,
    pin_memory=True,
    persistent_workers=False,
    prefetch_factor=2,
    download=False,
) -> DataLoader:
    is_train = (image_set == "train")
    datasets = build_voc_datasets(
        root=root,
        years=years,
        image_set=image_set,
        crop_size=crop_size,
        augment=is_train,
        download=download,
    )
    combined = ConcatDataset(datasets)

    return DataLoader(
        combined,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=is_train,
        persistent_workers=(persistent_workers and num_workers > 0),
        prefetch_factor=(prefetch_factor if num_workers > 0 else None),
    )
