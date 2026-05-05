from __future__ import annotations

import random

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.datasets import VOCSegmentation

from data.augmentations import (
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
NUM_CLASSES = 21


class VOCSegDataset(VOCSegmentation):
    """Pascal VOC segmentation dataset with optional RAM preload and augmentation."""

    def __init__(
        self,
        root,
        year="2012",
        image_set="train",
        crop_size=320,
        augment=False,
        download=False,
        copy_paste_prob=0.1,
        random_erasing_prob=0.5,
        preload=True,
        draft_size=640,
    ):
        super().__init__(root=root, year=year, image_set=image_set, download=download)
        self.crop_size = crop_size
        self.augment = augment
        self.normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        self.copy_paste_prob = copy_paste_prob if augment else 0.0
        self.random_erasing_prob = random_erasing_prob if augment else 0.0

        self._cache_images = None
        self._cache_masks = None
        if preload:
            self._cache_images = []
            self._cache_masks = []
            for idx in range(len(self.images)):
                image = Image.open(self.images[idx])
                image.draft("RGB", (draft_size, draft_size))
                self._cache_images.append(np.array(image.convert("RGB")))
                self._cache_masks.append(np.array(Image.open(self.masks[idx])))
            print(f"[VOCSegDataset] {year}/{image_set}: preloaded {len(self._cache_images)} samples")

    def _load(self, idx: int):
        if self._cache_images is not None:
            return self._cache_images[idx], self._cache_masks[idx]

        image = cv2.imread(self.images[idx], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = np.array(Image.open(self.masks[idx]))
        return image, mask

    def __getitem__(self, idx: int):
        image, mask = self._load(idx)

        if self.augment:
            image, mask = train_augment(image, mask, self.crop_size)
            if random.random() < self.copy_paste_prob:
                src_idx = random.randrange(len(self))
                if src_idx == idx:
                    src_idx = (src_idx + 1) % len(self)
                src_image, src_mask = self._load(src_idx)
                src_image, src_mask = augment_src_for_copy_paste(src_image, src_mask, self.crop_size)
                image, mask = copy_paste(image, mask, src_image, src_mask)
        else:
            image, mask = val_transform(image, mask)

        image = torch.from_numpy(image.transpose(2, 0, 1).copy()).float().div_(255.0)
        image = self.normalize(image)
        mask = torch.from_numpy(mask).long()

        if self.augment and random.random() < self.random_erasing_prob:
            i, j, h, w, value = T.RandomErasing.get_params(
                image,
                scale=ERASING_SCALE,
                ratio=ERASING_RATIO,
                value=[0.0],
            )
            image[:, i:i + h, j:j + w] = value
            mask[i:i + h, j:j + w] = IGNORE_INDEX

        return image, mask


def build_voc_datasets(
    root,
    years=["2007", "2012"],
    image_set="train",
    crop_size=320,
    augment=None,
    download=False,
    preload=True,
) -> list[VOCSegDataset]:
    if augment is None:
        augment = image_set == "train"
    return [
        VOCSegDataset(
            root=root,
            year=year,
            image_set=image_set,
            crop_size=crop_size,
            augment=augment,
            download=download,
            preload=preload,
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
    download=False,
    preload=True,
) -> DataLoader:
    is_train = image_set == "train"
    datasets = build_voc_datasets(
        root=root,
        years=years,
        image_set=image_set,
        crop_size=crop_size,
        augment=is_train,
        download=download,
        preload=preload,
    )
    return DataLoader(
        ConcatDataset(datasets),
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=is_train,
        persistent_workers=False,
        prefetch_factor=(2 if num_workers > 0 else None),
    )
