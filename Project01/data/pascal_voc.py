"""
Pascal VOC Segmentation Dataset + DataLoader 빌더.

Augmentation 로직은 data/transforms.py에 분리.
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
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
    [0, 0, 0],     [128, 0, 0],   [0, 128, 0],   [128, 128, 0],
    [0, 0, 128],   [128, 0, 128], [0, 128, 128], [128, 128, 128],
    [64, 0, 0],    [192, 0, 0],   [64, 128, 0],  [192, 128, 0],
    [64, 0, 128],  [192, 0, 128], [64, 128, 128], [192, 128, 128],
    [0, 64, 0],    [128, 64, 0],  [0, 192, 0],   [128, 192, 0], [0, 64, 128],
]

NUM_CLASSES = 21


class VOCSegDataset(VOCSegmentation):
    """VOC Segmentation Dataset + RAM preload + Copy-Paste + RandomErasing B."""

    def __init__(self, root, year="2012", image_set="train", crop_size=320, augment=False,
                 download=False, copy_paste_prob=0.3, random_erasing_prob=0.5,
                 preload=True, draft_size=640):
        super().__init__(root=root, year=year, image_set=image_set, download=download)
        self.crop_size = crop_size
        self.augment = augment
        self.normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        self.copy_paste_prob = copy_paste_prob if augment else 0.0
        self.random_erasing_prob = random_erasing_prob if augment else 0.0

        # --- RAM preload + PIL draft (대용량 데이터셋에선 preload=False) ---
        self.preload = preload
        self._cache_images = None
        self._cache_masks = None
        if preload:
            self._cache_images = []
            self._cache_masks = []
            for i in range(len(self.images)):
                img = Image.open(self.images[i])
                img.draft("RGB", (draft_size, draft_size))   # JPEG decode 단계에서 축소
                img = img.convert("RGB").copy()              # 파일 핸들 닫으려 copy
                msk = Image.open(self.masks[i]).copy()
                self._cache_images.append(img)
                self._cache_masks.append(msk)
            print(f"[VOCSegDataset] {year}/{image_set}: {len(self._cache_images)}장 RAM preload 완료")

    def _load(self, idx):
        if self._cache_images is not None:
            return self._cache_images[idx], self._cache_masks[idx]
        return super().__getitem__(idx)

    def __getitem__(self, idx):
        image, mask = self._load(idx)

        if self.augment:
            # 1) 기하 + 광도 aug
            image, mask = train_augment(image, mask, self.crop_size)

            # 2) Copy-Paste (확률적)
            if random.random() < self.copy_paste_prob:
                src_idx = random.randrange(len(self))
                if src_idx == idx:
                    src_idx = (src_idx + 1) % len(self)
                image_src, mask_src = self._load(src_idx)
                image_src, mask_src = augment_src_for_copy_paste(image_src, mask_src, self.crop_size)
                image, mask = copy_paste(image, mask, image_src, mask_src)
        else:
            image, mask = val_transform(image, mask)

        # 텐서 변환 + 정규화
        image = TF.to_tensor(image)
        image = self.normalize(image)
        mask  = torch.from_numpy(np.array(mask)).long()

        # 3) RandomErasing B (image + mask 동시 erase)
        if self.augment and random.random() < self.random_erasing_prob:
            i, j, h, w, v = T.RandomErasing.get_params(
                image, scale=ERASING_SCALE, ratio=ERASING_RATIO, value=[0.0]
            )
            image[:, i:i+h, j:j+w] = v
            mask[i:i+h, j:j+w] = IGNORE_INDEX

        return image, mask


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
    """
    preload=True : VOC 전체 RAM 적재. CopyPaste src fetch 속도 ↑
    preload=False: 디스크 I/O 유지. COCO 합류 등 대용량에서 RAM 보호
    """
    is_train = (image_set == "train")
    datasets = [
        VOCSegDataset(
            root=root, year=year, image_set=image_set,
            crop_size=crop_size, augment=is_train,
            download=download, preload=preload,
        )
        for year in years
    ]
    combined = ConcatDataset(datasets)

    return DataLoader(
        combined,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=is_train,
        persistent_workers=False,               # 메모리 누수 방지
        prefetch_factor=(2 if num_workers > 0 else None),
    )
