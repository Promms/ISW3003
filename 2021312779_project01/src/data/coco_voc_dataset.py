from __future__ import annotations

import os
import random

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset

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


COCO_TO_VOC: dict[int, int] = {
    5: 1,    # airplane -> aeroplane
    2: 2,    # bicycle
    16: 3,   # bird
    9: 4,    # boat
    44: 5,   # bottle
    6: 6,    # bus
    3: 7,    # car
    17: 8,   # cat
    62: 9,   # chair
    21: 10,  # cow
    67: 11,  # dining table
    18: 12,  # dog
    19: 13,  # horse
    4: 14,   # motorcycle
    1: 15,   # person
    64: 16,  # potted plant
    20: 17,  # sheep
    63: 18,  # couch
    7: 19,   # train
    72: 20,  # tv
}


class CocoVOCSegDataset(Dataset):
    """COCO instance annotations converted to VOC 21-class segmentation masks."""

    def __init__(
        self,
        img_root: str,
        ann_file: str,
        crop_size: int = 320,
        augment: bool = False,
        filter_empty: bool = True,
        copy_paste_prob: float = 0.1,
        random_erasing_prob: float = 0.5,
        overlap_policy: str = "smallest_first",
        mask_cache_dir: str | None = None,
    ):
        # Keep pycocotools lazy so evaluation and inference do not depend on COCO.
        try:
            from pycocotools.coco import COCO
        except ImportError as exc:
            raise ImportError("Install pycocotools to use COCO training data.") from exc

        if overlap_policy not in ("smallest_first", "ignore"):
            raise ValueError(f"Unknown overlap_policy: {overlap_policy}")

        self.img_root = img_root
        self.crop_size = crop_size
        self.augment = augment
        self.normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        self.copy_paste_prob = copy_paste_prob if augment else 0.0
        self.random_erasing_prob = random_erasing_prob if augment else 0.0
        self.overlap_policy = overlap_policy
        self.mask_cache_dir = mask_cache_dir

        self.coco = COCO(ann_file)
        if filter_empty:
            img_ids: set[int] = set()
            for category_id in COCO_TO_VOC:
                img_ids.update(self.coco.getImgIds(catIds=[category_id]))
            self.ids = sorted(img_ids)
        else:
            self.ids = sorted(self.coco.getImgIds())

        print(f"[CocoVOCSegDataset] {os.path.basename(ann_file)}: {len(self.ids)} images")

    def __len__(self) -> int:
        return len(self.ids)

    def _build_mask(self, img_id: int, height: int, width: int) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.uint8)
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)

        entries = []
        for ann in anns:
            voc_idx = COCO_TO_VOC.get(ann["category_id"])
            if voc_idx is None:
                continue
            try:
                ann_mask = self.coco.annToMask(ann).astype(bool)
            except Exception:
                continue
            if ann_mask.shape != mask.shape:
                continue
            area = int(ann_mask.sum())
            if area > 0:
                entries.append((area, voc_idx, ann_mask))

        if self.overlap_policy == "smallest_first":
            # VOC-style semantic masks keep one class per pixel when COCO instances overlap.
            entries.sort(key=lambda entry: -entry[0])
            for _, voc_idx, ann_mask in entries:
                mask[ann_mask] = voc_idx
        else:
            for _, voc_idx, ann_mask in entries:
                conflict = ann_mask & (mask != 0) & (mask != voc_idx)
                mask[ann_mask & ~conflict] = voc_idx
                mask[conflict] = IGNORE_INDEX

        return mask

    def _load(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        img_id = self.ids[idx]
        info = self.coco.loadImgs(img_id)[0]
        image_path = os.path.join(self.img_root, info["file_name"])

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = None
        if self.mask_cache_dir is not None:
            cache_path = os.path.join(self.mask_cache_dir, f"{img_id:012d}.png")
            if os.path.exists(cache_path):
                mask = cv2.imread(cache_path, cv2.IMREAD_UNCHANGED)
                if mask is None:
                    mask = np.array(Image.open(cache_path))

        if mask is None:
            mask = self._build_mask(img_id, info["height"], info["width"])
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


def get_combined_loader(
    voc_datasets: list,
    coco_dataset: CocoVOCSegDataset | None,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    shuffle: bool,
    drop_last: bool,
) -> DataLoader:
    datasets = list(voc_datasets)
    if coco_dataset is not None:
        datasets.append(coco_dataset)

    return DataLoader(
        ConcatDataset(datasets),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=False,
        prefetch_factor=(6 if num_workers > 0 else None),
    )
