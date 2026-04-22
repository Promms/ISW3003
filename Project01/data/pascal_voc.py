import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from torchvision.datasets import VOCSegmentation
import random
import numpy as np
from PIL import Image

VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "potted plant", "sheep", "sofa", "train", "tv/monitor",
]

VOC_COLORMAP = [
    [0, 0, 0],[128, 0, 0], [0, 128, 0], [128, 128, 0],
    [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
    [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
    [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
    [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128],
]

NUM_CLASSES = 21
IGNORE_INDEX = 255
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

_color_jitter = T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)

def train_augment(image, mask, crop_size):
    # 무작위 좌우 반전 (p=0.5)
    if random.random() > 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask)

    # 무작위 회전 (10도 이내) — image는 BILINEAR, mask는 NEAREST + fill=IGNORE_INDEX
    angle = random.uniform(-10.0, 10.0)
    image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
    mask  = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST, fill=IGNORE_INDEX)

    # 무작위 스케일링 0.5 ~ 2.0 (multi-scale training)
    scale = random.uniform(0.5, 2.0)
    new_h = int(image.height * scale)
    new_w = int(image.width * scale)
    image = TF.resize(image, (new_h, new_w), interpolation=Image.BILINEAR)
    mask  = TF.resize(mask, (new_h, new_w), interpolation=Image.NEAREST)

    # crop_size보다 작으면 padding
    pad_h = max(0, crop_size - image.height)
    pad_w = max(0, crop_size - image.width)
    if pad_h > 0 or pad_w > 0:
        image = TF.pad(image, (0, 0, pad_w, pad_h))
        mask  = TF.pad(mask, (0, 0, pad_w, pad_h), fill=IGNORE_INDEX)

    # 무작위 크롭 (crop_size × crop_size)
    i, j, h, w = T.RandomCrop.get_params(image, output_size=(crop_size, crop_size))
    image = TF.crop(image, i, j, h, w)
    mask  = TF.crop(mask, i, j, h, w)

    # 밝기, 대비, 채도 등을 무작위로 변경 (image에만)
    image = _color_jitter(image)

    return image, mask


def val_transform(image, mask):
    # 480 x 640 고정 resize (augmentation 없음)
    image = TF.resize(image, (480, 640), interpolation=Image.BILINEAR)
    mask  = TF.resize(mask, (480, 640), interpolation=Image.NEAREST)
    return image, mask


class VOCSegDataset(VOCSegmentation):
    def __init__(self, root, year="2012", image_set="train", crop_size=320, augment=False, download=False):
        super().__init__(root=root, year=year, image_set=image_set, download=download)
        self.crop_size = crop_size
        self.augment = augment
        self.normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __getitem__(self, idx):
        image, mask = super().__getitem__(idx)

        if self.augment:
            image, mask = train_augment(image, mask, self.crop_size)
        else:
            image, mask = val_transform(image, mask)

        # 텐서 변환
        image = TF.to_tensor(image)                      # (3, H, W)
        image = self.normalize(image)

        mask  = torch.from_numpy(np.array(mask)).long()  # (H, W), int64

        return image, mask


def get_loader(root, years=["2007", "2012"], image_set="train", crop_size=320, batch_size=8, num_workers=2, pin_memory=True, download=False):

    is_train = (image_set == "train")
    datasets = []

    for year in years:
        # 2007과 2012의 dataset을 각각 생성
        dataset = VOCSegDataset(
            root=root,
            year=year,
            image_set=image_set,
            crop_size=crop_size,
            augment=is_train,
            download=download,
        )
        datasets.append(dataset)
    
    # 두 dataset을 concat
    combined_dataset = ConcatDataset(datasets)

    loader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=is_train,  # train은 마지막 불완전 배치 버림
    )

    return loader