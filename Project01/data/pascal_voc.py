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

class VOCSegDataset(VOCSegmentation):
    def __init__(self, root, year="2012", image_set="train", crop_size=320, augment=False, download=False):
        super().__init__(root=root, year=year, image_set=image_set, download=download)
        self.crop_size = crop_size
        self.augment = augment
        self.normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __getitem__(self, idx):
        image, mask = super().__getitem__(idx)

        # augmentation (train only)
        if self.augment:
            # horizontal flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # random scale 0.5 ~ 2.0
            scale = random.uniform(0.5, 2.0)
            new_h = int(image.height * scale)
            new_w = int(image.width * scale)
            image = TF.resize(image, (new_h, new_w), interpolation=Image.BILINEAR)
            mask  = TF.resize(mask, (new_h, new_w), interpolation=Image.NEAREST)

            # crop_size보다 작아지면 padding 
            pad_h = max(0, self.crop_size - image.height)
            pad_w = max(0, self.crop_size - image.width)
            if pad_h > 0 or pad_w > 0:
                image = TF.pad(image, (0, 0, pad_w, pad_h))
                mask = TF.pad(mask, (0, 0, pad_w, pad_h), fill=IGNORE_INDEX)

            # random crop
            i, j, h, w = T.RandomCrop.get_params(image, output_size=(self.crop_size, self.crop_size))
            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)

        else:
            # 480 x 640 Resize
            image = TF.resize(image, (480, 640), interpolation=Image.BILINEAR)
            mask = TF.resize(mask, (480, 640), interpolation=Image.NEAREST)

        # 텐서 변환
        image = TF.to_tensor(image)                      # (3, H, W)
        image = self.normalize(image)

        mask  = torch.from_numpy(np.array(mask)).long()  # (H, W), int64

        return image, mask


def get_loader(root, years=["2007", "2012"], image_set="train", crop_size=320, batch_size=8, num_workers=2, download=False):

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
        pin_memory=True,
        drop_last=is_train,  # train은 마지막 불완전 배치 버림
    )

    return loader