"""
세그멘테이션 augmentation 함수 모음.

- train_augment: 일반 train용 (geometric + photometric)
- augment_src_for_copy_paste: Copy-Paste src 전용 (rotation/blur/colorjitter 제외)
- copy_paste: Simple Copy-Paste (Ghiasi et al. 2021) 합성
- val_transform: validation (고정 resize만)

Dataset 쪽에선 이 함수들을 import해서 __getitem__에서 호출.
"""

from __future__ import annotations

import random

import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

IGNORE_INDEX = 255
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# ColorJitter 인스턴스 (상태 없음 → 모듈 레벨 재사용 OK)
_color_jitter = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)

# RandomErasing B 방식 파라미터
# - scale=(0.02, 0.15): 지워지는 영역 면적 2~15%
# - ratio=(0.3, 3.3):    사각형 가로/세로 비
# B 방식: image와 mask 동시에 erase (mask는 IGNORE_INDEX로) → 학습 신호 모순 제거
ERASING_SCALE = (0.02, 0.15)
ERASING_RATIO = (0.3, 3.3)


def train_augment(image, mask, crop_size, apply_blur: bool = True):
    """
    기하 + 광도 augmentation.

    apply_blur: Copy-Paste src에는 False. 논문에서 src blur는 효과 없음 확인.
    """
    # HFlip (p=0.5)
    if random.random() > 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask)

    # Rotation ±7°, mask fill=IGNORE_INDEX
    angle = random.uniform(-7.0, 7.0)
    image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
    mask  = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST, fill=IGNORE_INDEX)

    # Large Scale Jittering [0.5, 1.75] — Copy-Paste 논문의 핵심 요소
    scale = random.uniform(0.5, 1.75)
    new_h = int(image.height * scale)
    new_w = int(image.width * scale)
    image = TF.resize(image, (new_h, new_w), interpolation=Image.BILINEAR)
    mask  = TF.resize(mask, (new_h, new_w), interpolation=Image.NEAREST)

    # crop_size 미달 시 padding
    pad_h = max(0, crop_size - image.height)
    pad_w = max(0, crop_size - image.width)
    if pad_h > 0 or pad_w > 0:
        image = TF.pad(image, (0, 0, pad_w, pad_h))
        mask  = TF.pad(mask, (0, 0, pad_w, pad_h), fill=IGNORE_INDEX)

    # RandomCrop
    i, j, h, w = T.RandomCrop.get_params(image, output_size=(crop_size, crop_size))
    image = TF.crop(image, i, j, h, w)
    mask  = TF.crop(mask, i, j, h, w)

    # Photometric (image only)
    if apply_blur and random.random() < 0.3:
        sigma = random.uniform(0.1, 1.5)
        image = TF.gaussian_blur(image, kernel_size=5, sigma=sigma)
    image = _color_jitter(image)

    return image, mask


def augment_src_for_copy_paste(image, mask, crop_size):
    """
    Copy-Paste src 전용: LSJ [0.3, 1.5] + HFlip만.

    의도적으로 제외:
      - Rotation: mask에 IGNORE 생성 → paste_mask 왜곡
      - ColorJitter/Blur: 붙여넣는 객체의 특징 흐림
      - RandomErasing: dst에서만으로 충분
    """
    if random.random() > 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask)

    scale = random.uniform(0.3, 1.5)
    new_h = max(1, int(image.height * scale))
    new_w = max(1, int(image.width * scale))
    image = TF.resize(image, (new_h, new_w), interpolation=Image.BILINEAR)
    mask  = TF.resize(mask, (new_h, new_w), interpolation=Image.NEAREST)

    pad_h = max(0, crop_size - image.height)
    pad_w = max(0, crop_size - image.width)
    if pad_h > 0 or pad_w > 0:
        image = TF.pad(image, (0, 0, pad_w, pad_h))
        mask  = TF.pad(mask, (0, 0, pad_w, pad_h), fill=IGNORE_INDEX)

    i, j, h, w = T.RandomCrop.get_params(image, output_size=(crop_size, crop_size))
    image = TF.crop(image, i, j, h, w)
    mask  = TF.crop(mask, i, j, h, w)
    return image, mask


def copy_paste(image_dst, mask_dst, image_src, mask_src, ignore_index: int = IGNORE_INDEX):
    """
    Simple Copy-Paste: src의 전경을 dst에 덮어씌움.

    전경 = (mask != 0) AND (mask != ignore_index)
    """
    img_dst = np.array(image_dst)
    msk_dst = np.array(mask_dst)
    img_src = np.array(image_src)
    msk_src = np.array(mask_src)

    paste_mask = (msk_src != 0) & (msk_src != ignore_index)
    if not paste_mask.any():
        return image_dst, mask_dst

    img_out = np.where(paste_mask[..., None], img_src, img_dst)
    msk_out = np.where(paste_mask, msk_src, msk_dst)
    return Image.fromarray(img_out.astype(np.uint8)), Image.fromarray(msk_out.astype(np.uint8))


def val_transform(image, mask, size=(480, 640)):
    """Validation: 고정 resize만 (aug 없음)."""
    image = TF.resize(image, size, interpolation=Image.BILINEAR)
    mask  = TF.resize(mask, size, interpolation=Image.NEAREST)
    return image, mask
