from __future__ import annotations

import random

import cv2
import numpy as np


IGNORE_INDEX = 255
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

ERASING_SCALE = (0.02, 0.15)
ERASING_RATIO = (0.3, 3.3)


def _resize(image: np.ndarray, mask: np.ndarray, new_h: int, new_w: int):
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return image, mask


def _rotate(image: np.ndarray, mask: np.ndarray, angle: float):
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    image = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    mask = cv2.warpAffine(
        mask,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=IGNORE_INDEX,
    )
    return image, mask


def _pad_to(image: np.ndarray, mask: np.ndarray, target: int):
    height, width = image.shape[:2]
    pad_h = max(0, target - height)
    pad_w = max(0, target - width)
    if pad_h == 0 and pad_w == 0:
        return image, mask
    image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=IGNORE_INDEX)
    return image, mask


def _random_crop(image: np.ndarray, mask: np.ndarray, crop_size: int):
    height, width = image.shape[:2]
    top = random.randint(0, height - crop_size)
    left = random.randint(0, width - crop_size)
    return (
        image[top:top + crop_size, left:left + crop_size],
        mask[top:top + crop_size, left:left + crop_size],
    )


def _color_jitter(
    image: np.ndarray,
    brightness: float = 0.3,
    contrast: float = 0.3,
    saturation: float = 0.3,
    hue: float = 0.05,
) -> np.ndarray:
    out = image
    if brightness > 0:
        factor = 1.0 + random.uniform(-brightness, brightness)
        out = np.clip(out.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    if contrast > 0:
        factor = 1.0 + random.uniform(-contrast, contrast)
        mean = out.reshape(-1, 3).mean(axis=0)
        out = np.clip((out.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)
    if saturation > 0 or hue > 0:
        hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV).astype(np.float32)
        if hue > 0:
            hsv[..., 0] = (hsv[..., 0] + random.uniform(-hue, hue) * 180.0) % 180.0
        if saturation > 0:
            factor = 1.0 + random.uniform(-saturation, saturation)
            hsv[..., 1] = np.clip(hsv[..., 1] * factor, 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return out


def train_augment(image: np.ndarray, mask: np.ndarray, crop_size: int, apply_blur: bool = True):
    """Apply geometric and photometric training augmentations."""
    if random.random() > 0.5:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    image, mask = _rotate(image, mask, random.uniform(-7.0, 7.0))

    scale = random.uniform(0.5, 1.75)
    height, width = image.shape[:2]
    image, mask = _resize(image, mask, max(1, int(height * scale)), max(1, int(width * scale)))

    image, mask = _pad_to(image, mask, crop_size)
    image, mask = _random_crop(image, mask, crop_size)

    if apply_blur and random.random() < 0.3:
        image = cv2.GaussianBlur(image, (5, 5), random.uniform(0.1, 1.5))
    return _color_jitter(image), mask


def augment_src_for_copy_paste(image: np.ndarray, mask: np.ndarray, crop_size: int):
    """Apply lightweight augmentation to the source sample for copy-paste."""
    if random.random() > 0.5:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    scale = random.uniform(0.3, 1.5)
    height, width = image.shape[:2]
    image, mask = _resize(image, mask, max(1, int(height * scale)), max(1, int(width * scale)))
    image, mask = _pad_to(image, mask, crop_size)
    return _random_crop(image, mask, crop_size)


def copy_paste(
    image_dst: np.ndarray,
    mask_dst: np.ndarray,
    image_src: np.ndarray,
    mask_src: np.ndarray,
    ignore_index: int = IGNORE_INDEX,
):
    paste_mask = (mask_src != 0) & (mask_src != ignore_index)
    if not paste_mask.any():
        return image_dst, mask_dst

    out_img = image_dst.copy()
    out_mask = mask_dst.copy()
    out_img[paste_mask] = image_src[paste_mask]
    out_mask[paste_mask] = mask_src[paste_mask]
    return out_img, out_mask


def val_transform(image: np.ndarray, mask: np.ndarray, size=(480, 640)):
    target_h, target_w = size
    image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return image, mask
