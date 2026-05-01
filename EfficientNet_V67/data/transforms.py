"""
세그멘테이션 augmentation 함수 모음 (OpenCV/NumPy 기반).

PIL 대비 2~3배 빠름. CPU-bound 환경 (Colab 2 vCPU 등) 에서 특히 이득.

I/O:
    - 입력: numpy (H, W, 3) uint8 RGB image + (H, W) uint8 mask
    - 출력: 동일 포맷 (Dataset.__getitem__ 마지막에 tensor 변환)

규칙:
    - cv2 의 shape 인자는 항상 (W, H) 순서 주의
    - 색공간은 RGB로 통일 (cv2.imread 직후 COLOR_BGR2RGB 변환 필요)
    - mask 보간은 반드시 INTER_NEAREST

함수:
    - train_augment: 일반 train용 (geometric + photometric)
    - augment_src_for_copy_paste: Copy-Paste src 전용 (rotation/blur/jitter 제외)
    - copy_paste: Simple Copy-Paste 합성
    - val_transform: validation (고정 resize만)
"""

from __future__ import annotations

import random

import cv2
import numpy as np

IGNORE_INDEX = 255
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# RandomErasing 파라미터 (dataset 쪽에서 사용)
ERASING_SCALE = (0.02, 0.15)
ERASING_RATIO = (0.3, 3.3)


# ---------- 저수준 헬퍼 ----------

def _resize(image: np.ndarray, mask: np.ndarray, new_h: int, new_w: int):
    """cv2.resize 는 (W, H) 순서. image=LINEAR, mask=NEAREST."""
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    mask  = cv2.resize(mask,  (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return image, mask


def _rotate(image: np.ndarray, mask: np.ndarray, angle: float):
    """center rotation. mask border = IGNORE_INDEX (학습에서 무시됨)."""
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    image = cv2.warpAffine(image, M, (w, h),
                           flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    mask  = cv2.warpAffine(mask, M, (w, h),
                           flags=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=IGNORE_INDEX)
    return image, mask


def _pad_to(image: np.ndarray, mask: np.ndarray, target: int):
    """crop_size 미달 시 우/하단 padding. image=0, mask=IGNORE."""
    h, w = image.shape[:2]
    pad_h = max(0, target - h)
    pad_w = max(0, target - w)
    if pad_h == 0 and pad_w == 0:
        return image, mask
    image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    mask  = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=IGNORE_INDEX)
    return image, mask


def _random_crop(image: np.ndarray, mask: np.ndarray, crop_size: int):
    h, w = image.shape[:2]
    top  = random.randint(0, h - crop_size)
    left = random.randint(0, w - crop_size)
    return (
        image[top:top + crop_size, left:left + crop_size],
        mask [top:top + crop_size, left:left + crop_size],
    )


def _color_jitter(image: np.ndarray,
                  brightness: float = 0.3, contrast: float = 0.3,
                  saturation: float = 0.3, hue: float = 0.05) -> np.ndarray:
    """
    PIL ColorJitter와 유사. RGB uint8 입력/출력.

    OpenCV HSV 범위 주의:
        - H: 0~179 (PIL/일반 정의의 절반)
        - S, V: 0~255
    """
    img = image
    # Brightness
    if brightness > 0:
        f = 1.0 + random.uniform(-brightness, brightness)
        img = np.clip(img.astype(np.float32) * f, 0, 255).astype(np.uint8)
    # Contrast (평균값 기준 스케일링)
    if contrast > 0:
        f = 1.0 + random.uniform(-contrast, contrast)
        mean = img.reshape(-1, 3).mean(axis=0)
        img = np.clip((img.astype(np.float32) - mean) * f + mean, 0, 255).astype(np.uint8)
    # Saturation + Hue → HSV 공간에서 한 번에
    if saturation > 0 or hue > 0:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        if hue > 0:
            h_shift = random.uniform(-hue, hue) * 180.0  # OpenCV H 스케일
            hsv[..., 0] = (hsv[..., 0] + h_shift) % 180.0
        if saturation > 0:
            f = 1.0 + random.uniform(-saturation, saturation)
            hsv[..., 1] = np.clip(hsv[..., 1] * f, 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return img


# ---------- 공개 함수 ----------

def train_augment(image: np.ndarray, mask: np.ndarray,
                  crop_size: int, apply_blur: bool = True):
    """기하 + 광도 augmentation. (RGB uint8 + uint8 mask 전제)"""
    # HFlip
    if random.random() > 0.5:
        image = cv2.flip(image, 1)
        mask  = cv2.flip(mask, 1)

    # Rotation ±7°
    angle = random.uniform(-7.0, 7.0)
    image, mask = _rotate(image, mask, angle)

    # Large Scale Jittering [0.5, 1.75]
    scale = random.uniform(0.5, 1.75)
    h, w = image.shape[:2]
    image, mask = _resize(image, mask, max(1, int(h * scale)), max(1, int(w * scale)))

    # Pad (미달 시) + RandomCrop
    image, mask = _pad_to(image, mask, crop_size)
    image, mask = _random_crop(image, mask, crop_size)

    # Photometric
    if apply_blur and random.random() < 0.3:
        sigma = random.uniform(0.1, 1.5)
        image = cv2.GaussianBlur(image, (5, 5), sigma)
    image = _color_jitter(image)

    return image, mask


def augment_src_for_copy_paste(image: np.ndarray, mask: np.ndarray, crop_size: int):
    """Copy-Paste src 전용: LSJ [0.3, 1.5] + HFlip만."""
    if random.random() > 0.5:
        image = cv2.flip(image, 1)
        mask  = cv2.flip(mask, 1)

    scale = random.uniform(0.3, 1.5)
    h, w = image.shape[:2]
    image, mask = _resize(image, mask, max(1, int(h * scale)), max(1, int(w * scale)))

    image, mask = _pad_to(image, mask, crop_size)
    image, mask = _random_crop(image, mask, crop_size)
    return image, mask


def copy_paste(image_dst: np.ndarray, mask_dst: np.ndarray,
               image_src: np.ndarray, mask_src: np.ndarray,
               ignore_index: int = IGNORE_INDEX):
    """Simple Copy-Paste: src의 전경(≠0, ≠ignore)을 dst에 덮어씌움."""
    paste_mask = (mask_src != 0) & (mask_src != ignore_index)
    if not paste_mask.any():
        return image_dst, mask_dst

    # copy 후 boolean indexing (np.where 보다 메모리/속도 유리)
    out_img = image_dst.copy()
    out_msk = mask_dst.copy()
    out_img[paste_mask] = image_src[paste_mask]
    out_msk[paste_mask] = mask_src[paste_mask]
    return out_img, out_msk


def val_transform(image: np.ndarray, mask: np.ndarray, size=(480, 640)):
    """Validation: 고정 resize만. size=(H, W)."""
    target_h, target_w = size
    image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    mask  = cv2.resize(mask,  (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return image, mask
