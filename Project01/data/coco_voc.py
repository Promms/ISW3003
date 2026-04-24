"""
COCO 2017 → VOC 21-class 매핑 Segmentation Dataset.

torchvision references/segmentation/v2_extras.py 의 `CocoDetectionToVOCSegmentation`
를 참고했으나, 우리 파이프라인(PIL + classic torchvision transforms) 스타일로 재작성.

핵심:
  - COCO category_id (1~90 불연속) → VOC index (0=bg, 1~20=foreground) 로 매핑
  - VOC에 없는 COCO 클래스(70개)는 전부 background(0) 처리
  - 여러 VOC 클래스가 겹치는 픽셀은 IGNORE_INDEX(255) — 학습 신호 모순 제거
  - 출력 포맷은 VOCSegDataset과 동일: (image_tensor, mask_long)
    → ConcatDataset 으로 그대로 합쳐서 쓸 수 있음

사용:
    from data.coco_voc import CocoVOCSegDataset, get_coco_loader
    ds = CocoVOCSegDataset(
        img_root="/content/coco/train2017",
        ann_file="/content/coco/annotations/instances_train2017.json",
        crop_size=320, augment=True,
    )
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset

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


# COCO category_id → VOC index (1~20). 0=background는 기본값이라 빠짐.
# 순서: VOC 공식 클래스 순서대로 나열 (aeroplane=1, bicycle=2, ..., tvmonitor=20)
COCO_TO_VOC: dict[int, int] = {
    5:  1,   # airplane   → aeroplane
    2:  2,   # bicycle
    16: 3,   # bird
    9:  4,   # boat
    44: 5,   # bottle
    6:  6,   # bus
    3:  7,   # car
    17: 8,   # cat
    62: 9,   # chair
    21: 10,  # cow
    67: 11,  # dining table → diningtable
    18: 12,  # dog
    19: 13,  # horse
    4:  14,  # motorcycle → motorbike
    1:  15,  # person
    64: 16,  # potted plant → pottedplant
    20: 17,  # sheep
    63: 18,  # couch → sofa
    7:  19,  # train
    72: 20,  # tv → tvmonitor
}


class CocoVOCSegDataset(Dataset):
    """
    COCO instances annotation → VOC 21-class segmentation mask.

    Args:
        img_root    : COCO 이미지 폴더 (예: .../train2017)
        ann_file    : instances_train2017.json 경로
        crop_size   : augment crop 크기
        augment     : True면 train_augment + copy-paste + random erasing 적용
        filter_empty: True면 VOC 매핑 클래스가 하나도 없는 이미지는 제외
                      (COCO 118k 중 ~65k 정도 남음 — 효율 ↑)
        copy_paste_prob, random_erasing_prob: VOC dataset과 동일 의미
        mark_overlap_ignore: True면 여러 VOC 클래스가 겹치는 픽셀을 ignore 처리
    """

    def __init__(
        self,
        img_root: str,
        ann_file: str,
        crop_size: int = 320,
        augment: bool = False,
        filter_empty: bool = True,
        copy_paste_prob: float = 0.3,
        random_erasing_prob: float = 0.5,
        overlap_policy: str = "smallest_first",  # "smallest_first" | "ignore"
        mask_cache_dir: str | None = None,        # 사전 렌더된 VOC-index PNG 폴더
    ):
        try:
            from pycocotools.coco import COCO
        except ImportError as e:
            raise ImportError(
                "pycocotools 가 필요합니다. `pip install pycocotools` 로 설치하세요."
            ) from e

        self.img_root = img_root
        self.crop_size = crop_size
        self.augment = augment
        self.normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        self.copy_paste_prob = copy_paste_prob if augment else 0.0
        self.random_erasing_prob = random_erasing_prob if augment else 0.0
        assert overlap_policy in ("smallest_first", "ignore"), \
            f"unknown overlap_policy: {overlap_policy}"
        self.overlap_policy = overlap_policy
        self.mask_cache_dir = mask_cache_dir

        self.coco = COCO(ann_file)
        self._voc_cat_ids = list(COCO_TO_VOC.keys())

        if filter_empty:
            img_ids: set[int] = set()
            for cid in self._voc_cat_ids:
                img_ids.update(self.coco.getImgIds(catIds=[cid]))
            self.ids = sorted(img_ids)
        else:
            self.ids = sorted(self.coco.getImgIds())

        print(f"[CocoVOCSegDataset] {os.path.basename(ann_file)}: "
              f"{len(self.ids)} images (filter_empty={filter_empty})")
        if self.mask_cache_dir:
            print(f"[CocoVOCSegDataset] mask_cache_dir = {self.mask_cache_dir} "
                  f"(없는 것은 on-the-fly 폴백)")

    def __len__(self) -> int:
        return len(self.ids)

    # ---------- internal ----------
    def _build_mask(self, img_id: int, H: int, W: int) -> np.ndarray:
        """
        COCO instance annotations → (H, W) uint8 VOC index mask.

        overlap_policy:
          - "smallest_first" (기본, 정석):
              큰 객체부터 그리고 작은 객체로 덮어쓰기 (Smallest Area First).
              작은 물체(병/안경 등)가 큰 물체(사람) 위에 있을 확률이 높다는 가설.
              → 작은 객체의 형태가 온전히 보존됨.
          - "ignore":
              서로 다른 VOC 클래스가 겹치는 픽셀을 IGNORE_INDEX(255)로 마킹.
              학습 신호 모순은 완전히 제거되지만, 작은 객체가 사라질 수 있음.

        crowd=True 인 어노테이션은 마스크 품질이 낮아 스킵.
        """
        mask = np.zeros((H, W), dtype=np.uint8)

        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)

        # VOC 매핑되는 annotation만 추출 + 마스크 미리 디코드
        entries = []  # list of (area, voc_idx, m_bool)
        for ann in anns:
            voc_idx = COCO_TO_VOC.get(ann["category_id"])
            if voc_idx is None:
                continue
            try:
                m = self.coco.annToMask(ann)
            except Exception:
                continue
            if m.shape != mask.shape:
                continue
            m_bool = m.astype(bool)
            # ann['area'] 가 있지만 segmentation area가 아닌 경우도 있어서 픽셀 count로 신뢰성 확보
            area = int(m_bool.sum())
            if area == 0:
                continue
            entries.append((area, voc_idx, m_bool))

        if self.overlap_policy == "smallest_first":
            # 큰 순 → 작은 순으로 그림 (작은 게 위로 올라와 덮음)
            entries.sort(key=lambda e: -e[0])
            for _, voc_idx, m_bool in entries:
                mask[m_bool] = voc_idx
        else:  # "ignore"
            for _, voc_idx, m_bool in entries:
                conflict = m_bool & (mask != 0) & (mask != voc_idx)
                mask[m_bool & ~conflict] = voc_idx
                mask[conflict] = IGNORE_INDEX

        return mask

    def _load(self, idx: int) -> tuple[Image.Image, Image.Image]:
        img_id = self.ids[idx]
        info = self.coco.loadImgs(img_id)[0]
        path = os.path.join(self.img_root, info["file_name"])
        image = Image.open(path).convert("RGB")

        # 사전 렌더된 PNG 우선 시도 (2~5ms) → 실패 시 annToMask 재계산 (15~50ms)
        mask = None
        if self.mask_cache_dir is not None:
            cache_path = os.path.join(self.mask_cache_dir, f"{img_id:012d}.png")
            if os.path.exists(cache_path):
                mask = Image.open(cache_path)
                mask.load()  # 파일 핸들 닫기 (worker 재사용 시 FD 누수 방지)

        if mask is None:
            mask_np = self._build_mask(img_id, info["height"], info["width"])
            mask = Image.fromarray(mask_np, mode="L")

        return image, mask

    # ---------- public ----------
    def __getitem__(self, idx: int):
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

        image = TF.to_tensor(image)
        image = self.normalize(image)
        mask  = torch.from_numpy(np.array(mask)).long()

        if self.augment and random.random() < self.random_erasing_prob:
            i, j, h, w, v = T.RandomErasing.get_params(
                image, scale=ERASING_SCALE, ratio=ERASING_RATIO, value=[0.0]
            )
            image[:, i:i+h, j:j+w] = v
            mask[i:i+h, j:j+w] = IGNORE_INDEX

        return image, mask


def get_coco_loader(
    img_root: str,
    ann_file: str,
    crop_size: int = 320,
    batch_size: int = 8,
    num_workers: int = 2,
    pin_memory: bool = True,
    augment: bool = True,
    filter_empty: bool = True,
) -> DataLoader:
    """COCO 단독 DataLoader. VOC과 합칠 땐 get_combined_loader 사용 권장."""
    ds = CocoVOCSegDataset(
        img_root=img_root, ann_file=ann_file,
        crop_size=crop_size, augment=augment, filter_empty=filter_empty,
    )
    return DataLoader(
        ds, batch_size=batch_size, shuffle=augment,
        num_workers=num_workers, pin_memory=pin_memory,
        drop_last=augment, persistent_workers=(num_workers > 0),
        prefetch_factor=(2 if num_workers > 0 else None),
    )


def get_combined_loader(
    voc_datasets: list,
    coco_dataset: CocoVOCSegDataset | None,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    shuffle: bool,
    drop_last: bool,
) -> DataLoader:
    """
    VOC datasets (list) + COCO dataset → ConcatDataset → DataLoader.

    최종 평가는 둘 다와 다른 자체 데이터셋이라 (generalization 목표),
    별도 weighted sampling 없이 uniform concat.
    """
    datasets = list(voc_datasets)
    if coco_dataset is not None:
        datasets.append(coco_dataset)
    combined = ConcatDataset(datasets)

    return DataLoader(
        combined, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
        drop_last=drop_last, persistent_workers=(num_workers > 0),
        prefetch_factor=(2 if num_workers > 0 else None),
    )
