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

# RandomErasing 파라미터 상수 (get_params 호출 시 직접 사용)
# - scale=(0.02, 0.2): 지워지는 영역 면적이 전체의 2~20%
# - ratio=(0.3, 3.3): 사각형 가로/세로 비
# - value=0.0: normalize 후 평균값 근처(대략 회색) → 너무 튀지 않음
# B 방식: image와 mask를 동시에 지워 불가능한 학습 신호를 제거함
#   - 큰 객체: 일부만 IGNORE → 나머지로 학습 가능
#   - 작은 객체: 전체가 IGNORE → penalty 없음 (올바른 처리)
_ERASING_SCALE = (0.02, 0.2)
_ERASING_RATIO = (0.3, 3.3)


def train_augment(image, mask, crop_size, apply_blur=True):
    """
    기하/광도 augmentation.

    apply_blur: Copy-Paste의 src 이미지에는 False로 호출.
      논문(Simple Copy-Paste)에서 붙여넣는 객체에 블러를 적용하면
      성능 향상이 없음을 확인하고 의도적으로 배제함.
    """
    # 무작위 좌우 반전 (p=0.5)
    if random.random() > 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask)

    # 무작위 회전 (10도 이내) — image는 BILINEAR, mask는 NEAREST + fill=IGNORE_INDEX
    angle = random.uniform(-10.0, 10.0)
    image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
    mask  = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST, fill=IGNORE_INDEX)

    # 무작위 스케일링 0.5 ~ 2.0 (Large Scale Jittering, LSJ)
    # Copy-Paste 논문에서 성능 향상의 핵심 요소로 꼽음
    # src/dst 각각 독립적으로 적용 → 다양한 크기의 객체 조합 학습
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

    # --- 광도 (photometric) augmentation: image에만 적용, mask 손대지 않음 ---

    # Gaussian Blur (p=0.5) — dst 이미지에만 적용 (apply_blur=True일 때)
    # - kernel_size=5, sigma는 0.1~2.0 사이 랜덤
    # - src 이미지(apply_blur=False): Copy-Paste 논문에서 붙여넣는 객체에 blur 적용 시
    #   성능 향상 없음 확인 → src 호출 시에는 생략
    if apply_blur and random.random() < 0.5:
        sigma = random.uniform(0.1, 2.0)
        image = TF.gaussian_blur(image, kernel_size=5, sigma=sigma)

    # 밝기, 대비, 채도, 색상을 무작위로 변경 (image에만)
    image = _color_jitter(image)

    return image, mask


def copy_paste(image_dst, mask_dst, image_src, mask_src, ignore_index=IGNORE_INDEX):
    """
    Copy-Paste augmentation (멀티이미지 증강).

    src 이미지에서 "전경(=배경 0도 아니고 ignore 255도 아닌 픽셀)"을 통째로 잘라
    dst 이미지 위에 붙여넣음. mask도 동일하게 덮어씌움.

    효과:
      - 같은 클래스를 다양한 배경에서 보게 됨 → 컨텍스트 일반화
      - 작은 클래스(bottle, bird 등)가 더 자주 등장하게 됨 → class imbalance 완화
      - 한 장에 더 많은 인스턴스가 등장 → 학습 효율 ↑

    구현:
      1) src의 mask에서 전경 픽셀을 boolean으로 뽑음 (paste_mask)
      2) np.where(paste_mask, src, dst)로 픽셀 단위 합성
      3) 모든 텐서/이미지가 동일 크기(crop_size×crop_size)임을 가정 (호출 전 train_augment 거친 상태)
    """
    # PIL → numpy
    img_dst = np.array(image_dst)              # (H, W, 3) uint8
    msk_dst = np.array(mask_dst)               # (H, W)    uint8
    img_src = np.array(image_src)              # (H, W, 3) uint8
    msk_src = np.array(mask_src)               # (H, W)    uint8

    # src에서 "유효한 전경" 픽셀만 paste 대상
    # - 배경(0): 붙이면 dst의 배경을 dst의 배경으로 덮는 의미 없는 짓 → 제외
    # - ignore(255): 라벨이 없는 픽셀이라 붙이면 학습 신호 망가짐 → 제외
    paste_mask = (msk_src != 0) & (msk_src != ignore_index)   # (H, W) bool

    if not paste_mask.any():
        # src에 전경이 하나도 없는 이미지면 그냥 dst 그대로 반환
        return image_dst, mask_dst

    # 이미지 합성: paste_mask 영역만 src로 덮어쓰기
    # paste_mask는 (H,W)인데 image는 (H,W,3) → [..., None]로 broadcast
    img_out = np.where(paste_mask[..., None], img_src, img_dst)
    # mask 합성: 동일 영역에 src의 클래스 ID로 덮어쓰기
    msk_out = np.where(paste_mask, msk_src, msk_dst)

    # numpy → PIL (이후 to_tensor 단계와 호환)
    image_out = Image.fromarray(img_out.astype(np.uint8))
    mask_out  = Image.fromarray(msk_out.astype(np.uint8))
    return image_out, mask_out


def val_transform(image, mask):
    # 480 x 640 고정 resize (augmentation 없음)
    image = TF.resize(image, (480, 640), interpolation=Image.BILINEAR)
    mask  = TF.resize(mask, (480, 640), interpolation=Image.NEAREST)
    return image, mask


class VOCSegDataset(VOCSegmentation):
    def __init__(self, root, year="2012", image_set="train", crop_size=320, augment=False,
                 download=False, copy_paste_prob=0.5, random_erasing_prob=0.5):
        super().__init__(root=root, year=year, image_set=image_set, download=download)
        self.crop_size = crop_size
        self.augment = augment
        self.normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

        # Copy-Paste 확률: train일 때만 사용. 0이면 비활성화
        self.copy_paste_prob = copy_paste_prob if augment else 0.0
        # RandomErasing 확률: 0이면 적용 안 함. T.RandomErasing 인스턴스의 p와 곱해지진 않고
        # 호출 자체를 확률적으로 결정 (가독성 위해 명시적으로 분리)
        self.random_erasing_prob = random_erasing_prob if augment else 0.0

    def __getitem__(self, idx):
        image, mask = super().__getitem__(idx)

        if self.augment:
            # 1) 기하/광도 augmentation 적용 (crop_size 정사각형으로 통일)
            image, mask = train_augment(image, mask, self.crop_size)

            # 2) Copy-Paste: 다른 인덱스에서 한 장 더 뽑아 전경을 합성
            #    - src 이미지에도 LSJ + flip 적용 (논문 방식: 각 이미지 독립적으로 aug)
            #    - apply_blur=False: 붙여넣는 객체에 blur 미적용 (논문 명시)
            #    - 그 후 copy_paste()로 전경만 dst에 덮어씌움
            if random.random() < self.copy_paste_prob:
                # 자기 자신을 src로 뽑으면 의미 없으므로 다른 idx를 뽑음
                src_idx = random.randrange(len(self))
                if src_idx == idx:
                    src_idx = (src_idx + 1) % len(self)
                image_src, mask_src = super().__getitem__(src_idx)
                image_src, mask_src = train_augment(image_src, mask_src, self.crop_size, apply_blur=False)
                image, mask = copy_paste(image, mask, image_src, mask_src)

        else:
            image, mask = val_transform(image, mask)

        # 텐서 변환
        image = TF.to_tensor(image)                      # (3, H, W), float [0, 1]
        image = self.normalize(image)                    # ImageNet 평균/표준편차 정규화
        mask  = torch.from_numpy(np.array(mask)).long()  # (H, W), int64

        # 3) RandomErasing B 방식: image + mask 동시 erase
        #    - image: 사각형 영역을 0(normalize 기준 회색)으로 덮어 occlusion 시뮬레이션
        #    - mask:  동일 영역을 IGNORE_INDEX(255)로 설정 → loss 계산에서 제외
        #    효과:
        #      - 큰 객체: 일부만 IGNORE → 나머지 픽셀로 정상 학습
        #      - 작은 객체: 전체가 IGNORE → 불가능한 신호 없음
        if self.augment and random.random() < self.random_erasing_prob:
            i, j, h, w, v = T.RandomErasing.get_params(
                image, scale=_ERASING_SCALE, ratio=_ERASING_RATIO, value=[0.0]
            )
            image[:, i:i+h, j:j+w] = v       # image 영역 지우기
            mask[i:i+h, j:j+w] = IGNORE_INDEX  # mask 동일 영역 ignore 처리

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