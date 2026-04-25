"""
제출용 inference 스크립트.

사용법:
    python predict.py \
        --ckpt checkpoints/model.pth \
        --backbone efficientnet \
        --img_dir submit/img \
        --pred_dir submit/pred

동작:
    submit/img/0001.jpg → submit/pred/0001.png (원본 해상도, class index 0~20)
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

from utils.model_factory import build_model


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# VOC 컬러 팔레트 (index PNG를 시각화할 때 자동 적용)
VOC_PALETTE = [
    0, 0, 0,      128, 0, 0,     0, 128, 0,     128, 128, 0,
    0, 0, 128,    128, 0, 128,   0, 128, 128,   128, 128, 128,
    64, 0, 0,     192, 0, 0,     64, 128, 0,    192, 128, 0,
    64, 0, 128,   192, 0, 128,   64, 128, 128,  192, 128, 128,
    0, 64, 0,     128, 64, 0,    0, 192, 0,     128, 192, 0,
    0, 64, 128,
]
VOC_PALETTE += [0] * (256 * 3 - len(VOC_PALETTE))  # 256색 팔레트로 패딩


def preprocess(image: Image.Image, infer_size: tuple[int, int]) -> torch.Tensor:
    """PIL 이미지 → 정규화된 텐서 (1, 3, H, W)."""
    img_resized = TF.resize(image, infer_size, interpolation=Image.BILINEAR)
    tensor = TF.to_tensor(img_resized)
    tensor = TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return tensor.unsqueeze(0)


@torch.no_grad()
def tta_forward(
    model: nn.Module,
    x: torch.Tensor,
    scales: tuple[float, ...],
    use_flip: bool,
) -> torch.Tensor:
    """
    Test-Time Augmentation forward.

    같은 입력을 여러 스케일 + 좌우 반전으로 변형해 forward → softmax 확률 평균 반환.

    동작:
      1) 각 scale마다 입력을 bilinear로 resize → forward → 원본 (H, W)로 resize back
      2) use_flip=True면 좌우 반전 버전도 forward → 출력도 다시 반전해 좌표계 복원
      3) 모든 버전의 softmax 확률을 단순 평균

    인자:
      x: (1, 3, H, W) 정규화된 입력 텐서 (infer_size로 이미 resize된 상태)
      scales: 입력에 곱할 스케일 (1.0 포함 권장)
      use_flip: True면 좌우 반전 버전도 평균에 포함

    반환:
      (1, C, H, W) 평균 softmax 확률. argmax 하면 최종 예측.

    비용: forward 횟수 = len(scales) × (2 if use_flip else 1)
    """
    _, _, H, W = x.shape
    prob_sum = None
    n_aug = 0

    for scale in scales:
        # 1) scale 적용
        if scale != 1.0:
            new_h = int(round(H * scale))
            new_w = int(round(W * scale))
            x_s = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
        else:
            x_s = x

        # 2) forward → 원본 (H, W)로 resize back 후 softmax
        logits = model(x_s)
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        probs = F.softmax(logits, dim=1)

        if prob_sum is None:
            prob_sum = torch.zeros_like(probs)
        prob_sum += probs
        n_aug += 1

        # 3) flip 버전
        if use_flip:
            x_flip = torch.flip(x_s, dims=[3])           # W축 반전
            logits_flip = model(x_flip)
            logits_flip = F.interpolate(logits_flip, size=(H, W), mode="bilinear", align_corners=False)
            logits_flip = torch.flip(logits_flip, dims=[3])   # 출력도 반전해 좌표계 복원
            prob_sum += F.softmax(logits_flip, dim=1)
            n_aug += 1

    return prob_sum / n_aug


@torch.no_grad()
def predict_one(
    model: nn.Module,
    image: Image.Image,
    device: torch.device,
    infer_size: tuple[int, int],
    tta: bool = False,
    tta_scales: tuple[float, ...] = (0.75, 1.0, 1.25),
    tta_flip: bool = True,
) -> np.ndarray:
    """
    단일 이미지 → (H, W) uint8 numpy array (class index 0~20).
    원본 크기로 복원해서 반환.

    tta=True면 multi-scale + flip softmax 평균으로 예측. 학습 단계에서는 쓰지 않고
    추론 단계에서만 mIoU를 끌어올리는 용도.
    """
    orig_w, orig_h = image.size  # PIL은 (W, H)

    # 1) inference 해상도로 resize → 모델 forward (TTA on/off)
    x = preprocess(image, infer_size).to(device)
    if tta:
        # TTA는 softmax 확률을 반환 — logits처럼 사용해도 argmax는 동일
        out = tta_forward(model, x, scales=tta_scales, use_flip=tta_flip)
    else:
        out = model(x)  # (1, C, H_infer, W_infer) logits

    # 2) 원본 크기로 복원 후 argmax
    #    argmax 후 NEAREST로 올리면 경계가 계단처럼 됨 → logits/probs를 먼저 올리는 게 더 좋음
    out = F.interpolate(out, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
    pred = out.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    return pred


def save_pred_png(pred: np.ndarray, out_path: Path) -> None:
    """
    class index를 PNG로 저장 (mode='P' 팔레트 적용).
    채점 스크립트는 보통 np.array(Image.open(...))로 읽으므로 팔레트 유무는 무관하지만,
    시각적으로 확인하기 쉽게 VOC 팔레트를 입힘.
    """
    img = Image.fromarray(pred, mode="P")
    img.putpalette(VOC_PALETTE)
    img.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True,
                        help="체크포인트 경로 (.pth)")
    parser.add_argument("--num_classes", type=int, default=21)
    parser.add_argument("--img_dir", type=str, default="submit/img",
                        help="입력 이미지 폴더 (jpg)")
    parser.add_argument("--pred_dir", type=str, default="submit/pred",
                        help="예측 결과 저장 폴더 (png)")
    parser.add_argument("--infer_h", type=int, default=480,
                        help="모델 입력 높이 (FLOPs 기준 480)")
    parser.add_argument("--infer_w", type=int, default=640,
                        help="모델 입력 너비 (FLOPs 기준 640)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--rename_by_index", action="store_true",
                        help="정렬 순서대로 000.png ~ 999.png 로 강제 저장 "
                             "(테스트 이미지 파일명이 000~999 형식이 아닐 때)")
    # --- TTA 옵션 ---
    parser.add_argument("--tta", action="store_true",
                        help="Test-Time Augmentation 사용 (multi-scale + flip 앙상블)")
    parser.add_argument("--tta_scales", type=float, nargs="+",
                        default=[0.75, 1.0, 1.25],
                        help="TTA 스케일 목록 (1.0 포함 권장)")
    parser.add_argument("--no_tta_flip", action="store_true",
                        help="TTA에서 좌우 반전을 사용하지 않음 (기본: 사용)")
    # --- EMA ---
    parser.add_argument("--use_ema", action="store_true",
                        help="체크포인트에 ema_state_dict가 있으면 EMA 가중치로 추론")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    infer_size = (args.infer_h, args.infer_w)

    # 모델 + 체크포인트
    model = build_model("efficientnet", args.num_classes).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)

    # EMA 가중치 우선, 없으면 일반 가중치로 fallback
    if args.use_ema and isinstance(ckpt, dict) and "ema_state_dict" in ckpt:
        state_dict = ckpt["ema_state_dict"]
        print("EMA 가중치로 추론 (ema_state_dict)")
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        if args.use_ema:
            print("⚠️  --use_ema 지정했으나 ckpt에 ema_state_dict 없음. 일반 가중치로 fallback")
    else:
        state_dict = ckpt  # raw state_dict 저장된 경우
    model.load_state_dict(state_dict)
    model.eval()
    print(f"체크포인트 로드: {args.ckpt}")
    if isinstance(ckpt, dict) and "iter" in ckpt:
        print(f"  iter={ckpt['iter']}, best_val_top1={ckpt.get('best_val_top1', 0.0):.4f}")

    # 입출력 폴더
    img_dir = Path(args.img_dir)
    pred_dir = Path(args.pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)

    # 지원 확장자
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    img_paths = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in exts])

    if len(img_paths) == 0:
        print(f"⚠️  {img_dir} 에 이미지가 없습니다.")
        return

    print(f"총 {len(img_paths)}장 inference 시작 (입력 크기: {infer_size})")

    if args.tta:
        print(f"TTA 활성화 — scales={args.tta_scales}, flip={not args.no_tta_flip}")

    max_val = 0
    for i, img_path in enumerate(img_paths):
        image = Image.open(img_path).convert("RGB")
        pred = predict_one(
            model, image, device, infer_size,
            tta=args.tta,
            tta_scales=tuple(args.tta_scales),
            tta_flip=not args.no_tta_flip,
        )

        # 픽셀값 범위 검증용
        cur_max = int(pred.max())
        if cur_max > max_val:
            max_val = cur_max

        # 파일명: 기본은 원본 stem 유지, --rename_by_index면 000~999 강제
        if args.rename_by_index:
            out_name = f"{i:03d}.png"
        else:
            out_name = img_path.stem + ".png"
        out_path = pred_dir / out_name

        save_pred_png(pred, out_path)

        if (i + 1) % 50 == 0 or (i + 1) == len(img_paths):
            print(f"  [{i+1:4d}/{len(img_paths)}] {img_path.name} → {out_name}")

    # --- 최종 검증 ---
    saved = sorted(pred_dir.glob("*.png"))
    print(f"\n✅ 완료: {pred_dir} 에 {len(saved)}장 저장됨")
    print(f"   픽셀값 max = {max_val} (요구: [0, 20])")

    if len(saved) != 1000:
        print(f"⚠️  개수가 1000이 아님 ({len(saved)})")
    if max_val > 20:
        print(f"⚠️  픽셀값이 20을 초과함 ({max_val}) — num_classes 확인 필요")

    # 파일명 규칙 점검
    expected = {f"{i:03d}.png" for i in range(1000)}
    actual = {p.name for p in saved}
    missing = expected - actual
    extra = actual - expected
    if not missing and not extra:
        print("   파일명 규칙 000.png~999.png 일치 ✓")
    else:
        print(f"⚠️  파일명 규칙 불일치 — missing: {len(missing)}, extra: {len(extra)}")
        if missing:
            print(f"      누락 예시: {sorted(missing)[:5]}")
        if extra:
            print(f"      초과 예시: {sorted(extra)[:5]}")
        print("   → --rename_by_index 플래그를 써보세요.")


if __name__ == "__main__":
    main()
