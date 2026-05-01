import argparse
import yaml
import torch
import torch.nn as nn
from torch import Tensor

from data.pascal_voc import get_loader
from utils.model_factory import build_model


IGNORE_INDEX = 255

VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train",
    "tvmonitor",
]


@torch.no_grad()
def accumulate_confusion(
    model: nn.Module,
    loader,
    device: torch.device,
    num_classes: int,
) -> Tensor:
    """
    전체 dataset에 대해 confusion matrix를 누적.
    conf[i, j] = 정답이 i인데 j로 예측한 픽셀 수
    """
    model.eval()
    conf = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=device)

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        pred = logits.argmax(dim=1)

        mask = (labels != IGNORE_INDEX)
        pred_flat = pred[mask]
        targ_flat = labels[mask]

        idx = targ_flat * num_classes + pred_flat
        binc = torch.bincount(idx, minlength=num_classes * num_classes)
        conf += binc.reshape(num_classes, num_classes)

    return conf


def compute_metrics(conf: Tensor) -> dict:
    """누적된 confusion matrix에서 per-class IoU, mIoU, pixel accuracy 계산."""
    conf = conf.float()
    TP = conf.diag()
    FN = conf.sum(dim=1) - TP   # 정답인데 예측 못 함
    FP = conf.sum(dim=0) - TP   # 잘못 예측함

    # per-class IoU
    denom = TP + FP + FN
    iou = torch.where(denom > 0, TP / denom, torch.full_like(TP, float("nan")))

    # 등장한 클래스만 평균
    valid = denom > 0
    mIoU = iou[valid].mean().item()

    # pixel accuracy
    pixel_acc = (TP.sum() / conf.sum()).item() * 100.0

    return {
        "per_class_iou": iou.cpu().tolist(),
        "mIoU": mIoU,
        "pixel_acc": pixel_acc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/semantic_segmentation_efficientnet.yaml")
    parser.add_argument("--ckpt", type=str, required=True, help="체크포인트 경로 (.pth)")
    parser.add_argument("--split", type=str, default="val", choices=["val", "train"])
    parser.add_argument("--use_ema", action="store_true",
                        help="체크포인트에 ema_state_dict가 있으면 EMA 가중치로 평가")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    num_classes = cfg["model"]["num_classes"]

    # 데이터 로더
    loader = get_loader(
        root=cfg["data"]["root"],
        years=cfg["data"]["years"],
        image_set=args.split,
        crop_size=cfg["data"]["crop_size"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
    )

    # 모델 + 체크포인트
    model = build_model("efficientnet", num_classes).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)

    # EMA 가중치 사용 여부 결정
    if args.use_ema:
        if "ema_state_dict" in ckpt:
            model.load_state_dict(ckpt["ema_state_dict"])
            print("EMA 가중치로 평가 (ema_state_dict)")
        else:
            print("⚠️  --use_ema 지정했으나 ckpt에 ema_state_dict 없음. 일반 가중치로 fallback")
            model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt["model_state_dict"])

    print(f"체크포인트 로드 완료: iter {ckpt.get('iter', '?')}, "
          f"best_val_top1 {ckpt.get('best_val_top1', 0.0):.4f}")

    # 평가
    conf = accumulate_confusion(model, loader, device, num_classes)
    metrics = compute_metrics(conf)

    # 출력
    print("\n=== Per-class IoU ===")
    for i, (cls, iou) in enumerate(zip(VOC_CLASSES, metrics["per_class_iou"])):
        iou_str = f"{iou:.4f}" if iou == iou else "  N/A "  # NaN 체크
        print(f"  [{i:2d}] {cls:<15s} {iou_str}")

    print(f"\n=== Summary ({args.split}) ===")
    print(f"  Pixel Accuracy : {metrics['pixel_acc']:.2f}%")
    print(f"  mIoU           : {metrics['mIoU']:.4f}")


if __name__ == "__main__":
    main()
