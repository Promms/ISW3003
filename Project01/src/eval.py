from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import yaml
from torch import Tensor

from data.pascal_voc import get_loader
from models.deeplabv3plus import deeplab_v3


IGNORE_INDEX = 255

VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train",
    "tvmonitor",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/semantic_segmentation.yaml")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--use_ema", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def accumulate_confusion(
    model: nn.Module,
    loader,
    device: torch.device,
    num_classes: int,
) -> Tensor:
    model.eval()
    conf = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=device)

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        pred = model(images).argmax(dim=1)

        valid = labels != IGNORE_INDEX
        idx = labels[valid] * num_classes + pred[valid]
        binc = torch.bincount(idx, minlength=num_classes * num_classes)
        conf += binc.reshape(num_classes, num_classes)

    return conf


def compute_metrics(conf: Tensor) -> dict:
    conf = conf.float()
    tp = conf.diag()
    fp = conf.sum(dim=0) - tp
    fn = conf.sum(dim=1) - tp
    denom = tp + fp + fn
    iou = torch.where(denom > 0, tp / denom, torch.full_like(tp, float("nan")))
    valid = denom > 0
    pixel_acc = (tp.sum() / conf.sum().clamp_min(1)).item() * 100.0
    return {
        "per_class_iou": iou.cpu().tolist(),
        "mIoU": iou[valid].mean().item(),
        "pixel_acc": pixel_acc,
    }


def main() -> None:
    args = parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    num_classes = cfg["model"]["num_classes"]

    loader = get_loader(
        root=cfg["data"]["root"],
        years=cfg["data"]["years"],
        image_set=args.split,
        crop_size=cfg["data"]["crop_size"],
        batch_size=cfg["training"].get("val_batch_size", cfg["training"]["batch_size"]),
        num_workers=cfg["data"].get("val_num_workers", 0),
        pin_memory=cfg["data"]["pin_memory"],
        preload=False,
    )

    model_cfg = cfg.get("model", {})
    model = deeplab_v3(
        num_classes=num_classes,
        aspp_channels=model_cfg.get("aspp_channels", 256),
        decoder_low_channels=model_cfg.get("decoder_low_channels", 48),
        pretrained_backbone=False,
    ).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    if args.use_ema and "ema_state_dict" in ckpt:
        model.load_state_dict(ckpt["ema_state_dict"])
        weight_name = "ema_state_dict"
    else:
        model.load_state_dict(ckpt["model_state_dict"])
        weight_name = "model_state_dict"

    print(f"Loaded {args.ckpt} ({weight_name})")
    print(f"iter={ckpt.get('iter', '?')} best_ema_mIoU={ckpt.get('best_ema_miou', 0.0):.4f}")

    metrics = compute_metrics(accumulate_confusion(model, loader, device, num_classes))

    print("\n=== Per-class IoU ===")
    for idx, (name, iou) in enumerate(zip(VOC_CLASSES, metrics["per_class_iou"])):
        iou_text = f"{iou:.4f}" if iou == iou else "N/A"
        print(f"  [{idx:2d}] {name:<15s} {iou_text}")

    print(f"\n=== Summary ({args.split}) ===")
    print(f"  Pixel Accuracy : {metrics['pixel_acc']:.2f}%")
    print(f"  mIoU           : {metrics['mIoU']:.4f}")


if __name__ == "__main__":
    main()
