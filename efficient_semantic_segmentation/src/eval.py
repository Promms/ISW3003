from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import yaml
from torch import Tensor

from models.deeplabv3plus import deeplab_v3_efficientnet


IGNORE_INDEX = 255
VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train",
    "tvmonitor",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on Pascal VOC.")
    parser.add_argument("--config", type=str, default="src/training_config.yaml")
    parser.add_argument("--ckpt", type=str, default="checkpoints/model.pth")
    parser.add_argument("--split", type=str, default="val", choices=["val", "train"])
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

        # Flatten valid pixels into a compact confusion-matrix index.
        valid = labels != IGNORE_INDEX
        idx = labels[valid] * num_classes + pred[valid]
        conf += torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)

    return conf


def compute_metrics(conf: Tensor) -> dict[str, object]:
    conf = conf.float()
    true_positive = conf.diag()
    false_negative = conf.sum(dim=1) - true_positive
    false_positive = conf.sum(dim=0) - true_positive

    denom = true_positive + false_positive + false_negative
    iou = torch.where(denom > 0, true_positive / denom, torch.full_like(denom, float("nan")))
    valid = denom > 0
    pixel_acc = (true_positive.sum() / conf.sum()).item() * 100.0

    return {
        "per_class_iou": iou.cpu().tolist(),
        "mIoU": iou[valid].mean().item(),
        "pixel_acc": pixel_acc,
    }


def load_state_dict(path: str, device: torch.device, use_ema: bool) -> dict:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and use_ema and "ema_state_dict" in ckpt:
        print("Evaluating EMA weights.")
        return ckpt["ema_state_dict"]
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        if use_ema:
            print("EMA weights were requested but not found; using model_state_dict.")
        return ckpt["model_state_dict"]
    return ckpt


def main() -> None:
    args = parse_args()
    from data.voc_dataset import get_loader

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
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
        preload=False,
    )

    model = deeplab_v3_efficientnet(num_classes, pretrained=False).to(device)
    model.load_state_dict(load_state_dict(args.ckpt, device, args.use_ema))

    metrics = compute_metrics(accumulate_confusion(model, loader, device, num_classes))

    print("\n=== Per-class IoU ===")
    for idx, (class_name, iou) in enumerate(zip(VOC_CLASSES, metrics["per_class_iou"])):
        iou_text = f"{iou:.4f}" if iou == iou else "N/A"
        print(f"[{idx:2d}] {class_name:<15s} {iou_text}")

    print(f"\n=== Summary ({args.split}) ===")
    print(f"Pixel Accuracy : {metrics['pixel_acc']:.2f}%")
    print(f"mIoU           : {metrics['mIoU']:.4f}")


if __name__ == "__main__":
    main()
