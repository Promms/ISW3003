from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import wandb
import yaml

from data.coco_voc import CocoVOCSegDataset, get_combined_loader
from data.pascal_voc import build_voc_datasets, get_loader
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.losses import CEDiceLovaszLoss
from utils.ema import ModelEMA
from utils.metrics import (
    AverageMeter,
    accuracy,
    accuracy_counts,
    confusion_matrix,
    miou_from_confusion,
    per_class_iou_from_confusion,
)
from utils.optim import build_optimizer, poly_lr_step, set_backbone_requires_grad
from models.deeplabv3plus import deeplab_v3

VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train",
    "tvmonitor",
]

VOC_PALETTE = np.array([
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
    [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
    [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
    [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
    [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
    [0, 64, 128],
], dtype=np.uint8)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/semantic_segmentation.yaml")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    collect_images: bool = False,
    max_images: int = 4,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    conf = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=device)
    examples = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        n = images.size(0)
        loss_meter.update(loss.item(), n)
        acc_meter.update(accuracy(logits, labels), n)
        conf += confusion_matrix(logits, labels, num_classes)

        if collect_images and len(examples) < max_images:
            preds = logits.argmax(dim=1)
            take = min(max_images - len(examples), images.size(0))
            for idx in range(take):
                examples.append((
                    images[idx].detach().cpu(),
                    labels[idx].detach().cpu(),
                    preds[idx].detach().cpu(),
                ))

    if was_training:
        model.train()

    per_class_iou = per_class_iou_from_confusion(conf).detach().cpu().tolist()
    return {
        "loss": loss_meter.avg,
        "acc": acc_meter.avg,
        "mIoU": miou_from_confusion(conf).item(),
        "per_class_iou": per_class_iou,
        "examples": examples,
    }


def make_prediction_overlay(image: torch.Tensor, pred: torch.Tensor, alpha: float = 0.45) -> np.ndarray:
    image_np = image.permute(1, 2, 0).numpy()
    image_np = np.clip((image_np * IMAGENET_STD + IMAGENET_MEAN) * 255.0, 0, 255).astype(np.uint8)
    pred_np = pred.numpy()
    color_mask = VOC_PALETTE[np.clip(pred_np, 0, len(VOC_PALETTE) - 1)]
    return np.clip(image_np * (1.0 - alpha) + color_mask * alpha, 0, 255).astype(np.uint8)


def wandb_per_class(prefix: str, values: list[float]) -> dict[str, float]:
    logs = {}
    for name, value in zip(VOC_CLASSES, values):
        if value == value:
            logs[f"{prefix}/{name}"] = value
    return logs


def build_train_loader(cfg: dict):
    voc_train = build_voc_datasets(
        root=cfg["data"]["root"],
        years=cfg["data"]["years"],
        image_set="train",
        crop_size=cfg["data"]["crop_size"],
        augment=True,
        download=cfg["data"]["download"],
    )
    voc_total = sum(len(d) for d in voc_train)

    coco_cfg = cfg["data"].get("coco", {})
    coco_train = None
    if coco_cfg.get("enabled", False):
        coco_train = CocoVOCSegDataset(
            img_root=coco_cfg["img_root"],
            ann_file=coco_cfg["ann_file"],
            crop_size=cfg["data"]["crop_size"],
            augment=True,
            filter_empty=coco_cfg.get("filter_empty", True),
            overlap_policy=coco_cfg.get("overlap_policy", "smallest_first"),
            mask_cache_dir=coco_cfg.get("mask_cache_dir"),
        )
        print(f"[data] VOC {voc_total} + COCO {len(coco_train)} samples")
    else:
        print(f"[data] VOC only: {voc_total} samples")

    return get_combined_loader(
        voc_datasets=voc_train,
        coco_dataset=coco_train,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
        persistent_workers=cfg["data"].get("persistent_workers", False),
        prefetch_factor=cfg["data"].get("prefetch_factor", 2),
        shuffle=True,
        drop_last=True,
    )


def build_val_loader(cfg: dict):
    return get_loader(
        root=cfg["data"]["root"],
        years=cfg["data"]["years"],
        image_set=cfg["data"].get("val_split", "val"),
        crop_size=cfg["data"]["crop_size"],
        batch_size=cfg["training"].get("val_batch_size", cfg["training"]["batch_size"]),
        num_workers=cfg["data"].get("val_num_workers", 0),
        pin_memory=cfg["data"]["pin_memory"],
        persistent_workers=cfg["data"].get("persistent_workers", False),
        prefetch_factor=cfg["data"].get("prefetch_factor", 2),
    )


def find_wandb_run_id(paths: list[str]) -> str | None:
    for path in paths:
        if not os.path.exists(path):
            continue
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            print(f"[wandb] could not read {path}: {exc}")
            continue
        run_id = ckpt.get("wandb_run_id")
        if run_id:
            return run_id
    return None


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    run_name = cfg["wandb"]["run_name"]
    ckpt_dir = cfg["checkpoint"]["dir"]
    best_ema_path = os.path.join(ckpt_dir, f"best_ema_{run_name}.pth")
    last_path = os.path.join(ckpt_dir, f"last_{run_name}.pth")
    os.makedirs(ckpt_dir, exist_ok=True)

    wandb_run_id = None
    if cfg["training"].get("resume", False):
        wandb_run_id = find_wandb_run_id([last_path, best_ema_path])

    run = wandb.init(
        entity=cfg["wandb"]["team"],
        project=cfg["wandb"]["project"],
        name=run_name,
        config=cfg,
        id=wandb_run_id,
        resume="allow",
    )
    run.define_metric("*", step_metric="step")

    try:
        train_loader = build_train_loader(cfg)
        val_loader = build_val_loader(cfg)

        num_classes = cfg["model"]["num_classes"]
        model_cfg = cfg.get("model", {})
        model = deeplab_v3(
            num_classes=num_classes,
            aspp_channels=model_cfg.get("aspp_channels", 256),
            decoder_low_channels=model_cfg.get("decoder_low_channels", 48),
            pretrained_backbone=model_cfg.get("pretrained_backbone", True),
        ).to(device)
        print(
            "Backbone: mobilenet_v3_large | "
            f"ASPP/decoder channels: {model_cfg.get('aspp_channels', 256)}"
        )
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

        freeze_iters = cfg["training"].get("freeze_iters", 0)
        if freeze_iters > 0:
            set_backbone_requires_grad(model, False)
            print(f"Backbone frozen for first {freeze_iters} iterations")

        criterion = CEDiceLovaszLoss(
            num_classes=num_classes,
            ignore_index=255,
            dice_weight=cfg["training"].get("dice_weight", 1.0),
            lovasz_weight=cfg["training"].get("lovasz_weight", 0.5),
        )
        optimizer = build_optimizer(model, cfg)

        use_amp = cfg["training"].get("amp", False) and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        ema = ModelEMA(model, decay=cfg["training"].get("ema_decay", 0.9998))

        iter_count = 0
        best_ema_miou = 0.0
        if cfg["training"].get("resume", False):
            resume_path = last_path if os.path.exists(last_path) else best_ema_path
            if os.path.exists(resume_path):
                meta = load_checkpoint(resume_path, model, optimizer, scaler, ema, device)
                iter_count = meta["iter"]
                best_ema_miou = meta["best_ema_miou"]
                print(f"Resumed {os.path.basename(resume_path)} at iter {iter_count}")
            else:
                print("resume=true but no checkpoint was found; starting fresh")

        total_iters = cfg["training"]["total_iters"]
        log_interval = cfg["training"]["log_interval"]
        eval_interval = cfg["training"]["eval_interval"]
        ema_update_every = cfg["training"].get("ema_update_every", 1)
        grad_clip_norm = cfg["training"].get("grad_clip_norm", 0.0)

        loss_accum = torch.zeros((), device=device)
        correct_accum = torch.zeros((), device=device, dtype=torch.long)
        total_accum = torch.zeros((), device=device, dtype=torch.long)
        data_iter = iter(train_loader)
        model.train()

        while iter_count < total_iters:
            if freeze_iters > 0 and iter_count == freeze_iters:
                set_backbone_requires_grad(model, True)
                print(f"[iter {iter_count}] backbone unfrozen")

            try:
                images, labels = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                images, labels = next(data_iter)

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            if (iter_count + 1) % ema_update_every == 0:
                ema.update(model)

            poly_lr_step(
                optimizer,
                iter_count,
                total_iters,
                warmup_iters=cfg["training"].get("warmup_iters", 0),
            )

            loss_accum += loss.detach()
            correct, total = accuracy_counts(logits, labels)
            correct_accum += correct
            total_accum += total
            iter_count += 1

            if iter_count % log_interval == 0:
                avg_loss = (loss_accum / log_interval).item()
                avg_acc = (
                    correct_accum.float() / total_accum.clamp(min=1).float() * 100.0
                ).item()
                frozen = freeze_iters > 0 and iter_count <= freeze_iters
                lr_backbone = 0.0 if frozen else optimizer.param_groups[0]["lr"]
                lr_head = optimizer.param_groups[2]["lr"]

                run.log({
                    "step": iter_count,
                    "train/loss": avg_loss,
                    "train/acc": avg_acc,
                    "lr/backbone": lr_backbone,
                    "lr/head": lr_head,
                })
                print(
                    f"iter {iter_count:6d}/{total_iters} | "
                    f"loss {avg_loss:.4f} | acc {avg_acc:.2f}% | "
                    f"lr(bb) {lr_backbone:.2e} | lr(head) {lr_head:.2e}"
                )
                loss_accum.zero_()
                correct_accum.zero_()
                total_accum.zero_()

            if iter_count % eval_interval == 0:
                raw_metrics = evaluate(model, val_loader, criterion, device, num_classes)
                ema_metrics = evaluate(
                    ema.ema_model,
                    val_loader,
                    criterion,
                    device,
                    num_classes,
                    collect_images=True,
                )
                ema_miou = ema_metrics["mIoU"]

                log_payload = {
                    "step": iter_count,
                    "val/loss": raw_metrics["loss"],
                    "val/acc": raw_metrics["acc"],
                    "val/mIoU": raw_metrics["mIoU"],
                    "val/ema_loss": ema_metrics["loss"],
                    "val/ema_acc": ema_metrics["acc"],
                    "val/ema_mIoU": ema_miou,
                    "best/ema_mIoU": max(best_ema_miou, ema_miou),
                }
                log_payload.update(wandb_per_class("val_iou", raw_metrics["per_class_iou"]))
                log_payload.update(wandb_per_class("val_ema_iou", ema_metrics["per_class_iou"]))

                if ema_metrics["examples"]:
                    log_payload["val/ema_prediction_overlay"] = [
                        wandb.Image(
                            make_prediction_overlay(image, pred),
                            caption=f"sample_{idx}",
                        )
                        for idx, (image, _label, pred) in enumerate(ema_metrics["examples"])
                    ]

                run.log(log_payload)

                if ema_miou > best_ema_miou:
                    best_ema_miou = ema_miou
                    save_checkpoint(
                        best_ema_path,
                        iter_count,
                        model,
                        optimizer,
                        scaler=scaler,
                        ema=ema,
                        cfg=cfg,
                        wandb_run_id=run.id,
                        best_ema_miou=best_ema_miou,
                    )
                    print(f"  saved best EMA checkpoint: mIoU {best_ema_miou:.4f}")

                save_checkpoint(
                    last_path,
                    iter_count,
                    model,
                    optimizer,
                    scaler=scaler,
                    ema=ema,
                    cfg=cfg,
                    wandb_run_id=run.id,
                    best_ema_miou=best_ema_miou,
                )
                print(
                    f"  val mIoU {raw_metrics['mIoU']:.4f} | "
                    f"EMA mIoU {ema_miou:.4f} | best EMA {best_ema_miou:.4f}"
                )
    finally:
        run.finish()


if __name__ == "__main__":
    main()
