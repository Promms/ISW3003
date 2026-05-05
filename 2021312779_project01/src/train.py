from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import yaml

from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.losses import CEDiceLovaszLoss
from utils.ema import ModelEMA
from utils.metrics import AverageMeter, accuracy, accuracy_counts, mIoU
from models.deeplabv3plus import deeplab_v3_efficientnet
from utils.optimizer import build_optimizer, poly_lr_step, set_backbone_requires_grad
from utils.parameters import log_parameter_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the EfficientNet-B1 DeepLabV3+ model.")
    parser.add_argument("--config", type=str, default="src/training_config.yaml")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_train_loader(cfg: dict):
    from data.coco_voc_dataset import CocoVOCSegDataset, get_combined_loader
    from data.voc_dataset import build_voc_datasets

    voc_train = build_voc_datasets(
        root=cfg["data"]["root"],
        years=cfg["data"]["years"],
        image_set="train",
        crop_size=cfg["data"]["crop_size"],
        augment=True,
        download=cfg["data"]["download"],
        preload=cfg["data"].get("preload", True),
    )
    voc_total = sum(len(dataset) for dataset in voc_train)

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
        print(f"[data] VOC {voc_total} + COCO {len(coco_train)} train samples")
    else:
        print(f"[data] VOC {voc_total} train samples")

    return get_combined_loader(
        voc_datasets=voc_train,
        coco_dataset=coco_train,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
        shuffle=True,
        drop_last=True,
    )


def build_val_loader(cfg: dict):
    from data.voc_dataset import get_loader

    return get_loader(
        root=cfg["data"]["root"],
        years=cfg["data"]["years"],
        image_set="val",
        crop_size=cfg["data"]["crop_size"],
        batch_size=cfg["training"].get("val_batch_size", cfg["training"]["batch_size"]),
        num_workers=0,
        pin_memory=cfg["data"]["pin_memory"],
        preload=False,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    miou_meter = AverageMeter()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        batch_size = images.size(0)

        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(accuracy(logits, labels), batch_size)
        miou_meter.update(mIoU(logits, labels, num_classes), batch_size)

    if was_training:
        model.train()

    return {"loss": loss_meter.avg, "acc": acc_meter.avg, "mIoU": miou_meter.avg}


def find_wandb_run_id(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"[wandb] Could not read run id from {path}: {exc}")
        return None
    run_id = ckpt.get("wandb_run_id") if isinstance(ckpt, dict) else None
    if run_id:
        return run_id
    return None


def checkpoint_path(cfg: dict) -> str:
    return cfg["checkpoint"].get(
        "path",
        os.path.join(cfg["checkpoint"].get("dir", "checkpoints"), "model.pth"),
    )


def maybe_resume_or_finetune(
    cfg: dict,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    ema: ModelEMA | None,
    device: torch.device,
    submit_ckpt_path: str,
) -> tuple[int, float, float]:
    if cfg["training"]["resume"]:
        if os.path.exists(submit_ckpt_path):
            meta = load_checkpoint(submit_ckpt_path, model, optimizer, scaler, ema, device)
            print(
                f"Resumed {submit_ckpt_path} at iter {meta['iter']} "
                f"(best raw={meta['best_raw_miou']:.4f}, best ema={meta['best_ema_miou']:.4f})"
            )
            return meta["iter"], meta["best_raw_miou"], meta["best_ema_miou"]
        print(f"resume=true but {submit_ckpt_path} was not found; training from scratch.")
        return 0, 0.0, 0.0

    finetune_from = cfg["training"].get("finetune_from")
    if finetune_from and os.path.exists(finetune_from):
        ckpt = torch.load(finetune_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if ema is not None and ckpt.get("ema_state_dict") is not None:
            ema.ema_model.load_state_dict(ckpt["ema_state_dict"])
        print(f"Fine-tuning from {finetune_from}; optimizer state is reset.")
        return ckpt.get("iter", 0), 0.0, 0.0

    if finetune_from:
        print(f"finetune_from was set but not found: {finetune_from}")
    return 0, 0.0, 0.0


def main() -> None:
    args = parse_args()
    import wandb

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    submit_ckpt_path = checkpoint_path(cfg)
    os.makedirs(os.path.dirname(submit_ckpt_path), exist_ok=True)

    wandb_run_id = None
    if cfg["training"]["resume"]:
        wandb_run_id = find_wandb_run_id(submit_ckpt_path)

    run = wandb.init(
        entity=cfg["wandb"]["team"],
        project=cfg["wandb"]["project"],
        name=cfg["wandb"]["run_name"],
        config=cfg,
        id=wandb_run_id,
        resume="allow",
    )
    run.define_metric("*", step_metric="step")

    try:
        train_loader = build_train_loader(cfg)
        val_loader = build_val_loader(cfg)

        num_classes = cfg["model"]["num_classes"]
        model = deeplab_v3_efficientnet(
            num_classes,
            pretrained=cfg["model"].get("pretrained", True),
        ).to(device)
        print("Backbone: efficientnet_b1")
        log_parameter_counts(model)

        freeze_iters = cfg["training"].get("freeze_iters", 0)
        if freeze_iters > 0:
            set_backbone_requires_grad(model, False)
            print(f"Backbone frozen until iter {freeze_iters}.")

        criterion = CEDiceLovaszLoss(
            num_classes=num_classes,
            ignore_index=255,
            dice_weight=1.0,
            lovasz_weight=cfg["training"].get("lovasz_weight", 0.5),
        )
        optimizer = build_optimizer(model, cfg)

        use_amp = cfg["training"].get("amp", False) and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        ema = None
        if cfg["training"].get("ema", False):
            ema = ModelEMA(model, decay=cfg["training"].get("ema_decay", 0.9998))

        iter_count, best_raw_miou, best_ema_miou = maybe_resume_or_finetune(
            cfg,
            model,
            optimizer,
            scaler if use_amp else None,
            ema,
            device,
            submit_ckpt_path,
        )
        best_submit_miou = max(best_raw_miou, best_ema_miou)

        total_iters = cfg["training"]["total_iters"]
        log_interval = cfg["training"]["log_interval"]
        eval_interval = cfg["training"]["eval_interval"]
        ema_eval_interval = cfg["training"].get("ema_eval_interval", 10000)
        ema_update_every = cfg["training"].get("ema_update_every", 1)

        loss_accum = torch.zeros((), device=device)
        correct_accum = torch.zeros((), device=device, dtype=torch.long)
        total_accum = torch.zeros((), device=device, dtype=torch.long)
        data_iter = iter(train_loader)
        model.train()

        while iter_count < total_iters:
            if freeze_iters > 0 and iter_count == freeze_iters:
                set_backbone_requires_grad(model, True)
                print(f"[iter {iter_count}] Backbone unfrozen.")

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
            scaler.step(optimizer)
            scaler.update()

            if ema is not None and (iter_count + 1) % ema_update_every == 0:
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
                avg_acc = (correct_accum.float() / total_accum.clamp(min=1).float() * 100.0).item()
                lr_backbone = optimizer.param_groups[0]["lr"]
                lr_head = optimizer.param_groups[2]["lr"]

                run.log({
                    "step": iter_count,
                    "train/loss": avg_loss,
                    "train/acc": avg_acc,
                    "lr/backbone": lr_backbone,
                    "lr/head": lr_head,
                })
                print(
                    f"iter {iter_count:6d}/{total_iters} | loss {avg_loss:.4f} | "
                    f"acc {avg_acc:.2f}% | lr_backbone {lr_backbone:.2e} | lr_head {lr_head:.2e}"
                )
                loss_accum.zero_()
                correct_accum.zero_()
                total_accum.zero_()

            if iter_count % eval_interval == 0:
                val_metrics = evaluate(model, val_loader, criterion, device, num_classes)
                raw_miou = val_metrics["mIoU"]
                log_payload = {
                    "step": iter_count,
                    "val/loss": val_metrics["loss"],
                    "val/acc": val_metrics["acc"],
                    "val/mIoU": raw_miou,
                }

                ema_miou = None
                if ema is not None and iter_count % ema_eval_interval == 0:
                    ema_metrics = evaluate(ema.ema_model, val_loader, criterion, device, num_classes)
                    ema_miou = ema_metrics["mIoU"]
                    log_payload["val/ema_loss"] = ema_metrics["loss"]
                    log_payload["val/ema_mIoU"] = ema_miou

                run.log(log_payload)

                if raw_miou > best_raw_miou:
                    best_raw_miou = raw_miou

                if ema_miou is not None and ema_miou > best_ema_miou:
                    best_ema_miou = ema_miou

                # Submit the best validation checkpoint among raw and EMA weights.
                current_submit_miou = max(raw_miou, ema_miou if ema_miou is not None else -1.0)
                if current_submit_miou > best_submit_miou:
                    best_submit_miou = current_submit_miou
                    save_checkpoint(
                        submit_ckpt_path,
                        iter_count,
                        model,
                        optimizer,
                        scaler=scaler,
                        ema=ema,
                        cfg=cfg,
                        wandb_run_id=run.id,
                        best_raw_miou=best_raw_miou,
                        best_ema_miou=best_ema_miou,
                    )
                    print(f"[checkpoint] saved {submit_ckpt_path} (best submit mIoU={best_submit_miou:.4f})")
                print(f"[val] loss {val_metrics['loss']:.4f} | mIoU {raw_miou:.4f}")
    finally:
        run.finish()


if __name__ == "__main__":
    main()
