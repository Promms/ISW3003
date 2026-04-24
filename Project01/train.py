"""
통합 학습 스크립트.

백본 선택:
    yaml의 model.backbone = "mobilenet" | "efficientnet"

사용법:
    python train.py --config src/semantic_segmentation.yaml
    python train.py --config src/semantic_segmentation_efficientnet.yaml

핵심 기능:
    - Differential LR (backbone < head) + Poly LR decay
    - Backbone freeze → unfreeze 2-stage
    - Mixed Precision (AMP)
    - torch.compile (1.3~1.5× 가속)
    - EMA (Exponential Moving Average) shadow model
    - WandB 로깅 + best ckpt 자동 저장
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn
import wandb
import yaml

from data.pascal_voc import get_loader
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.diceloss import CEDiceLoss
from utils.ema import ModelEMA
from utils.metrics import AverageMeter, accuracy, mIoU
from utils.model_factory import build_model
from utils.optim import build_optimizer, poly_lr_step, set_backbone_requires_grad


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/semantic_segmentation.yaml",
                        help="YAML 설정 파일 경로")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion: nn.Module, device: torch.device,
             num_classes: int) -> dict[str, float]:
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    mIoU_meter = AverageMeter()

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        n = images.size(0)
        loss_meter.update(loss.item(), n)
        acc_meter.update(accuracy(logits, labels), n)
        mIoU_meter.update(mIoU(logits, labels, num_classes), n)

    model.train()
    return {"loss": loss_meter.avg, "acc": acc_meter.avg, "top1/mIoU": mIoU_meter.avg}


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["seed"])

    run = wandb.init(
        entity=cfg["wandb"]["team"],
        project=cfg["wandb"]["project"],
        name=cfg["wandb"]["run_name"],
        config=cfg,
    )
    run.define_metric("*", step_metric="step")

    try:
        # --- 데이터 ---
        train_loader = get_loader(
            root=cfg["data"]["root"],
            years=cfg["data"]["years"],
            image_set="train",
            crop_size=cfg["data"]["crop_size"],
            batch_size=cfg["training"]["batch_size"],
            num_workers=cfg["data"]["num_workers"],
            pin_memory=cfg["data"]["pin_memory"],
            download=cfg["data"]["download"],
            preload=cfg["data"].get("preload", True),
        )
        # val: preload=False + num_workers=0 (train preload 캐시 COW 복사 폭탄 방지)
        val_loader = get_loader(
            root=cfg["data"]["root"],
            years=cfg["data"]["years"],
            image_set="val",
            crop_size=cfg["data"]["crop_size"],
            batch_size=cfg["training"]["batch_size"],
            num_workers=0,
            pin_memory=cfg["data"]["pin_memory"],
            preload=False,
        )

        # --- 모델 (백본 선택은 yaml.model.backbone) ---
        backbone = cfg["model"]["backbone"]
        num_classes = cfg["model"]["num_classes"]
        model = build_model(backbone, num_classes).to(device)
        print(f"Backbone: {backbone}")

        # --- Freeze (Stage 1): compile보다 먼저 설정 ---
        freeze_iters = cfg["training"].get("freeze_iters", 0)
        if freeze_iters > 0:
            set_backbone_requires_grad(model, False)
            print(f"Backbone freeze: 0 ~ {freeze_iters} iter")

        # --- torch.compile ---
        if cfg["training"].get("compile", True):
            try:
                model = torch.compile(model, mode="reduce-overhead")
                print("torch.compile 적용 (mode=reduce-overhead)")
            except Exception as e:
                print(f"torch.compile 실패 — 일반 모델로 진행: {e}")

        # --- Loss / Optimizer ---
        criterion = CEDiceLoss(num_classes=num_classes, ignore_index=255, dice_weight=1.0)
        optimizer = build_optimizer(model, cfg)

        total_iters      = cfg["training"]["total_iters"]
        log_interval     = cfg["training"]["log_interval"]
        eval_interval    = cfg["training"]["eval_interval"]

        # --- AMP ---
        use_amp = cfg["training"].get("amp", False) and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        if use_amp:
            print("Mixed Precision (FP16 autocast) 활성화")

        # --- EMA ---
        use_ema = cfg["training"].get("ema", False)
        ema_decay = cfg["training"].get("ema_decay", 0.9998)
        ema_eval_interval = cfg["training"].get("ema_eval_interval", 10000)
        ema = None
        if use_ema:
            ema = ModelEMA(model, decay=ema_decay)
            print(f"EMA 활성화 (decay={ema_decay}, eval every {ema_eval_interval} iter)")

        # --- 체크포인트 경로 ---
        ckpt_dir = cfg["checkpoint"]["dir"]
        ckpt_name = f"best_{cfg['wandb']['run_name']}.pth"
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"체크포인트 경로: {ckpt_path}")

        # --- Resume ---
        iter_count = 0
        best_val_top1 = 0.0
        if cfg["training"]["resume"] and os.path.exists(ckpt_path):
            meta = load_checkpoint(ckpt_path, model, optimizer,
                                   scaler if use_amp else None, ema, device)
            iter_count = meta["iter"]
            best_val_top1 = meta["best_val_top1"]
            print(f"체크포인트 불러옴: iter {iter_count}, best mIoU {best_val_top1:.4f}"
                  + ("  [EMA 포함]" if meta["has_ema"] else ""))
        else:
            print("처음부터 학습 시작")

        # --- 학습 루프 ---
        loss_meter = AverageMeter()
        acc_meter  = AverageMeter()
        data_iter = iter(train_loader)
        model.train()

        while iter_count < total_iters:
            # Unfreeze (Stage 2)
            if freeze_iters > 0 and iter_count == freeze_iters:
                unfreeze_target = model._orig_mod if hasattr(model, "_orig_mod") else model
                set_backbone_requires_grad(unfreeze_target, True)
                print(f"[Iter {iter_count}] Backbone unfreeze")

            try:
                images, labels = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                images, labels = next(data_iter)

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if ema is not None:
                ema.update(model)

            poly_lr_step(optimizer, iter_count, total_iters)

            n = images.size(0)
            loss_meter.update(loss.item(), n)
            acc_meter.update(accuracy(logits, labels), n)

            iter_count += 1

            # --- Logging ---
            if iter_count % log_interval == 0:
                is_frozen = (freeze_iters > 0 and iter_count <= freeze_iters)
                lr_backbone = 0.0 if is_frozen else optimizer.param_groups[0]["lr"]
                lr_head     = optimizer.param_groups[2]["lr"]
                run.log({
                    "step":        iter_count,
                    "train/loss":  loss_meter.avg,
                    "train/acc":   acc_meter.avg,
                    "lr/backbone": lr_backbone,
                    "lr/head":     lr_head,
                })
                print(
                    f"Iter {iter_count:6d}/{total_iters} | "
                    f"Loss: {loss_meter.avg:.4f} | "
                    f"Accuracy: {acc_meter.avg:.2f}% | "
                    f"LR(bb): {lr_backbone:.2e} | LR(head): {lr_head:.2e}"
                    + ("  [frozen]" if is_frozen else "")
                )
                loss_meter.reset()
                acc_meter.reset()

            # --- Eval + best ckpt ---
            if iter_count % eval_interval == 0:
                val_metrics = evaluate(model, val_loader, criterion, device, num_classes)
                log_payload = {
                    "step":          iter_count,
                    "val/loss":      val_metrics["loss"],
                    "val/acc":       val_metrics["acc"],
                    "val/top1_mIoU": val_metrics["top1/mIoU"],
                }

                # EMA eval (ema_eval_interval 주기만)
                ema_mIoU = None
                if ema is not None and iter_count % ema_eval_interval == 0:
                    ema_metrics = evaluate(ema.ema_model, val_loader, criterion, device, num_classes)
                    ema_mIoU = ema_metrics["top1/mIoU"]
                    log_payload["val/ema_loss"] = ema_metrics["loss"]
                    log_payload["val/ema_top1_mIoU"] = ema_mIoU
                    print(f"  [EMA Val] Loss: {ema_metrics['loss']:.4f} | mIoU: {ema_mIoU:.4f}")

                run.log(log_payload)

                current_best = max(val_metrics["top1/mIoU"], ema_mIoU) if ema_mIoU is not None \
                               else val_metrics["top1/mIoU"]
                is_best = current_best > best_val_top1
                if is_best:
                    best_val_top1 = current_best
                    save_checkpoint(
                        ckpt_path, iter_count, model, optimizer,
                        scaler=scaler, ema=ema,
                        best_val_top1=best_val_top1, cfg=cfg,
                    )
                print(
                    f"  [Val] Loss: {val_metrics['loss']:.4f} | "
                    f"mIoU: {val_metrics['top1/mIoU']:.4f}"
                    + ("  [best]" if is_best else "")
                )
    finally:
        run.finish()


if __name__ == "__main__":
    main()
