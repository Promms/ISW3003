"""
EfficientNet-B1 전용 학습 스크립트 (V7).

V5_2(B0) 대비 변경:
    - 백본 efficientnet_b0 → efficientnet_b1 (width 동일, depth +10%)
    - 그 외 셋팅(loss, aug, lr, freeze 등) V5_2와 동일

사용법:
    python train.py --config src/semantic_segmentation_efficientnet.yaml

핵심 기능:
    - Differential LR (backbone < head) + Poly LR decay
    - Backbone freeze → unfreeze 2-stage
    - Mixed Precision (AMP)
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

from data.coco_voc import CocoVOCSegDataset, get_combined_loader
from data.pascal_voc import build_voc_datasets, get_loader
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.diceloss import CEDiceLoss
from utils.ema import ModelEMA
from utils.metrics import AverageMeter, accuracy, accuracy_counts, mIoU
from utils.model_factory import build_model
from utils.optim import build_optimizer, poly_lr_step, set_backbone_requires_grad


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/semantic_segmentation_efficientnet.yaml",
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

    # --- WandB resume: ckpt에서 run_id 미리 뽑아 같은 run에 이어 기록 ---
    ckpt_dir = cfg["checkpoint"]["dir"]
    run_name = cfg["wandb"]["run_name"]
    ckpt_path = os.path.join(ckpt_dir, f"best_{run_name}.pth")
    last_ckpt_path = os.path.join(ckpt_dir, f"last_{run_name}.pth")

    wandb_run_id = None
    if cfg["training"]["resume"]:
        for p in (last_ckpt_path, ckpt_path):
            if os.path.exists(p):
                try:
                    meta = torch.load(p, map_location="cpu", weights_only=False)
                    wandb_run_id = meta.get("wandb_run_id")
                except Exception as e:
                    print(f"[wandb] ckpt에서 run_id 읽기 실패: {e}")
                if wandb_run_id:
                    print(f"[wandb] resume run_id={wandb_run_id}")
                    break

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
        # --- Train 데이터: VOC + COCO(선택) 합류 ---
        voc_train = build_voc_datasets(
            root=cfg["data"]["root"],
            years=cfg["data"]["years"],
            image_set="train",
            crop_size=cfg["data"]["crop_size"],
            augment=True,
            download=cfg["data"]["download"],
            preload=cfg["data"].get("preload", True),
        )
        voc_total = sum(len(d) for d in voc_train)

        coco_cfg = cfg["data"].get("coco")
        coco_train = None
        if coco_cfg and coco_cfg.get("enabled", True):
            coco_train = CocoVOCSegDataset(
                img_root=coco_cfg["img_root"],
                ann_file=coco_cfg["ann_file"],
                crop_size=cfg["data"]["crop_size"],
                augment=True,
                filter_empty=coco_cfg.get("filter_empty", True),
                overlap_policy=coco_cfg.get("overlap_policy", "smallest_first"),
                mask_cache_dir=coco_cfg.get("mask_cache_dir"),
            )
            print(f"[data] VOC {voc_total} + COCO {len(coco_train)} "
                  f"= {voc_total + len(coco_train)} train samples")
        else:
            print(f"[data] VOC only: {voc_total} train samples")

        train_loader = get_combined_loader(
            voc_datasets=voc_train,
            coco_dataset=coco_train,
            batch_size=cfg["training"]["batch_size"],
            num_workers=cfg["data"]["num_workers"],
            pin_memory=cfg["data"]["pin_memory"],
            shuffle=True,
            drop_last=True,
        )

        # --- Val: VOC val만 사용 (지표 일관성. 최종 평가는 별도 custom set) ---
        # preload=False + num_workers=0 (train preload 캐시 COW 복사 폭탄 방지)
        val_loader = get_loader(
            root=cfg["data"]["root"],
            years=cfg["data"]["years"],
            image_set="val",
            crop_size=cfg["data"]["crop_size"],
            batch_size=cfg["training"].get("val_batch_size", cfg["training"]["batch_size"]),
            num_workers=0,
            pin_memory=cfg["data"]["pin_memory"],
            preload=False,
        )

        # --- 모델 (백본 선택은 yaml.model.backbone) ---
        backbone = cfg["model"]["backbone"]
        num_classes = cfg["model"]["num_classes"]
        model = build_model(backbone, num_classes).to(device)
        print(f"Backbone: {backbone}")

        # --- Freeze (Stage 1) ---
        freeze_iters = cfg["training"].get("freeze_iters", 0)
        if freeze_iters > 0:
            set_backbone_requires_grad(model, False)
            print(f"Backbone freeze: 0 ~ {freeze_iters} iter")

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
        ema_update_every = cfg["training"].get("ema_update_every", 1)
        ema = None
        if use_ema:
            ema = ModelEMA(model, decay=ema_decay)
            print(f"EMA 활성화 (decay={ema_decay}, eval every {ema_eval_interval} iter)")

        # --- 체크포인트 경로 (위에서 이미 계산됨) ---
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"체크포인트 경로: {ckpt_path}")
        print(f"Latest 체크포인트: {last_ckpt_path}")

        # --- Resume ---
        # latest 우선 (crash 직후 가장 최근 state). 없으면 best로 폴백.
        iter_count = 0
        best_val_top1 = 0.0
        if cfg["training"]["resume"]:
            resume_path = last_ckpt_path if os.path.exists(last_ckpt_path) else \
                          (ckpt_path if os.path.exists(ckpt_path) else None)
            if resume_path is not None:
                meta = load_checkpoint(resume_path, model, optimizer,
                                       scaler if use_amp else None, ema, device)
                iter_count = meta["iter"]
                best_val_top1 = meta["best_val_top1"]
                print(f"체크포인트 불러옴: {os.path.basename(resume_path)} | "
                      f"iter {iter_count}, best mIoU {best_val_top1:.4f}"
                      + ("  [EMA 포함]" if meta["has_ema"] else ""))
            else:
                print("resume=true지만 ckpt 없음 → 처음부터 학습")
        else:
            print("처음부터 학습 시작")

        # --- 학습 루프 ---
        # loss/acc는 매 iter .item() 하지 않고 GPU 텐서로 누적 → log_interval 마다만 sync
        # (GPU sync 1/log_interval 로 감소해서 CUDA 파이프라인이 막히지 않음)
        loss_accum  = torch.zeros((), device=device)
        correct_accum = torch.zeros((), device=device, dtype=torch.long)
        total_accum = torch.zeros((), device=device, dtype=torch.long)
        data_iter = iter(train_loader)
        model.train()

        while iter_count < total_iters:
            # Unfreeze (Stage 2)
            if freeze_iters > 0 and iter_count == freeze_iters:
                set_backbone_requires_grad(model, True)
                print(f"[Iter {iter_count}] Backbone unfreeze")

            try:
                images, labels = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                images, labels = next(data_iter)

            # non_blocking: pin_memory 와 함께 H2D 전송을 다음 compute와 오버랩
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
                optimizer, iter_count, total_iters,
                warmup_iters=cfg["training"].get("warmup_iters", 0),
            )

            # GPU 텐서로만 누적 (sync 없음)
            loss_accum    += loss.detach()
            correct, total = accuracy_counts(logits, labels)
            correct_accum += correct
            total_accum   += total

            iter_count += 1

            # --- Logging --- (여기서만 .item() = GPU sync)
            if iter_count % log_interval == 0:
                avg_loss = (loss_accum / log_interval).item()
                avg_acc  = (correct_accum.float() / total_accum.clamp(min=1).float() * 100.0).item()

                is_frozen = (freeze_iters > 0 and iter_count <= freeze_iters)
                lr_backbone = 0.0 if is_frozen else optimizer.param_groups[0]["lr"]
                lr_head     = optimizer.param_groups[2]["lr"]
                run.log({
                    "step":        iter_count,
                    "train/loss":  avg_loss,
                    "train/acc":   avg_acc,
                    "lr/backbone": lr_backbone,
                    "lr/head":     lr_head,
                })
                print(
                    f"Iter {iter_count:6d}/{total_iters} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Accuracy: {avg_acc:.2f}% | "
                    f"LR(bb): {lr_backbone:.2e} | LR(head): {lr_head:.2e}"
                    + ("  [frozen]" if is_frozen else "")
                )
                loss_accum.zero_()
                correct_accum.zero_()
                total_accum.zero_()

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
                        wandb_run_id=run.id,
                    )
                # latest ckpt: mIoU 갱신 여부와 무관하게 매 eval마다 덮어씀 (crash 복구용)
                save_checkpoint(
                    last_ckpt_path, iter_count, model, optimizer,
                    scaler=scaler, ema=ema,
                    best_val_top1=best_val_top1, cfg=cfg,
                    wandb_run_id=run.id,
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
