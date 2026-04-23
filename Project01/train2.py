import yaml
import torch
import torch.nn as nn
import wandb

from models.deeplabv3plus_efficientnet import deeplab_v3_efficientnet
from data.pascal_voc import get_loader
from utils.metrics import accuracy, mIoU, AverageMeter
from utils.diceloss import CEDiceLoss


def load_config(path: str = "src/semantic_segmentation_efficientnet.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_backbone_requires_grad(model: nn.Module, requires_grad: bool) -> None:
    # Backbone (EfficientNet-B0) 파라미터의 requires_grad를 일괄 변경
    for param in model.backbone_low.parameters():
        param.requires_grad = requires_grad
    for param in model.backbone_high.parameters():
        param.requires_grad = requires_grad


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion: nn.Module, device: torch.device, num_classes: int) -> dict[str, float]:
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


def main():

    cfg = load_config()
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
        # 학습 데이터
        train_loader = get_loader(
            root=cfg["data"]["root"],
            years=cfg["data"]["years"],
            image_set="train",
            crop_size=cfg["data"]["crop_size"],
            batch_size=cfg["training"]["batch_size"],
            num_workers=cfg["data"]["num_workers"],
            pin_memory=cfg["data"]["pin_memory"],
            download=cfg["data"]["download"],
        )

        # 검증 데이터
        val_loader = get_loader(
            root=cfg["data"]["root"],
            years=cfg["data"]["years"],
            image_set="val",
            crop_size=cfg["data"]["crop_size"],
            batch_size=cfg["training"]["batch_size"],
            num_workers=cfg["data"]["num_workers"],
            pin_memory=cfg["data"]["pin_memory"],
        )

        # 모델 (EfficientNet-B0 백본)
        model = deeplab_v3_efficientnet(num_classes=cfg["model"]["num_classes"]).to(device)

        # --- Backbone freeze (Stage 1) ---
        freeze_iters = cfg["training"].get("freeze_iters", 0)
        if freeze_iters > 0:
            set_backbone_requires_grad(model, False)
            print(f"Backbone freeze: 0 ~ {freeze_iters} iter 동안 head만 학습")

        # --- 손실함수 및 옵티마이저 ---
        criterion = CEDiceLoss(num_classes=cfg["model"]["num_classes"], ignore_index=255, dice_weight=1.0)

        # Differential LR: pretrained backbone은 작은 lr, 새로 학습하는 head는 큰 lr
        head_lr     = cfg["training"]["learning_rate"]
        backbone_lr = head_lr * cfg["training"]["backbone_lr_scale"]
        optimizer = torch.optim.AdamW(
            [
                {"params": model.backbone_low.parameters(),  "lr": backbone_lr, "initial_lr": backbone_lr},
                {"params": model.backbone_high.parameters(), "lr": backbone_lr, "initial_lr": backbone_lr},
                {"params": model.aspp.parameters(),          "lr": head_lr,     "initial_lr": head_lr},
                {"params": model.decoder.parameters(),       "lr": head_lr,     "initial_lr": head_lr},
            ],
            weight_decay=cfg["training"]["weight_decay"],
        )

        total_iters  = cfg["training"]["total_iters"]
        log_interval = cfg["training"]["log_interval"]
        eval_interval = cfg["training"]["eval_interval"]

        # --- Mixed Precision ---
        # amp=True: autocast()로 forward/loss 계산을 FP16으로 실행 → 메모리/속도 이득
        # GradScaler: FP16 backward의 작은 gradient가 0으로 underflow 되는 것을 방지
        # use_amp=False면 scaler가 no-op처럼 동작해서 일반 FP32 학습과 동일
        use_amp = cfg["training"].get("amp", False) and device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        if use_amp:
            print("Mixed Precision (FP16 autocast) 활성화")

        # --- 구간 평균 미터 (log_interval마다 초기화) ---
        loss_meter = AverageMeter()
        acc_meter  = AverageMeter()

        # --- 최고 성능 체크포인트 추적 ---
        # run_name별로 파일을 분리 저장해서 실험 비교가 용이하도록 함
        import os
        best_val_top1 = 0.0
        ckpt_dir = cfg["checkpoint"]["dir"]
        ckpt_name = f"best_{cfg['wandb']['run_name']}.pth"
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"체크포인트 경로: {ckpt_path}")

        # --- 체크포인트 이어받기 ---
        iter_count = 0
        if cfg["training"]["resume"] and os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            iter_count = ckpt["iter"]
            best_val_top1 = ckpt.get("best_val_top1", 0.0)
            if "scaler_state_dict" in ckpt and use_amp:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            print(f"체크포인트 불러옴: iter {iter_count}, best mIoU {best_val_top1:.4f}")
        else:
            print("처음부터 학습 시작")

        # --- iteration 기반 학습 루프 ---
        data_iter = iter(train_loader)
        model.train()

        while iter_count < total_iters:
            # --- Backbone unfreeze (Stage 2) ---
            if freeze_iters > 0 and iter_count == freeze_iters:
                set_backbone_requires_grad(model, True)
                print(f"[Iter {iter_count}] Backbone unfreeze: 전체 모델 학습 시작")

            try:
                images, labels = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                images, labels = next(data_iter)

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            # autocast 내부는 FP16로 자동 실행 (BN/Softmax 등 민감한 연산은 자동 FP32 유지)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Poly LR: 각 그룹의 initial_lr 기준으로 감소 (backbone/head 비율 유지)
            decay = (1 - iter_count / total_iters) ** 0.9
            for pg in optimizer.param_groups:
                pg["lr"] = pg["initial_lr"] * decay

            n = images.size(0)
            loss_meter.update(loss.item(), n)
            acc_meter.update(accuracy(logits, labels), n)

            iter_count += 1

            if iter_count % log_interval == 0:
                # param_groups: [0,1]=backbone(low/high, 같은 lr)  [2,3]=head(aspp/decoder, 같은 lr)
                # 실제로 서로 다른 lr은 backbone, head 2개뿐이라 2개만 로깅
                # freeze 중엔 backbone이 학습 안 되므로 "effective LR = 0"으로 기록
                is_frozen = (freeze_iters > 0 and iter_count <= freeze_iters)
                lr_backbone = 0.0 if is_frozen else optimizer.param_groups[0]["lr"]
                lr_head     = optimizer.param_groups[2]["lr"]
                run.log(
                    {
                        "step":         iter_count,
                        "train/loss":   loss_meter.avg,
                        "train/acc":    acc_meter.avg,
                        "lr/backbone": lr_backbone,
                        "lr/head":     lr_head,
                    },
                )
                print(
                    f"Iter {iter_count:6d}/{total_iters} | "
                    f"Loss: {loss_meter.avg:.4f} | "
                    f"Accuracy: {acc_meter.avg:.2f}% | "
                    f"LR(bb): {lr_backbone:.2e} | LR(head): {lr_head:.2e}"
                    + ("  [frozen]" if is_frozen else "")
                )
                loss_meter.reset()
                acc_meter.reset()

            if iter_count % eval_interval == 0:
                val_metrics = evaluate(model, val_loader, criterion, device, cfg["model"]["num_classes"])
                run.log(
                    {
                        "step":          iter_count,
                        "val/loss":      val_metrics["loss"],
                        "val/acc":       val_metrics["acc"],
                        "val/top1_mIoU": val_metrics["top1/mIoU"],
                    },
                )
                is_best = val_metrics["top1/mIoU"] > best_val_top1
                if is_best:
                    best_val_top1 = val_metrics["top1/mIoU"]
                    torch.save(
                        {
                            "iter": iter_count,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scaler_state_dict": scaler.state_dict(),  # AMP 이어받기용
                            "best_val_top1": best_val_top1,
                            "config": cfg,
                        },
                        ckpt_path,
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
