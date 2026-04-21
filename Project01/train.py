import yaml
import torch
import torch.nn as nn
import wandb

from models.deeplabv3plus_mobilenet import deeplab_v3
from data.pascal_voc import get_loader
from utils.metrics import accuracy, mIoU, AverageMeter


def load_config(path: str = "src/semantic_segmentation.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
    
@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion: nn.Module, device: torch.device, num_classes: int) -> dict[str, float]:
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    mIoU_meter = AverageMeter()

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss   = criterion(logits, labels)
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

        # 모델
        model = deeplab_v3(num_classes=cfg["model"]["num_classes"]).to(device)

        # --- 파라미터 수를 run summary에 한 번만 기록 (정적 메타데이터) ---
        # param_stats = log_parameter_counts(model)
        # run.summary.update({f"params/{k}": v for k, v in param_stats.items()})

        # --- 손실함수 및 옵티마이저 ---
        criterion = nn.CrossEntropyLoss(ignore_index=255)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["training"]["learning_rate"],
            weight_decay=cfg["training"]["weight_decay"],
        )

        total_iters = cfg["training"]["total_iters"]
        log_interval = cfg["training"]["log_interval"]
        eval_interval = cfg["training"]["eval_interval"]
        init_lr = cfg["training"]["learning_rate"]

        # --- 구간 평균 미터 (log_interval마다 초기화) ---
        loss_meter = AverageMeter()
        acc_meter  = AverageMeter()

        # --- 최고 성능 체크포인트 추적 ---
        best_val_top1 = 0.0
        ckpt_path = "/content/drive/MyDrive/checkpoints/best_checkpoint.pth"

        # --- 체크포인트 이어받기 ---
        import os
        iter_count = 0
        if cfg["training"]["resume"] and os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            iter_count = ckpt["iter"]
            best_val_top1 = ckpt.get("best_val_top1", 0.0)
            print(f"체크포인트 불러옴: iter {iter_count}, best mIoU {best_val_top1:.4f}")
        else:
            print("처음부터 학습 시작")

        # --- iteration 기반 학습 루프 ---
        data_iter = iter(train_loader)
        model.train()

        while iter_count < total_iters:
            try:
                images, labels = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                images, labels = next(data_iter)

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            # Poly LR: lr × (1 - iter/total_iter)^0.9
            poly_lr = init_lr * (1 - iter_count / total_iters) ** 0.9
            for pg in optimizer.param_groups:
                pg["lr"] = poly_lr

            n = images.size(0)
            loss_meter.update(loss.item(), n)
            acc_meter.update(accuracy(logits, labels), n)

            iter_count += 1

            if iter_count % log_interval == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                run.log(
                    {
                        "step":           iter_count,
                        "train/loss":     loss_meter.avg,
                        "train/acc": acc_meter.avg,
                        "lr":             current_lr,
                    },
                )
                print(
                    f"Iter {iter_count:6d}/{total_iters} | "
                    f"Loss: {loss_meter.avg:.4f} | "
                    f"Accuracy: {acc_meter.avg:.2f}% | "
                    f"LR: {current_lr:.2e}"
                )
                loss_meter.reset()
                acc_meter.reset()

            if iter_count % eval_interval == 0:
                val_metrics = evaluate(model, val_loader, criterion, device, cfg["model"]["num_classes"])
                run.log(
                    {
                        "step":         iter_count,
                        "val/loss":     val_metrics["loss"],
                        "val/acc": val_metrics["acc"],
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