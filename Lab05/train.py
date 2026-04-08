import yaml
import torch
import torch.nn as nn
import wandb

from models.resnet import build_model
from data.tiny_imagenet import get_tiny_imagenet_dataloaders
from utils.metrics import accuracy, AverageMeter
from utils.param_utils import log_parameter_counts


def load_config(path: str = "config/tiny_imagenet.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    loss_meter = AverageMeter()
    acc_meter  = AverageMeter()

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss   = criterion(logits, labels)
        n = images.size(0)
        loss_meter.update(loss.item(), n)
        acc_meter.update(accuracy(logits, labels), n)

    model.train()
    return {"loss": loss_meter.avg, "top1": acc_meter.avg}


def main():
    cfg = load_config()

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["seed"])

    # --- wandb ---
    run = wandb.init(
        entity=cfg["wandb"]["team"],
        project=cfg["wandb"]["project"],
        name=cfg["wandb"]["run_name"],
        config=cfg,
    )
    run.define_metric("*", step_metric="step")

    try:
        # --- Data ---
        train_loader, val_loader = get_tiny_imagenet_dataloaders(
            dataset_name=cfg["data"]["dataset_name"],
            batch_size=cfg["training"]["batch_size"],
            num_workers=cfg["data"]["num_workers"],
            pin_memory=cfg["data"]["pin_memory"] and torch.cuda.is_available(),
        )

        # --- Model ---
        model = build_model(
            num_classes=cfg["model"]["num_classes"],
            dropout=cfg["model"]["dropout"],
        ).to(device)

        # --- Log parameter counts once as run summary (static metadata) ---
        param_stats = log_parameter_counts(model)
        run.summary.update({f"params/{k}": v for k, v in param_stats.items()})

        # --- Loss and optimizer ---
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg["training"]["learning_rate"],
            weight_decay=cfg["training"]["weight_decay"],
        )

        total_iters   = cfg["training"]["total_iters"]
        log_interval  = cfg["training"]["log_interval"]
        eval_interval = cfg["training"]["eval_interval"]
        lr_decay      = cfg["training"]["lr_decay"]

        # --- Rolling meters (reset every log_interval) ---
        loss_meter = AverageMeter()
        acc_meter  = AverageMeter()

        # --- Best checkpoint tracking ---
        best_val_top1 = 0.0
        ckpt_path     = "best_checkpoint.pth"

        # --- Iteration-based training loop ---
        data_iter  = iter(train_loader)
        iter_count = 0
        model.train()

        while iter_count < total_iters:
            try:
                images, labels = next(data_iter)
            except StopIteration:
                data_iter      = iter(train_loader)
                images, labels = next(data_iter)

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            # Exponential LR decay: multiply every param group's lr by lr_decay
            for pg in optimizer.param_groups:
                pg["lr"] *= lr_decay

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
                        "train/top1_acc": acc_meter.avg,
                        "lr":             current_lr,
                    },
                )
                print(
                    f"Iter {iter_count:6d}/{total_iters} | "
                    f"Loss: {loss_meter.avg:.4f} | "
                    f"Top-1: {acc_meter.avg:.2f}% | "
                    f"LR: {current_lr:.2e}"
                )
                loss_meter.reset()
                acc_meter.reset()

            if iter_count % eval_interval == 0:
                val_metrics = evaluate(model, val_loader, criterion, device)
                run.log(
                    {
                        "step":         iter_count,
                        "val/loss":     val_metrics["loss"],
                        "val/top1_acc": val_metrics["top1"],
                    },
                )
                is_best = val_metrics["top1"] > best_val_top1
                if is_best:
                    best_val_top1 = val_metrics["top1"]
                    torch.save(
                        {"iter": iter_count, "model_state_dict": model.state_dict(), "config": cfg},
                        ckpt_path,
                    )
                print(
                    f"  [Val] Loss: {val_metrics['loss']:.4f} | "
                    f"Top-1: {val_metrics['top1']:.2f}%"
                    + ("  [best]" if is_best else "")
                )

    finally:
        run.finish()


if __name__ == "__main__":
    main()
