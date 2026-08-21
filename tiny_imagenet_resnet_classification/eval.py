import argparse
import yaml
import torch
import torch.nn as nn

from models.resnet import build_model
from data.tiny_imagenet import get_tiny_imagenet_dataloaders
from utils.metrics import accuracy, AverageMeter
from utils.param_utils import log_parameter_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TinyImageNet model")
    parser.add_argument("--checkpoint", type=str, default="best_checkpoint.pth",
                        help="Path to .pth checkpoint file")
    parser.add_argument("--config", type=str, default="config/tiny_imagenet.yaml")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override config batch size")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    # --- Load checkpoint ---
    ckpt = torch.load(args.checkpoint, map_location=device)

    # --- Build model ---
    model = build_model(
        num_classes=cfg["model"]["num_classes"],
        dropout=cfg["model"]["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    log_parameter_counts(model)

    # --- Data ---
    _, val_loader = get_tiny_imagenet_dataloaders(
        dataset_name=cfg["data"]["dataset_name"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"] and torch.cuda.is_available(),
    )

    # --- Evaluate ---
    criterion  = nn.CrossEntropyLoss()
    loss_meter = AverageMeter()
    acc_meter  = AverageMeter()

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss   = criterion(logits, labels)
            n = images.size(0)
            loss_meter.update(loss.item(), n)
            acc_meter.update(accuracy(logits, labels), n)

    print("=== Evaluation Results ===")
    print(f"  Loss : {loss_meter.avg:.4f}")
    print(f"  Top-1: {acc_meter.avg:.2f}%")


if __name__ == "__main__":
    main()
