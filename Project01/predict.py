from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

from models.deeplabv3plus import deeplab_v3


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

VOC_PALETTE = [
    0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0,
    0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128,
    64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0,
    64, 0, 128, 192, 0, 128, 64, 128, 128, 192, 128, 128,
    0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0,
    0, 64, 128,
]
VOC_PALETTE += [0] * (256 * 3 - len(VOC_PALETTE))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--num_classes", type=int, default=21)
    parser.add_argument("--backbone", type=str, default="mobilenet_v3_large",
                        choices=["mobilenet_v2", "mobilenet_v3_large"])
    parser.add_argument("--aspp_channels", type=int, default=224)
    parser.add_argument("--decoder_low_channels", type=int, default=48)
    parser.add_argument("--img_dir", type=str, default="submit/img")
    parser.add_argument("--pred_dir", type=str, default="submit/pred")
    parser.add_argument("--infer_h", type=int, default=480)
    parser.add_argument("--infer_w", type=int, default=640)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--rename_by_index", action="store_true")
    return parser.parse_args()


def preprocess(image: Image.Image, infer_size: tuple[int, int]) -> torch.Tensor:
    image = TF.resize(image, infer_size, interpolation=Image.BILINEAR)
    tensor = TF.to_tensor(image)
    tensor = TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return tensor.unsqueeze(0)


@torch.no_grad()
def predict_one(
    model,
    image: Image.Image,
    device: torch.device,
    infer_size: tuple[int, int],
) -> np.ndarray:
    orig_w, orig_h = image.size
    x = preprocess(image, infer_size).to(device)
    logits = model(x)
    logits = F.interpolate(logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
    return logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)


def save_index_png(pred: np.ndarray, out_path: Path) -> None:
    image = Image.fromarray(pred, mode="P")
    image.putpalette(VOC_PALETTE)
    image.save(out_path)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    infer_size = (args.infer_h, args.infer_w)

    model = deeplab_v3(
        num_classes=args.num_classes,
        backbone=args.backbone,
        aspp_channels=args.aspp_channels,
        decoder_low_channels=args.decoder_low_channels,
        pretrained_backbone=False,
    ).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    if args.use_ema and "ema_state_dict" in ckpt:
        model.load_state_dict(ckpt["ema_state_dict"])
        weight_name = "ema_state_dict"
    else:
        model.load_state_dict(ckpt["model_state_dict"])
        weight_name = "model_state_dict"
    model.eval()

    img_dir = Path(args.img_dir)
    pred_dir = Path(args.pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    img_paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in exts)
    if not img_paths:
        print(f"No images found in {img_dir}")
        return

    print(f"Loaded {args.ckpt} ({weight_name})")
    print(f"Predicting {len(img_paths)} images at input size {infer_size}")

    max_label = 0
    for idx, img_path in enumerate(img_paths):
        image = Image.open(img_path).convert("RGB")
        pred = predict_one(model, image, device, infer_size)
        max_label = max(max_label, int(pred.max()))
        out_name = f"{idx:03d}.png" if args.rename_by_index else f"{img_path.stem}.png"
        save_index_png(pred, pred_dir / out_name)

        if (idx + 1) % 50 == 0 or (idx + 1) == len(img_paths):
            print(f"  [{idx + 1:4d}/{len(img_paths)}] {img_path.name} -> {out_name}")

    print(f"Done: saved {len(img_paths)} masks to {pred_dir}")
    print(f"Max predicted label: {max_label}")
    if max_label >= args.num_classes:
        print("Warning: predicted label exceeds num_classes")


if __name__ == "__main__":
    main()
