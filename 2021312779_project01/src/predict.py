from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

from models.deeplabv3plus import deeplab_v3_efficientnet


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
    parser = argparse.ArgumentParser(description="Run semantic segmentation inference.")
    parser.add_argument("--ckpt", type=str, default="checkpoints/model.pth")
    parser.add_argument("--num_classes", type=int, default=21)
    parser.add_argument("--img_dir", type=str, default="submit/img")
    parser.add_argument("--pred_dir", type=str, default="submit/pred")
    parser.add_argument("--infer_h", type=int, default=480)
    parser.add_argument("--infer_w", type=int, default=640)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--rename_by_index", action="store_true")
    parser.add_argument("--use_ema", action="store_true")
    return parser.parse_args()


def preprocess(image: Image.Image, infer_size: tuple[int, int]) -> torch.Tensor:
    image = TF.resize(image, infer_size, interpolation=Image.BILINEAR)
    tensor = TF.to_tensor(image)
    tensor = TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return tensor.unsqueeze(0)


@torch.no_grad()
def predict_one(
    model: torch.nn.Module,
    image: Image.Image,
    device: torch.device,
    infer_size: tuple[int, int],
) -> np.ndarray:
    original_width, original_height = image.size
    x = preprocess(image, infer_size).to(device)
    # A single fixed-size forward pass keeps inference reproducible.
    out = model(x)
    out = F.interpolate(out, size=(original_height, original_width), mode="bilinear", align_corners=False)
    return out.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)


def load_state_dict(path: str, device: torch.device, use_ema: bool) -> dict:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    # The same model.pth can contain both raw and EMA weights.
    if isinstance(ckpt, dict) and use_ema and "ema_state_dict" in ckpt:
        print("Loaded EMA weights.")
        return ckpt["ema_state_dict"]
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        if use_ema:
            print("EMA weights were requested but not found; using model_state_dict.")
        return ckpt["model_state_dict"]
    return ckpt


def save_pred_png(pred: np.ndarray, out_path: Path) -> None:
    image = Image.fromarray(pred, mode="P")
    image.putpalette(VOC_PALETTE)
    image.save(out_path)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    infer_size = (args.infer_h, args.infer_w)

    model = deeplab_v3_efficientnet(args.num_classes, pretrained=False).to(device)
    model.load_state_dict(load_state_dict(args.ckpt, device, args.use_ema))
    model.eval()

    img_dir = Path(args.img_dir)
    pred_dir = Path(args.pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    img_paths = sorted(path for path in img_dir.iterdir() if path.suffix.lower() in exts)
    if not img_paths:
        print(f"No images found in {img_dir}.")
        return

    max_class = 0
    for idx, img_path in enumerate(img_paths):
        image = Image.open(img_path).convert("RGB")
        pred = predict_one(
            model,
            image,
            device,
            infer_size,
        )
        max_class = max(max_class, int(pred.max()))

        out_name = f"{idx:03d}.png" if args.rename_by_index else f"{img_path.stem}.png"
        save_pred_png(pred, pred_dir / out_name)

        if (idx + 1) % 50 == 0 or (idx + 1) == len(img_paths):
            print(f"[{idx + 1}/{len(img_paths)}] {img_path.name} -> {out_name}")

    print(f"Saved {len(img_paths)} predictions to {pred_dir}.")
    print(f"Maximum predicted class index: {max_class}")
    if max_class >= args.num_classes:
        raise ValueError(f"Predicted class index {max_class} exceeds num_classes={args.num_classes}.")


if __name__ == "__main__":
    main()
