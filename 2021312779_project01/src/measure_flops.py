from __future__ import annotations

import argparse

import torch

from models.deeplabv3plus import deeplab_v3_efficientnet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure FLOPs for the submitted model.")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--num_classes", type=int, default=21)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = deeplab_v3_efficientnet(args.num_classes, pretrained=False).to(device)
    model.eval()

    x = torch.randn(1, 3, args.height, args.width, device=device)
    with torch.no_grad():
        with torch.profiler.profile(with_flops=True) as prof:
            model(x)

    total_flops = sum(event.flops for event in prof.key_averages() if event.flops is not None)
    print(f"Input shape: 1x3x{args.height}x{args.width}")
    print(f"FLOPs: {total_flops:,}")
    print(f"GFLOPs: {total_flops / 1e9:.3f}")


if __name__ == "__main__":
    main()
