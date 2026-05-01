from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.deeplabv3plus import deeplab_v3  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="mobilenet_v2",
                        choices=["mobilenet_v2", "mobilenet_v3_large"])
    parser.add_argument("--aspp_channels", type=int, default=224)
    parser.add_argument("--decoder_low_channels", type=int, default=48)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--num_classes", type=int, default=21)
    return parser.parse_args()


def conv2d_flops(module: torch.nn.Conv2d, output: torch.Tensor) -> int:
    batch, out_ch, out_h, out_w = output.shape
    kernel_h, kernel_w = module.kernel_size
    in_per_group = module.in_channels // module.groups
    macs = batch * out_h * out_w * out_ch * in_per_group * kernel_h * kernel_w
    return macs * 2


@torch.no_grad()
def measure(model: torch.nn.Module, shape: tuple[int, int, int, int]) -> tuple[float, dict[str, float]]:
    model.eval()
    total = 0
    by_top_level: dict[str, int] = {}
    handles = []

    def hook(name: str):
        def inner(module, _inputs, output):
            nonlocal total
            if not isinstance(output, torch.Tensor):
                return
            flops = conv2d_flops(module, output)
            total += flops
            top = name.split(".", 1)[0]
            by_top_level[top] = by_top_level.get(top, 0) + flops
        return inner

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            handles.append(module.register_forward_hook(hook(name)))

    model(torch.randn(*shape))

    for handle in handles:
        handle.remove()

    return total / 1e9, {key: val / 1e9 for key, val in by_top_level.items()}


def main() -> None:
    args = parse_args()
    model = deeplab_v3(
        num_classes=args.num_classes,
        backbone=args.backbone,
        aspp_channels=args.aspp_channels,
        decoder_low_channels=args.decoder_low_channels,
        pretrained_backbone=False,
    )
    gflops, parts = measure(model, (1, 3, args.height, args.width))
    params = sum(p.numel() for p in model.parameters())

    print(f"backbone: {args.backbone}")
    print(f"aspp_channels: {args.aspp_channels}")
    print(f"input: 1x3x{args.height}x{args.width}")
    print(f"params: {params:,}")
    print(f"GFLOPs: {gflops:.3f}")
    for name, value in sorted(parts.items()):
        print(f"  {name:<14s} {value:.3f}")


if __name__ == "__main__":
    main()
