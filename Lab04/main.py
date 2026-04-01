from argparse import ArgumentParser

import torch

from models import deeplab_v3, mobilenet_v2, monodepth_unet, resnet50, resnet50_fpn
from utils.compute_utils import compute_flops
from utils.param_utils import count_parameters


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="OpenSource AI Lab04")
    parser.add_argument(
        "--model",
        type=str,
        default="resnet50",
        choices=["resnet50", "mobilenet_v2", "resnet50_fpn", "monodepth_unet", "deeplab_v3"],
    )
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--device", type=str, default="cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.model == "resnet50":
        model = resnet50(num_classes=args.num_classes)
    elif args.model == "mobilenet_v2":
        model = mobilenet_v2(num_classes=args.num_classes)
    elif args.model == "resnet50_fpn":
        model = resnet50_fpn()
    elif args.model == "monodepth_unet":
        model = monodepth_unet()
    elif args.model == "deeplab_v3":
        model = deeplab_v3(num_classes=args.num_classes)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    params = count_parameters(model)

    input_size = (args.batch_size, 3, args.height, args.width)
    flops = compute_flops(model, input_size=input_size, device=args.device)

    print(f"Model: {args.model}")
    print(f"Parameters: {params:,}")
    print(f"FLOPs: {flops:,}")

    # Sanity forward
    model.eval()
    with torch.no_grad():
        x = torch.zeros(input_size, device=args.device)
        y = model.to(args.device)(x)
        
    if isinstance(y, dict):
        shapes = {k: [tuple(t.shape) for t in v] for k, v in y.items()}
        print(f"Output shapes: {shapes}")
    elif isinstance(y, (list, tuple)):
        shapes = [tuple(t.shape) for t in y]
        print(f"Output shapes: {shapes}")
    else:
        print(f"Output shape: {tuple(y.shape)}")


if __name__ == "__main__":
    main()
