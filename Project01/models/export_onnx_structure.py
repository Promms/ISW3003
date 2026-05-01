from __future__ import annotations

import argparse

import onnx
import torch

from models.deeplabv3plus import deeplab_v3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="mobilenet_v3_large",
                        choices=["mobilenet_v2", "mobilenet_v3_large"])
    parser.add_argument("--aspp_channels", type=int, default=224)
    parser.add_argument("--decoder_low_channels", type=int, default=48)
    parser.add_argument("--num_classes", type=int, default=21)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--output", type=str, default="model_structure.onnx")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = deeplab_v3(
        num_classes=args.num_classes,
        backbone=args.backbone,
        aspp_channels=args.aspp_channels,
        decoder_low_channels=args.decoder_low_channels,
        pretrained_backbone=False,
    )
    model.eval()

    dummy_input = torch.randn(1, 3, args.height, args.width)
    temp_filename = "temp_model.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        temp_filename,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )

    model_onnx = onnx.load(temp_filename)
    for init in model_onnx.graph.initializer:
        init.ClearField("raw_data")
        init.ClearField("float_data")
        init.ClearField("int32_data")
        init.ClearField("int64_data")

    onnx.save(model_onnx, args.output)
    print(f"Saved ONNX structure: {args.output}")


if __name__ == "__main__":
    main()
