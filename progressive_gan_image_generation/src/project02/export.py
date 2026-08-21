from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from project02.checkpoint import config_from_checkpoint, load_checkpoint
from project02.config import load_config
from project02.models.generator import build_generator_from_config


def _progressive_inference_args(cfg: dict) -> dict:
    progressive_cfg = cfg.get("progressive", {}) or {}
    model_cfg = cfg.get("model", {}) or {}
    resolution = progressive_cfg.get("resolution") or model_cfg.get("image_size")
    if resolution is None:
        raise ValueError(
            "Inference resolution missing: set progressive.resolution or model.image_size."
        )
    return {"resolution": int(resolution), "alpha": 1.0, "stage_mode": "stabilize"}


def _validate_checkpoint_profile(config_cfg: dict, checkpoint_path: str) -> None:
    ckpt_cfg = config_from_checkpoint(checkpoint_path)
    if ckpt_cfg is None:
        return
    config_generator = build_generator_from_config(config_cfg["model"])
    ckpt_generator = build_generator_from_config(ckpt_cfg["model"])
    config_stages = tuple(getattr(config_generator, "internal_resolutions", ()))
    ckpt_stages = tuple(getattr(ckpt_generator, "internal_resolutions", ()))
    if config_stages != ckpt_stages:
        raise ValueError(
            "Checkpoint/config active stage length mismatch: "
            f"config={config_stages}, checkpoint={ckpt_stages}"
        )


class SubmissionWrapper(nn.Module):
    """Wraps a progressive generator to produce (B, 3, target_resolution, target_resolution) images.

    The native generator output is bilinearly upsampled if its spatial resolution
    does not already match *target_resolution*. This is a submission interface
    guarantee only — it does not imply native-resolution quality.
    """

    def __init__(
        self,
        generator: nn.Module,
        resolution: int,
        alpha: float = 1.0,
        stage_mode: str = "stabilize",
        target_resolution: int = 1024,
    ) -> None:
        super().__init__()
        self.generator = generator
        self.resolution = int(resolution)
        self.alpha = float(alpha)
        self.stage_mode = str(stage_mode)
        self.target_resolution = target_resolution

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        image = self.generator(
            z,
            resolution=self.resolution,
            alpha=self.alpha,
            stage_mode=self.stage_mode,
        )
        if image.shape[-1] != self.target_resolution:
            image = F.interpolate(
                image,
                size=(self.target_resolution, self.target_resolution),
                mode="bilinear",
                align_corners=False,
            )
        return image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export trained generator to ONNX submission format: (B, 512) -> (B, 3, H, H)."
    )
    parser.add_argument("--config", type=str, default=None, help="Optional config YAML.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model.pth", help="Path to checkpoint .pth file.")
    parser.add_argument("--output", type=str, default="model.onnx", help="Output ONNX file path.")
    parser.add_argument(
        "--no-ema",
        action="store_true",
        help="Use raw generator weights even if EMA weights exist.",
    )
    parser.add_argument(
        "--target-resolution",
        type=int,
        default=1024,
        help="Target output spatial resolution (default: 1024).",
    )
    parser.add_argument(
        "--z-dim",
        type=int,
        default=None,
        help="Override latent dimension (default: taken from config model.z_dim).",
    )
    return parser.parse_args()


def main() -> None:
    """Export the submission ONNX wrapper from (B, 512) z to target-resolution images.

    The wrapper emits (B, 3, target_resolution, target_resolution), resizing
    native generator output to the requested submission resolution when needed.
    """

    args = parse_args()
    if args.config is None:
        cfg = config_from_checkpoint(args.checkpoint)
        if cfg is None:
            raise ValueError("--config is required when checkpoint has no saved config.")
    else:
        cfg = load_config(args.config)

    device = torch.device("cpu")
    generator = build_generator_from_config(cfg["model"]).to(device).eval()
    checkpoint_cfg = config_from_checkpoint(args.checkpoint)
    checkpoint_forward_kwargs = _progressive_inference_args(checkpoint_cfg) if checkpoint_cfg is not None else None
    if args.config is not None:
        _validate_checkpoint_profile(cfg, args.checkpoint)
    load_checkpoint(args.checkpoint, generator, prefer_ema=not args.no_ema, map_location=device)
    weight_source = "raw" if args.no_ema else "ema"
    print(f"Loaded checkpoint ({weight_source}): {args.checkpoint}")

    z_dim = args.z_dim if args.z_dim is not None else int(cfg["model"].get("z_dim", 512))
    forward_kwargs = _progressive_inference_args(cfg)
    native_resolution = int(forward_kwargs["resolution"])
    checkpoint_resolution = (
        int(checkpoint_forward_kwargs["resolution"]) if checkpoint_forward_kwargs is not None else native_resolution
    )
    print(f"Loaded checkpoint config resolution: {checkpoint_resolution}")
    print(f"Native generator output resolution: {native_resolution}")
    print(f"Target export resolution: {args.target_resolution}")
    print(f"Using bilinear submission wrapper: {'yes' if native_resolution != args.target_resolution else 'no'}")
    print(f"Weight source: {weight_source}")
    wrapper = SubmissionWrapper(
        generator,
        target_resolution=args.target_resolution,
        **forward_kwargs,
    ).eval()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for sidecar in output_path.parent.glob(output_path.name + "*"):
        if sidecar != output_path:
            sidecar.unlink()

    # Verify PyTorch wrapper outputs before export.
    dummy_z = torch.randn(1, z_dim)
    with torch.no_grad():
        out_1 = wrapper(dummy_z)
    print(
        f"Wrapper output shape (batch=1): {tuple(out_1.shape)}, "
        f"range [{float(out_1.min()):.3f}, {float(out_1.max()):.3f}]"
    )

    dummy_z_4 = torch.randn(4, z_dim)
    with torch.no_grad():
        out_4 = wrapper(dummy_z_4)
    print(f"Wrapper output shape (batch=4): {tuple(out_4.shape)}")

    # Export to ONNX.
    export_kwargs = {
        "input_names": ["z"],
        "output_names": ["image"],
        "dynamic_axes": {"z": {0: "batch"}, "image": {0: "batch"}},
        "opset_version": 17,
        "external_data": False,
        "dynamo": False,
    }
    try:
        torch.onnx.export(wrapper, dummy_z, str(output_path), **export_kwargs)
    except TypeError as exc:
        if "dynamo" not in str(exc):
            raise
        export_kwargs.pop("dynamo")
        torch.onnx.export(wrapper, dummy_z, str(output_path), **export_kwargs)
    for sidecar in output_path.parent.glob(output_path.name + "*"):
        if sidecar != output_path:
            raise RuntimeError(f"Unexpected ONNX sidecar file: {sidecar}")
    print(f"ONNX model saved to: {output_path}")


if __name__ == "__main__":
    main()
