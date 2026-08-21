from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from project02.checkpoint import config_from_checkpoint, load_checkpoint
from project02.config import load_config
from project02.models.generator import build_generator_from_config
from project02.utils import save_sample_grid, seed_everything


def _progressive_inference_args(cfg: dict) -> dict:
    """Resolve progressive forward kwargs for inference.

    Inference always uses stage_mode=stabilize and alpha=1.0. The resolution is
    read from `progressive.resolution`, falling back to `model.image_size`.
    """
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate images from random noise.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model.pth")
    parser.add_argument("--no-ema", action="store_true", help="Use raw generator weights even if EMA weights exist.")
    parser.add_argument(
        "--target-resolution",
        type=int,
        default=1024,
        help="Output image resolution after optional resize. Use 512 to keep native 512 samples.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg.get("seed", 42)))

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    generator = build_generator_from_config(cfg["model"]).to(device)

    weight_source = "random_init"
    if args.checkpoint:
        _validate_checkpoint_profile(cfg, args.checkpoint)
        weight_source = "raw" if args.no_ema else "ema"
        load_checkpoint(args.checkpoint, generator, prefer_ema=not args.no_ema, map_location=device)
        print(f"Loaded checkpoint: {args.checkpoint} ({weight_source})")

    generator.eval()
    num_images = int(cfg.get("sampling", {}).get("num_images", 16))
    nrow = int(cfg.get("sampling", {}).get("nrow", 4))
    forward_kwargs = _progressive_inference_args(cfg)
    with torch.no_grad():
        z = generator.sample_z(num_images, device)
        images = generator(z, **forward_kwargs)
        native_resolution = int(images.shape[-1])
        target_resolution = int(args.target_resolution)
        if native_resolution != target_resolution:
            images = F.interpolate(
                images,
                size=(target_resolution, target_resolution),
                mode="bilinear",
                align_corners=False,
            )
    save_sample_grid(images, args.output, nrow=nrow)
    print(f"native_resolution={native_resolution}")
    print(f"output_resolution={target_resolution}")
    print(f"Generated {weight_source} sample grid saved to {args.output}")


if __name__ == "__main__":
    main()
