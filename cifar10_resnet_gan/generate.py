import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from models.generator import build_generator
from train import select_device


def parse_args():
    p = argparse.ArgumentParser(description="Generate samples from a trained GAN checkpoint.")
    p.add_argument("--ckpt", type=str, default="gan_checkpoint_n3.pth")
    p.add_argument("--out", type=str, default="visualization/n3/generated.png")
    p.add_argument("--num", type=int, default=64, help="number of images to generate")
    p.add_argument("--nrow", type=int, default=8, help="images per row in the saved grid")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default="cuda", help="preferred device (cuda/mps/cpu)")
    return p.parse_args()


def main():
    args = parse_args()
    device = select_device(args.device)
    print(f"Using device: {device}")

    if args.seed is not None:
        torch.manual_seed(args.seed)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    G = build_generator(
        z_dim=cfg["model"]["z_dim"],
        base_channels=cfg["model"]["g_base_channels"],
    ).to(device)
    G.load_state_dict(ckpt["generator"])
    G.eval()

    z = G.sample_z(args.num, device)
    with torch.no_grad():
        samples = G(z)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(samples, out_path, nrow=args.nrow, normalize=True, value_range=(-1, 1))
    print(f"saved {args.num} samples (iter {ckpt.get('iter', '?')}) -> {out_path}")


if __name__ == "__main__":
    main()
