from __future__ import annotations

import argparse
from pathlib import Path

import torch

from project02.config import load_config
from project02.data import build_dataloaders
from project02.utils import save_sample_grid, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check FFHQ dataloaders.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="checkpoints/data_checks")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg.get("seed", 42)))

    loaders = build_dataloaders(cfg["data"])
    output_dir = Path(args.output_dir)

    for split, loader in loaders.items():
        dataset = loader.dataset
        batch = next(iter(loader))
        if not isinstance(batch, torch.Tensor):
            raise TypeError(f"Expected tensor batch for {split}, got {type(batch)}")

        print(f"Split: {split}")
        print(f"Root: {dataset.root}")
        print(f"Images: {len(dataset):,}")
        print(f"Batch shape: {list(batch.shape)}")
        print(f"Value range: [{batch.min().item():.3f}, {batch.max().item():.3f}]")

        nrow = int(cfg.get("sampling", {}).get("nrow", 4))
        output_path = output_dir / f"data_check_{split}.png"
        save_sample_grid(batch, output_path, nrow=nrow)
        print(f"Saved sample grid to {output_path}")


if __name__ == "__main__":
    main()
