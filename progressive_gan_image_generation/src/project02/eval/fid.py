from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import shutil

from PIL import Image

from project02.data.ffhq import IMAGE_EXTENSIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute FID with pytorch-fid.")
    parser.add_argument("--real", type=str, required=True, help="Directory containing real images.")
    parser.add_argument("--fake", type=str, required=True, help="Directory containing generated images.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dims", type=int, default=2048)
    return parser.parse_args()


def compute_fid(
    real_dir: str | Path,
    fake_dir: str | Path,
    batch_size: int = 32,
    device: str = "cuda",
    dims: int = 2048,
) -> float:
    from pytorch_fid.fid_score import calculate_fid_given_paths

    return float(
        calculate_fid_given_paths(
            [str(real_dir), str(fake_dir)],
            batch_size=int(batch_size),
            device=str(device),
            dims=int(dims),
            num_workers=0,
        )
    )


def _list_image_files(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Valid root does not exist: {root}")
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def _stage_subset_manifest_matches(
    manifest_path: Path,
    valid_root: Path,
    image_size: int,
    num_images: int,
    seed: int,
) -> bool:
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if manifest.get("valid_root") != str(valid_root):
        return False
    if int(manifest.get("image_size", -1)) != int(image_size):
        return False
    if int(manifest.get("num_images", -1)) != int(num_images):
        return False
    if int(manifest.get("seed", -1)) != int(seed):
        return False
    if manifest.get("resize") != "FFHQDataset.ResizeBICUBIC.CenterCrop":
        return False
    files = manifest.get("files", [])
    if len(files) != int(num_images):
        return False
    subset_dir = manifest_path.parent
    return all((subset_dir / item.get("file", "")).exists() for item in files)


def prepare_or_reuse_stage_valid_subset(
    valid_root: str | Path,
    output_dir: str | Path,
    image_size: int,
    num_images: int,
    seed: int,
) -> Path:
    """Materialise a deterministic resized subset of `valid_root` for FID scoring.

    Reuses the existing directory when its manifest matches the requested
    parameters; otherwise re-samples `num_images` items with `seed`, centre-
    crops them to `image_size`, and writes a fresh manifest. Used by the
    progressive trainer so that each FID call against the same stage receives
    a stable real-image subset.
    """
    valid_root = Path(valid_root)
    output_dir = Path(output_dir)
    manifest_path = output_dir / "subset_manifest.json"
    image_size = int(image_size)
    num_images = int(num_images)
    seed = int(seed)

    if _stage_subset_manifest_matches(manifest_path, valid_root, image_size, num_images, seed):
        return output_dir

    image_paths = _list_image_files(valid_root)
    if len(image_paths) < num_images:
        raise ValueError(f"Valid subset requires {num_images} images, but found {len(image_paths)} in {valid_root}")

    selected = random.Random(seed).sample(image_paths, num_images)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    copied_files = []
    for index, source in enumerate(selected):
        target = output_dir / f"{index:06d}.png"
        with Image.open(source) as image:
            image = image.convert("RGB")
            width, height = image.size
            scale = image_size / min(width, height)
            resized_size = (round(width * scale), round(height * scale))
            image = image.resize(resized_size, Image.Resampling.BICUBIC)
            left = max((image.width - image_size) // 2, 0)
            top = max((image.height - image_size) // 2, 0)
            image = image.crop((left, top, left + image_size, top + image_size))
            image.save(target)
        copied_files.append({"source": str(source), "file": target.name})

    manifest = {
        "valid_root": str(valid_root),
        "image_size": image_size,
        "num_images": num_images,
        "seed": seed,
        "resize": "FFHQDataset.ResizeBICUBIC.CenterCrop",
        "files": copied_files,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_dir


def main() -> None:
    args = parse_args()
    value = compute_fid(args.real, args.fake, args.batch_size, args.device, args.dims)
    print(value)


if __name__ == "__main__":
    main()
