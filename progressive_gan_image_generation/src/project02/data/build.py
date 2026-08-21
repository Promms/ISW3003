from __future__ import annotations

from pathlib import Path
import shutil

from torch.utils.data import DataLoader

from project02.data.ffhq import FFHQDataset, IMAGE_EXTENSIONS


def build_dataloader(data_cfg: dict, split: str, shuffle: bool | None = None) -> DataLoader:
    root = data_cfg.get(f"{split}_root")
    if root is None:
        raise ValueError(f"No root configured for split: {split}")
    root = Path(root)

    cache_mode = str(data_cfg.get("cache_mode", "none"))
    dataset_cache_mode = "decoded_uint8" if cache_mode == "decoded_uint8" and split == "train" else "none"
    if cache_mode == "shm_files" and split == "train":
        root = _prepare_shm_cache(root, data_cfg)
    elif cache_mode not in {"none", "decoded_uint8", "shm_files"}:
        raise ValueError(f"Unknown data cache mode: {cache_mode}")

    dataset = FFHQDataset(
        root=root,
        image_size=int(data_cfg["image_size"]),
        horizontal_flip=bool(data_cfg.get("horizontal_flip", False)) and split == "train",
        cache_mode=dataset_cache_mode,
        return_uint8=bool(data_cfg.get("normalize_on_gpu", False)) and dataset_cache_mode == "decoded_uint8",
    )
    if shuffle is None:
        shuffle = split == "train"

    num_workers = int(data_cfg.get("num_workers", 2))
    loader_kwargs = {
        "batch_size": int(data_cfg.get("batch_size", 8)),
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": bool(data_cfg.get("pin_memory", True)),
        "drop_last": split == "train",
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(data_cfg.get("persistent_workers", False))
        loader_kwargs["prefetch_factor"] = int(data_cfg.get("prefetch_factor", 2))

    return DataLoader(dataset, **loader_kwargs)


def _prepare_shm_cache(root: Path, data_cfg: dict) -> Path:
    shm_root = Path(str(data_cfg.get("shm_root", f"/dev/shm/project02_ffhd_{data_cfg['image_size']}")))
    if not Path("/dev/shm").exists():
        raise FileNotFoundError("/dev/shm is not available on this system")

    refresh = bool(data_cfg.get("refresh_shm_cache", False))
    if refresh and shm_root.exists():
        shutil.rmtree(shm_root)

    source_count = _count_images(root)
    target_count = _count_images(shm_root) if shm_root.exists() else 0
    if target_count != source_count:
        print(f"[data] Copying {source_count:,} image files to RAM disk: {shm_root}")
        shm_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(root, shm_root, dirs_exist_ok=True)
        target_count = _count_images(shm_root)
        if target_count != source_count:
            raise RuntimeError(f"Incomplete /dev/shm cache: expected {source_count}, found {target_count}")
    else:
        print(f"[data] Reusing RAM disk dataset: {shm_root}")

    return shm_root


def _count_images(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def build_dataloaders(data_cfg: dict) -> dict[str, DataLoader]:
    loaders: dict[str, DataLoader] = {}
    for split in ("train", "valid", "test"):
        if data_cfg.get(f"{split}_root") is not None:
            loaders[split] = build_dataloader(data_cfg, split)
    return loaders
