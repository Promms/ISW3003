from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.coco_voc import COCO_TO_VOC  # noqa: E402

_COCO = None


def _init_worker(ann_file: str) -> None:
    global _COCO
    from pycocotools.coco import COCO

    _COCO = COCO(ann_file)


def _build_mask_smallest_first(img_id: int, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)

    ann_ids = _COCO.getAnnIds(imgIds=img_id, iscrowd=False)
    annotations = _COCO.loadAnns(ann_ids)

    entries = []
    for ann in annotations:
        voc_idx = COCO_TO_VOC.get(ann["category_id"])
        if voc_idx is None:
            continue
        try:
            ann_mask = _COCO.annToMask(ann)
        except Exception:
            continue
        if ann_mask.shape != mask.shape:
            continue

        ann_mask = ann_mask.astype(bool)
        area = int(ann_mask.sum())
        if area > 0:
            entries.append((area, voc_idx, ann_mask))

    entries.sort(key=lambda item: -item[0])
    for _, voc_idx, ann_mask in entries:
        mask[ann_mask] = voc_idx

    return mask


def _process_one(args: tuple[int, int, int, str]) -> tuple[int, bool]:
    img_id, height, width, out_path = args
    mask = _build_mask_smallest_first(img_id, height, width)
    Image.fromarray(mask, mode="L").save(out_path, optimize=False)
    return img_id, bool((mask != 0).any())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-render COCO instance annotations into VOC-index PNG masks."
    )
    parser.add_argument("--ann_file", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--filter_empty", action="store_true", default=True)
    parser.add_argument("--no_filter_empty", dest="filter_empty", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from pycocotools.coco import COCO

    print(f"[prerender] loading {args.ann_file} ...")
    start_time = time.time()
    coco = COCO(args.ann_file)
    print(f"[prerender] loaded annotations in {time.time() - start_time:.1f}s")

    if args.filter_empty:
        img_ids: set[int] = set()
        for category_id in COCO_TO_VOC:
            img_ids.update(coco.getImgIds(catIds=[category_id]))
        img_ids = sorted(img_ids)
    else:
        img_ids = sorted(coco.getImgIds())

    tasks = []
    skipped = 0
    for img_id in img_ids:
        out_path = os.path.join(args.out_dir, f"{img_id:012d}.png")
        if not args.overwrite and os.path.exists(out_path):
            skipped += 1
            continue
        info = coco.loadImgs(img_id)[0]
        tasks.append((img_id, info["height"], info["width"], out_path))

    print(
        f"[prerender] target images: {len(img_ids)} | "
        f"to create: {len(tasks)} | skipped: {skipped}"
    )
    if not tasks:
        print("[prerender] nothing to do")
        return

    start_time = time.time()
    done = 0
    empty = 0
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(args.ann_file,),
    ) as executor:
        futures = [executor.submit(_process_one, task) for task in tasks]
        for future in as_completed(futures):
            _, has_foreground = future.result()
            done += 1
            empty += int(not has_foreground)

            if done % 1000 == 0 or done == len(tasks):
                elapsed = time.time() - start_time
                rate = done / elapsed
                eta = (len(tasks) - done) / rate if rate > 0 else 0.0
                print(
                    f"  [{done:6d}/{len(tasks)}] "
                    f"{rate:.1f} img/s, ETA {eta / 60:.1f} min, empty={empty}"
                )

    total_time = time.time() - start_time
    print(f"[prerender] done: {done} masks in {total_time / 60:.1f} min")
    print(f"[prerender] output: {args.out_dir}")


if __name__ == "__main__":
    main()
