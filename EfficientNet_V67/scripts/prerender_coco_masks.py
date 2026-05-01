"""
COCO → VOC 매핑 세그멘테이션 마스크 사전 렌더링.

학습 중 pycocotools `annToMask()` + smallest_first 정렬이 매 샘플마다 15~50ms
소요되는 CPU 병목을 없애기 위해, 모든 대상 이미지의 최종 VOC-index 마스크를
미리 PNG로 디스크에 저장해둔다.

저장 포맷:
    <out_dir>/<img_id:012d>.png  (PIL mode='L', uint8, 값 0~20 + 255=ignore)

학습 중엔 `CocoVOCSegDataset(mask_cache_dir=<out_dir>)` 로 연결 →
    PIL.Image.open(path) 한 번이면 끝 (2~5ms).

사용:
    python scripts/prerender_coco_masks.py \
        --ann_file /content/coco/annotations/instances_train2017.json \
        --out_dir  /content/coco/voc_masks \
        --workers 2

주의:
    - `data.coco_voc` 의 `COCO_TO_VOC` 매핑과 smallest_first 로직을 그대로 호출.
      매핑 변경되면 이 캐시도 무효 → 재생성 필요.
    - filter_empty 적용해서 VOC 매핑 클래스 있는 이미지만 저장 (~65k/118k).
      학습 시에도 동일 필터를 쓰므로 대상 이미지 집합이 정합.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# 프로젝트 루트 import path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from PIL import Image

from data.coco_voc import COCO_TO_VOC
from data.transforms import IGNORE_INDEX


# 워커 프로세스 전역 (fork 후 초기화됨)
_COCO = None


def _init_worker(ann_file: str) -> None:
    """각 워커에서 COCO 인덱스 1회 로드 (fork 후 공유 불가능 → 각자 로드)."""
    global _COCO
    from pycocotools.coco import COCO
    _COCO = COCO(ann_file)


def _build_mask_smallest_first(img_id: int, H: int, W: int) -> np.ndarray:
    """
    data.coco_voc._build_mask 의 "smallest_first" 분기와 동일 로직.

    재구현 이유: CocoVOCSegDataset 인스턴스를 만들지 않고 단순 함수만으로 실행하기 위함.
    COCO_TO_VOC 매핑 / IGNORE_INDEX 는 동일 소스를 import.
    """
    mask = np.zeros((H, W), dtype=np.uint8)

    ann_ids = _COCO.getAnnIds(imgIds=img_id, iscrowd=False)
    anns = _COCO.loadAnns(ann_ids)

    entries = []
    for ann in anns:
        voc_idx = COCO_TO_VOC.get(ann["category_id"])
        if voc_idx is None:
            continue
        try:
            m = _COCO.annToMask(ann)
        except Exception:
            continue
        if m.shape != mask.shape:
            continue
        m_bool = m.astype(bool)
        area = int(m_bool.sum())
        if area == 0:
            continue
        entries.append((area, voc_idx, m_bool))

    # 큰 순 → 작은 순 (작은 게 위로 올라와 덮음)
    entries.sort(key=lambda e: -e[0])
    for _, voc_idx, m_bool in entries:
        mask[m_bool] = voc_idx

    return mask


def _process_one(args: tuple[int, int, int, str]) -> tuple[int, bool]:
    """
    한 이미지 → mask PNG 저장.

    args: (img_id, H, W, out_path)
    반환: (img_id, has_any_fg)
        has_any_fg=False 이면 mask가 전부 0 (이론상 filter_empty 후엔 거의 없음)
    """
    img_id, H, W, out_path = args
    mask = _build_mask_smallest_first(img_id, H, W)

    # mode='L' uint8 저장. 학습 시 PIL.Image.open → np.array 바로 사용 가능.
    Image.fromarray(mask, mode="L").save(out_path, optimize=False)

    has_fg = bool((mask != 0).any())
    return img_id, has_fg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann_file", type=str, required=True,
                        help="COCO instances_*.json 경로")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="출력 PNG 폴더")
    parser.add_argument("--workers", type=int, default=2,
                        help="병렬 워커 수 (Colab 무료 CPU 2 → 2 권장)")
    parser.add_argument("--filter_empty", action="store_true", default=True,
                        help="VOC 매핑 클래스 있는 이미지만 렌더 (default True)")
    parser.add_argument("--no_filter_empty", dest="filter_empty", action="store_false")
    parser.add_argument("--overwrite", action="store_true",
                        help="이미 있는 PNG도 덮어쓰기 (default: skip)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # --- main 프로세스에서 COCO 한 번 로드 (작업 리스트 구성용) ---
    from pycocotools.coco import COCO
    print(f"[prerender] loading {args.ann_file} ...")
    t0 = time.time()
    coco = COCO(args.ann_file)
    print(f"[prerender] annotation 로드 완료 ({time.time()-t0:.1f}s)")

    # filter_empty: VOC 매핑 클래스 있는 이미지만
    if args.filter_empty:
        voc_cat_ids = list(COCO_TO_VOC.keys())
        img_ids: set[int] = set()
        for cid in voc_cat_ids:
            img_ids.update(coco.getImgIds(catIds=[cid]))
        img_ids = sorted(img_ids)
    else:
        img_ids = sorted(coco.getImgIds())

    print(f"[prerender] 대상 이미지: {len(img_ids)}장 "
          f"(filter_empty={args.filter_empty})")

    # 작업 리스트 구성
    tasks = []
    skipped = 0
    for img_id in img_ids:
        out_path = os.path.join(args.out_dir, f"{img_id:012d}.png")
        if (not args.overwrite) and os.path.exists(out_path):
            skipped += 1
            continue
        info = coco.loadImgs(img_id)[0]
        tasks.append((img_id, info["height"], info["width"], out_path))

    print(f"[prerender] 생성할 파일: {len(tasks)}개 / 스킵: {skipped}개")
    if not tasks:
        print("[prerender] 할 일 없음. 종료.")
        return

    # --- 병렬 렌더 ---
    t0 = time.time()
    done = 0
    empty = 0
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(args.ann_file,),
    ) as ex:
        futures = [ex.submit(_process_one, t) for t in tasks]
        for fut in as_completed(futures):
            _, has_fg = fut.result()
            done += 1
            if not has_fg:
                empty += 1
            if done % 1000 == 0 or done == len(tasks):
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (len(tasks) - done) / rate if rate > 0 else 0
                print(f"  [{done:6d}/{len(tasks)}] "
                      f"{rate:.1f} img/s, ETA {eta/60:.1f}min "
                      f"(empty so far: {empty})")

    total_time = time.time() - t0
    print(f"\n[prerender] 완료: {done}장 렌더링 ({total_time/60:.1f}min)")
    print(f"  전경 없는 마스크: {empty}장")
    print(f"  출력: {args.out_dir}")


if __name__ == "__main__":
    main()
