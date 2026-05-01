"""
파라미터 수 확인 스크립트.

사용법:
    python check_params.py              # V6 + V7 모두
    python check_params.py --model v6   # V6만
    python check_params.py --model v7   # V7만
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

BASE = os.path.dirname(os.path.abspath(__file__))

MODELS = {
    "v6": ("EfficientNet_V6", "EfficientNet_V6 (lovasz)"),
    "v7": ("EfficientNet_V7", "EfficientNet_V7 (B1)"),
}

# 각 프로젝트 디렉터리 안에서 독립 실행되는 미니 스크립트
_RUNNER = """
import sys
sys.path.insert(0, ".")
from utils.model_factory import build_model
from utils.param_utils import log_parameter_counts
model = build_model("efficientnet", num_classes=21)
log_parameter_counts(model)
"""


def check(folder: str, label: str) -> None:
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")

    result = subprocess.run(
        [sys.executable, "-c", _RUNNER],
        cwd=os.path.join(BASE, folder),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"[ERROR] {label} 실행 실패 (returncode={result.returncode})")
        print(result.stderr)
    else:
        print(result.stdout, end="")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["v6", "v7"],
        default=None,
        help="확인할 모델 (생략하면 전체)",
    )
    args = parser.parse_args()

    targets = [args.model] if args.model else list(MODELS)
    for key in targets:
        folder, label = MODELS[key]
        check(folder, label)
    print()


if __name__ == "__main__":
    main()
