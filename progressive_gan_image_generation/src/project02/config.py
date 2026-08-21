from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def dump_config(cfg: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if config_path.suffix.lower() in {".yaml", ".yml"}:
        with config_path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        return {} if loaded is None else loaded

    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)
