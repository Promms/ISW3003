from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path
from typing import Any

from project02.config import dump_config, load_config


DEFAULT_PIPELINE_CONFIG = Path("src/config/v2_pipeline.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Project02 progressive training stages.")
    parser.add_argument("--pipeline-config", type=str, default=str(DEFAULT_PIPELINE_CONFIG))
    parser.add_argument("--work-dir", type=str, default=None)
    parser.add_argument("--stage", type=str, default=None, help="Run only one stage id.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--trainer", type=str, default=None, help="Override the training module.")
    return parser.parse_args()


def _resolve_existing_path(path_value: str | Path, *, base_dir: Path, project_root: Path, label: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")
        return path

    candidates = [base_dir / path, project_root / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"{label} does not exist: {path_value}")


def _resolve_work_dir(path_value: str | Path, *, project_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return project_root / path


def _stage_items(raw_stages: Any) -> list[dict[str, str]]:
    if isinstance(raw_stages, dict):
        return [{"id": str(stage_id), "config": str(config)} for stage_id, config in raw_stages.items()]
    if isinstance(raw_stages, list):
        stages = []
        for item in raw_stages:
            if not isinstance(item, dict) or "id" not in item or "config" not in item:
                raise ValueError("Each pipeline stage must contain id and config.")
            stages.append({"id": str(item["id"]), "config": str(item["config"])})
        return stages
    raise TypeError("pipeline stages must be a mapping or a list of stage entries.")


def _stage_mode(cfg: dict[str, Any]) -> str:
    progressive = cfg.get("progressive", {})
    mode = str(progressive.get("stage_mode", "stabilize")).lower()
    return "stabilize" if mode == "base" else mode


def _inject_runtime_paths(
    cfg: dict[str, Any],
    *,
    stage_id: str,
    stage_output_dir: Path,
    checkpoint_name: str,
    previous_checkpoint: Path | None,
    dry_run: bool,
) -> tuple[dict[str, Any], str]:
    cfg = copy.deepcopy(cfg)
    training = cfg.setdefault("training", {})
    stage_checkpoint = stage_output_dir / checkpoint_name
    training["output_dir"] = str(stage_output_dir)

    same_stage_resume = stage_checkpoint.exists()
    training["resume_mode"] = "strict"

    if same_stage_resume:
        training["resume"] = str(stage_checkpoint)
        training["init_from"] = {"enabled": False, "checkpoint": None}
        action = f"resume={stage_checkpoint}"
    else:
        training["resume"] = None
        if previous_checkpoint is None:
            training["init_from"] = {"enabled": False, "checkpoint": None}
            action = "fresh"
        else:
            if not previous_checkpoint.exists() and not dry_run:
                raise FileNotFoundError(
                    f"{stage_id} needs previous stage checkpoint, but it does not exist: {previous_checkpoint}"
                )
            training["init_from"] = {"enabled": True, "checkpoint": str(previous_checkpoint)}
            action = f"init_from={previous_checkpoint}"

    consistency = training.setdefault("consistency", {})
    if _stage_mode(cfg) == "transition":
        if previous_checkpoint is None:
            raise ValueError(f"{stage_id} is a transition stage but has no previous stage.")
        if not previous_checkpoint.exists() and not dry_run:
            raise FileNotFoundError(
                f"{stage_id} needs teacher checkpoint, but it does not exist: {previous_checkpoint}"
            )
        consistency["teacher_checkpoint"] = str(previous_checkpoint)
    else:
        if isinstance(consistency, dict):
            consistency["teacher_checkpoint"] = None

    return cfg, action


def _run_stage(entrypoint: str, runtime_config: Path) -> None:
    command = [sys.executable, "-m", entrypoint, "--config", str(runtime_config)]
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    project_root = Path.cwd()
    pipeline_path = _resolve_existing_path(
        args.pipeline_config,
        base_dir=project_root,
        project_root=project_root,
        label="pipeline config",
    )
    pipeline = load_config(pipeline_path)
    pipeline_dir = pipeline_path.parent

    work_dir_value = args.work_dir or pipeline.get("work_dir")
    if not work_dir_value:
        raise ValueError("pipeline config requires work_dir, or pass --work-dir.")
    work_dir = _resolve_work_dir(work_dir_value, project_root=project_root)
    runtime_config_dir = work_dir / "configs"
    checkpoint_name = str(pipeline.get("checkpoint_name", "model.pth"))
    entrypoint = args.trainer or str(pipeline.get("entrypoint", "project02.train"))

    all_stages = _stage_items(pipeline.get("stages"))
    stages = all_stages
    initial_previous_checkpoint: Path | None = None
    if args.stage is not None:
        selected_index = next((index for index, stage in enumerate(all_stages) if stage["id"] == args.stage), None)
        if selected_index is None:
            raise ValueError(f"Unknown stage id: {args.stage}")
        stages = [all_stages[selected_index]]
        if selected_index > 0:
            previous_stage_id = all_stages[selected_index - 1]["id"]
            initial_previous_checkpoint = work_dir / previous_stage_id / checkpoint_name

    previous_checkpoint = initial_previous_checkpoint
    for stage in stages:
        stage_id = stage["id"]
        stage_config_path = _resolve_existing_path(
            stage["config"],
            base_dir=pipeline_dir,
            project_root=project_root,
            label=f"{stage_id} config",
        )
        stage_cfg = load_config(stage_config_path)
        stage_output_dir = work_dir / stage_id
        runtime_cfg, action = _inject_runtime_paths(
            stage_cfg,
            stage_id=stage_id,
            stage_output_dir=stage_output_dir,
            checkpoint_name=checkpoint_name,
            previous_checkpoint=previous_checkpoint,
            dry_run=args.dry_run,
        )
        runtime_config_path = runtime_config_dir / f"{stage_id}.yaml"

        print(f"[{stage_id}] {action}")
        print(f"  source: {stage_config_path}")
        print(f"  runtime: {runtime_config_path}")
        print(f"  output: {stage_output_dir}")

        if not args.dry_run:
            dump_config(runtime_cfg, runtime_config_path)
            _run_stage(entrypoint, runtime_config_path)

        previous_checkpoint = stage_output_dir / checkpoint_name


if __name__ == "__main__":
    main()
