from __future__ import annotations

from typing import Mapping, Protocol

from project02.checkpoint import find_wandb_run_id


class RunLogger(Protocol):
    def log(self, metrics: Mapping[str, object]) -> None: ...
    @property
    def run_id(self) -> str | None: ...
    def finish(self) -> None: ...


class NoOpLogger:
    def log(self, metrics): pass
    @property
    def run_id(self) -> str | None:
        return None
    def finish(self) -> None:
        pass


class WandbLogger:
    def __init__(self, cfg: dict, g_params: int, d_params: int, resume_path: str | None):
        import wandb  # top-level import 금지 — NoOp 경로에서 wandb 미설치 환경을 보호.

        wandb_cfg = cfg.get("wandb", {})
        self._run = wandb.init(
            project=wandb_cfg.get("project", "project02"),
            entity=wandb_cfg.get("entity"),
            name=wandb_cfg.get("run_name"),
            mode=wandb_cfg.get("mode", "online"),
            config=cfg,
            id=find_wandb_run_id(resume_path) if resume_path else None,
            resume="allow",
        )
        self._run.define_metric("*", step_metric="step")
        self._run.config.update(
            {
                "params/generator": g_params,
                "params/discriminator": d_params,
            },
            allow_val_change=True,
        )

    def log(self, metrics):
        self._run.log(dict(metrics), step=int(metrics["step"]))

    @property
    def run_id(self) -> str | None:
        return self._run.id

    def finish(self) -> None:
        self._run.finish()


def build_logger(
    cfg: dict,
    g_params: int,
    d_params: int,
    resume_path: str | None,
) -> RunLogger:
    if not bool(cfg.get("wandb", {}).get("enabled", False)):
        return NoOpLogger()
    return WandbLogger(cfg, g_params, d_params, resume_path)
