from __future__ import annotations

import argparse
import copy
import gc
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import shutil
import time
import warnings

import torch
import torch.nn.functional as F

from project02.checkpoint import config_from_checkpoint, load_checkpoint, save_checkpoint
from project02.config import load_config
from project02.data import build_dataloader
from project02.eval.fid import compute_fid, prepare_or_reuse_stage_valid_subset
from project02.loss import (
    discriminator_adversarial_loss,
    exact_gp_value_and_param_grads,
    generator_adversarial_loss,
    inject_gp_param_grads,
)
from project02.models.discriminator import build_discriminator_from_config
from project02.models.generator import build_generator_from_config
from project02.resume import (
    _init_from_config,
    _stage_config,
    _validate_resume_state,
    load_previous_stage_init,
    progressive_state_from_config,
    read_progressive_state,
)
from project02.logger import build_logger
from project02.utils import count_parameters, export_generated_images, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project02 training entrypoint")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def grad_norm(parameters) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        total += float(param.grad.detach().pow(2).sum().cpu())
    return total**0.5


def set_requires_grad(module: torch.nn.Module, requires_grad: bool) -> None:
    for param in module.parameters():
        param.requires_grad_(requires_grad)


def update_ema(source: torch.nn.Module, target: torch.nn.Module, beta: float) -> None:
    """Update target parameters as beta * target + (1 - beta) * source."""

    with torch.no_grad():
        pairs = list(zip(target.parameters(), source.parameters()))
        if not all(
            target_param.device == source_param.device and target_param.dtype == source_param.dtype
            for target_param, source_param in pairs
        ):
            for target_param, source_param in pairs:
                target_param.mul_(beta).add_(
                    source_param.detach().to(device=target_param.device, dtype=target_param.dtype),
                    alpha=1.0 - beta,
                )
            return

        grouped: dict[tuple[torch.device, torch.dtype], tuple[list[torch.Tensor], list[torch.Tensor]]] = {}
        for target_param, source_param in pairs:
            key = (target_param.device, target_param.dtype)
            target_group, source_group = grouped.setdefault(key, ([], []))
            target_group.append(target_param)
            source_group.append(source_param.detach())

        for target_group, source_group in grouped.values():
            torch._foreach_mul_(target_group, beta)
            torch._foreach_add_(target_group, source_group, alpha=1.0 - beta)


def sync_ema_buffers(source: torch.nn.Module, target: torch.nn.Module) -> None:
    with torch.no_grad():
        for target_buffer, source_buffer in zip(target.buffers(), source.buffers()):
            target_buffer.copy_(source_buffer)


def amp_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(f"Unknown AMP dtype: {name}")


def make_scaler(enabled: bool):
    if not enabled:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=True)


def autocast_context(device: torch.device, enabled: bool, dtype: torch.dtype):
    if not enabled or device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def repeat_loader(loader):
    while True:
        yield from loader


def check_finite_many(step: int, named_tensors: list[tuple[str, torch.Tensor]]) -> None:
    if not named_tensors:
        return

    ok = torch.ones((), device=named_tensors[0][1].device, dtype=torch.bool)
    for _, tensor in named_tensors:
        ok = ok & torch.isfinite(tensor).all()

    if bool(ok):
        return

    for name, tensor in named_tensors:
        if not bool(torch.isfinite(tensor).all()):
            raise FloatingPointError(f"Non-finite {name} at step {step}")
    raise FloatingPointError(f"Non-finite tensor at step {step}")


def prepare_batch(batch: torch.Tensor, device: torch.device) -> torch.Tensor:
    batch = batch.to(device, non_blocking=True)
    if batch.dtype == torch.uint8:
        batch = batch.float().div(127.5).sub(1.0)
    return batch


def resolve_schedule_value(spec, stage_kimg: float, fallback: float) -> float:
    """Resolve a scalar or linear schedule spec at the current stage kimg."""

    if spec is None:
        return float(fallback)
    if isinstance(spec, (int, float)):
        return float(spec)
    if not isinstance(spec, dict):
        raise TypeError("Schedule spec must be a scalar or dict.")
    start = float(spec.get("start", fallback))
    end = float(spec.get("end", start))
    duration = float(spec["duration_kimg"])
    if duration <= 0.0:
        return end
    progress = max(0.0, min(float(stage_kimg) / duration, 1.0))
    return start + (end - start) * progress


def _build_generator(model_cfg: dict) -> torch.nn.Module:
    return build_generator_from_config(model_cfg)


def _build_discriminator(model_cfg: dict) -> torch.nn.Module:
    return build_discriminator_from_config(model_cfg)


def _module_max_resolution(module: torch.nn.Module) -> int:
    supported = getattr(module, "supported_resolutions", ())
    if not supported:
        raise ValueError("Progressive module does not expose supported_resolutions.")
    return int(max(supported))


def _validate_generator_discriminator_profiles(generator: torch.nn.Module, discriminator: torch.nn.Module) -> None:
    g_res = tuple(getattr(generator, "internal_resolutions", ()))
    d_res = tuple(getattr(discriminator, "internal_resolutions", ()))
    if g_res != d_res:
        raise ValueError(
            "Generator/discriminator active stage lengths must match: "
            f"generator={g_res}, discriminator={d_res}"
        )


def maybe_copy_to_rgb_from_previous(generator: torch.nn.Module, prev_resolution: int, resolution: int) -> bool:
    """Copy previous toRGB weights into the current stage when shapes match."""

    prev = generator.to_rgb[str(int(prev_resolution))]
    cur = generator.to_rgb[str(int(resolution))]
    if prev.conv.weight.shape != cur.conv.weight.shape:
        return False
    with torch.no_grad():
        cur.conv.weight.copy_(prev.conv.weight)
    return True


def load_teacher_generator(
    model_cfg: dict,
    checkpoint_path: str | Path,
    device: torch.device,
    prefer_ema: bool,
    expected_resolution: int,
) -> torch.nn.Module:
    """Build and load a frozen previous-resolution teacher generator."""

    teacher_state = read_progressive_state(checkpoint_path)
    if int(teacher_state["resolution"]) != int(expected_resolution):
        raise ValueError(
            "Teacher checkpoint resolution does not match prev_resolution: "
            f"teacher={teacher_state.get('resolution')!r}, prev_resolution={expected_resolution!r}"
        )
    teacher_cfg = config_from_checkpoint(checkpoint_path)
    if teacher_cfg is not None:
        teacher_model_cfg = teacher_cfg.get("model", {})
    else:
        fallback_teacher = _build_generator(model_cfg)
        if _module_max_resolution(fallback_teacher) != int(expected_resolution):
            raise ValueError(
                "Teacher checkpoint has no saved config and current model config does not match "
                f"expected teacher resolution {expected_resolution}."
            )
        teacher_model_cfg = model_cfg
    teacher = _build_generator(teacher_model_cfg).to(device)
    load_checkpoint(checkpoint_path, teacher, map_location=device, prefer_ema=prefer_ema)
    teacher.eval()
    set_requires_grad(teacher, False)
    return teacher


def consistency_loss(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    z: torch.Tensor,
    resolution: int,
    prev_resolution: int,
    alpha: float,
    weight: float,
) -> torch.Tensor:
    """Measure transition low-resolution consistency against teacher output."""

    with torch.no_grad():
        teacher_rgb = teacher(z, resolution=prev_resolution, alpha=1.0, stage_mode="stabilize")
    student_rgb = student(z, resolution=resolution, alpha=alpha, stage_mode="transition")
    student_low = F.interpolate(
        student_rgb,
        size=(prev_resolution, prev_resolution),
        mode="bicubic",
        align_corners=False,
    )
    return torch.as_tensor(float(weight), dtype=student_low.dtype, device=student_low.device) * F.l1_loss(
        student_low,
        teacher_rgb,
    )


@dataclass
class ProgressiveContext:
    """Stage metadata consumed by the training loop."""

    resolution: int
    prev_resolution: int | None
    stage_mode: str
    alpha_spec: object
    progressive_state: dict
    start_step: int
    total_steps: int
    stage_seen_images_total: int


@dataclass
class TrainOptions:
    """Scalar training controls and long-lived optional modules for one stage."""

    amp_enabled: bool
    dtype: torch.dtype
    r3_gamma: float
    grad_accum_steps: int
    micro_batch: int
    effective_batch: int
    ema_beta: float
    use_ema: bool
    consistency_enabled: bool
    consistency_weight_spec: object
    teacher_generator: torch.nn.Module | None
    log_interval: int
    ckpt_interval: int
    output_dir: Path


@dataclass
class FidOptions:
    """FID scheduling and generated-image export settings for the loop."""

    enabled: bool
    valid_root: str | Path | None
    num_real: int
    num_fake: int
    subset_seed: int
    interval: int
    batch_size: int
    export_batch_size: int
    dims: int
    device: str
    fail_on_error: bool
    use_ema: bool


def _train_loop(
    *,
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    generator_ema: torch.nn.Module,
    g_optimizer: torch.optim.Optimizer,
    d_optimizer: torch.optim.Optimizer,
    batches,
    logger,
    device: torch.device,
    cfg: dict,
    progressive: ProgressiveContext,
    train_opts: TrainOptions,
    fid_opts: FidOptions,
) -> None:
    """Run one progressive stage training loop.

    Each step performs the discriminator update with exact R1/R2 GP, then the
    generator update and EMA update. Grad accumulation, logging, FID, and
    checkpoint side effects are handled inside the loop.
    """

    resolution = progressive.resolution
    prev_resolution = progressive.prev_resolution
    stage_mode = progressive.stage_mode
    alpha_spec = progressive.alpha_spec
    progressive_state = progressive.progressive_state
    start_step = progressive.start_step
    total_steps = progressive.total_steps
    stage_seen_images_total = progressive.stage_seen_images_total

    amp_enabled = train_opts.amp_enabled
    dtype = train_opts.dtype
    r3_gamma = train_opts.r3_gamma
    grad_accum_steps = train_opts.grad_accum_steps
    micro_batch = train_opts.micro_batch
    effective_batch = train_opts.effective_batch
    ema_beta = train_opts.ema_beta
    use_ema = train_opts.use_ema
    consistency_enabled = train_opts.consistency_enabled
    consistency_weight_spec = train_opts.consistency_weight_spec
    teacher_generator = train_opts.teacher_generator
    log_interval = train_opts.log_interval
    ckpt_interval = train_opts.ckpt_interval
    output_dir = train_opts.output_dir

    fid_enabled = fid_opts.enabled
    valid_root = fid_opts.valid_root
    fid_num_real = fid_opts.num_real
    fid_num_fake = fid_opts.num_fake
    fid_subset_seed = fid_opts.subset_seed
    fid_interval = fid_opts.interval
    fid_batch_size = fid_opts.batch_size
    fid_export_batch_size = fid_opts.export_batch_size
    fid_dims = fid_opts.dims
    fid_device = fid_opts.device
    fid_fail_on_error = fid_opts.fail_on_error
    fid_use_ema = fid_opts.use_ema

    def G_cur(module: torch.nn.Module, z_batch: torch.Tensor, alpha_value: float) -> torch.Tensor:
        return module(z_batch, resolution=resolution, alpha=alpha_value, stage_mode=stage_mode)

    def D_cur(x: torch.Tensor, alpha_value: float) -> torch.Tensor:
        return discriminator(x, resolution=resolution, alpha=alpha_value, stage_mode=stage_mode)

    for step in range(start_step + 1, total_steps + 1):
        step_start = time.perf_counter()
        should_log = step % log_interval == 0 or step == 1
        should_collect_metrics = should_log
        stage_seen_images = stage_seen_images_total
        stage_step = int(stage_seen_images // max(effective_batch, 1))
        stage_kimg = float(stage_seen_images) / 1000.0
        alpha_current = 1.0
        if stage_mode == "transition":
            alpha_current = resolve_schedule_value(alpha_spec, stage_kimg, fallback=1.0)
        consistency_weight_current = 0.0
        if stage_mode == "transition" and consistency_enabled:
            consistency_weight_current = resolve_schedule_value(consistency_weight_spec, stage_kimg, fallback=0.0)

        d_optimizer.zero_grad(set_to_none=True)
        d_adv_total = torch.zeros((), device=device)
        r1_total = torch.zeros((), device=device)
        r2_total = torch.zeros((), device=device)
        real_score_sum = 0.0
        fake_score_sum = 0.0
        for _ in range(grad_accum_steps):
            real_images = prepare_batch(next(batches), device)
            z = generator.sample_z(real_images.size(0), device)
            with torch.no_grad(), autocast_context(device, amp_enabled, dtype):
                fake_seed = G_cur(generator, z, alpha_current)

            real_input = real_images.detach()
            fake_input = fake_seed.detach()
            del fake_seed

            with autocast_context(device, amp_enabled, dtype):
                real_logits = D_cur(real_input, alpha_current)
                fake_logits = D_cur(fake_input, alpha_current)
                adv_loss = discriminator_adversarial_loss(real_logits, fake_logits)
            check_finite_many(step, [("d_adv_loss", adv_loss)])
            (adv_loss / grad_accum_steps).backward()
            gp_scale = (r3_gamma * 0.5) / grad_accum_steps

            r1_loss, r1_param_grads = exact_gp_value_and_param_grads(
                discriminator,
                real_input,
                resolution=resolution,
                alpha=alpha_current,
                stage_mode=stage_mode,
            )
            check_finite_many(step, [("r1_loss", r1_loss)])
            inject_gp_param_grads(discriminator, r1_param_grads, gp_scale)
            del r1_param_grads

            r2_loss, r2_param_grads = exact_gp_value_and_param_grads(
                discriminator,
                fake_input,
                resolution=resolution,
                alpha=alpha_current,
                stage_mode=stage_mode,
            )
            check_finite_many(step, [("r2_loss", r2_loss)])
            inject_gp_param_grads(discriminator, r2_param_grads, gp_scale)
            del r2_param_grads

            d_adv_total = d_adv_total + adv_loss.detach()
            r1_total = r1_total + r1_loss.detach()
            r2_total = r2_total + r2_loss.detach()
            real_score_sum += float(real_logits.detach().mean().cpu()) if should_collect_metrics else 0.0
            fake_score_sum += float(fake_logits.detach().mean().cpu()) if should_collect_metrics else 0.0

        adv_loss_value = d_adv_total / grad_accum_steps
        r1_loss = r1_total / grad_accum_steps
        r2_loss = r2_total / grad_accum_steps
        d_loss = adv_loss_value + (r3_gamma * 0.5) * (r1_loss + r2_loss)
        check_finite_many(
            step,
            [
                ("d_loss", d_loss),
                ("r1_loss", r1_loss),
                ("r2_loss", r2_loss),
            ],
        )

        d_grad_norm = (
            grad_norm(discriminator.parameters()) if should_collect_metrics else 0.0
        )
        d_optimizer.step()

        set_requires_grad(discriminator, False)
        g_optimizer.zero_grad(set_to_none=True)
        g_loss_total = torch.zeros((), device=device)
        consistency_total = torch.zeros((), device=device)
        for _ in range(grad_accum_steps):
            real_images_for_g = prepare_batch(next(batches), device)
            z = generator.sample_z(real_images_for_g.size(0), device)
            consistency_value = torch.zeros((), device=device)
            with autocast_context(device, amp_enabled, dtype):
                fake_images = G_cur(generator, z, alpha_current)
                fake_for_g = fake_images
                real_for_g = real_images_for_g.detach()
                fake_logits_for_g = D_cur(fake_for_g, alpha_current)
                with torch.no_grad():
                    real_logits_for_g = D_cur(real_for_g, alpha_current)
                g_loss = generator_adversarial_loss(real_logits_for_g, fake_logits_for_g)
                consistency_check = False
                if consistency_enabled and teacher_generator is not None and prev_resolution is not None:
                    consistency_value = consistency_loss(
                        teacher_generator,
                        generator,
                        z,
                        resolution,
                        prev_resolution,
                        alpha_current,
                        consistency_weight_current,
                    )
                    g_loss = g_loss + consistency_value
                    consistency_check = True
                g_finite_tensors = [
                    ("fake_logits_for_g", fake_logits_for_g),
                    ("real_logits_for_g", real_logits_for_g),
                    ("g_loss", g_loss),
                ]
                if consistency_check:
                    g_finite_tensors.append(("loss_consistency", consistency_value))
            check_finite_many(step, g_finite_tensors)
            g_loss_total = g_loss_total + g_loss.detach()
            consistency_total = consistency_total + consistency_value.detach()
            (g_loss / grad_accum_steps).backward()

        g_loss = g_loss_total / grad_accum_steps
        consistency_value = consistency_total / grad_accum_steps
        g_grad_norm = grad_norm(generator.parameters()) if should_collect_metrics else 0.0
        g_optimizer.step()
        set_requires_grad(discriminator, True)
        if use_ema:
            update_ema(generator, generator_ema, ema_beta)

        stage_seen_images_total += effective_batch
        stage_seen_images = stage_seen_images_total
        stage_step = int(stage_seen_images // max(effective_batch, 1))
        stage_kimg = float(stage_seen_images) / 1000.0

        step_sec = time.perf_counter() - step_start
        images_per_sec = float(effective_batch * 2) / max(step_sec, 1e-8)
        real_score_value = real_score_sum / grad_accum_steps if should_collect_metrics else 0.0
        fake_score_value = fake_score_sum / grad_accum_steps if should_collect_metrics else 0.0
        metrics = {
            "step": step,
            "loss/d": float(d_loss.detach().cpu()) if should_collect_metrics else 0.0,
            "loss/g": float(g_loss.detach().cpu()) if should_collect_metrics else 0.0,
            "loss/consistency": float(consistency_value.detach().cpu()) if should_collect_metrics else 0.0,
            "logits/d_real": real_score_value,
            "logits/d_fake": fake_score_value,
            "logits/d_margin": real_score_value - fake_score_value,
            "prob/d_real_sigmoid": float(torch.sigmoid(torch.tensor(real_score_value))),
            "prob/d_fake_sigmoid": float(torch.sigmoid(torch.tensor(fake_score_value))),
            "grad_norm/discriminator": d_grad_norm,
            "grad_norm/generator": g_grad_norm,
            "regularization/r1": float(r1_loss.detach().cpu()) if should_collect_metrics else 0.0,
            "regularization/r2": float(r2_loss.detach().cpu()) if should_collect_metrics else 0.0,
            "time/step_sec": step_sec,
            "time/images_per_sec": images_per_sec,
            "progressive/stage_step": stage_step,
            "progressive/stage_seen_images": stage_seen_images,
            "progressive/stage_kimg": stage_kimg,
            "progressive/optimizer_step": stage_step,
            "progressive/alpha": alpha_current,
            "consistency/weight": consistency_weight_current,
        }
        if should_log:
            print(
                f"step {step:06d} "
                f"d_loss={metrics['loss/d']:.4f} "
                f"g_loss={metrics['loss/g']:.4f} "
                f"consistency={metrics['loss/consistency']:.4f} "
                f"d_real={real_score_value:.4f} "
                f"d_fake={fake_score_value:.4f}"
            )
            logger.log(metrics)

        should_fid = (
            fid_enabled
            and valid_root is not None
            and fid_interval > 0
            and (step % fid_interval == 0 or step == total_steps)
        )
        if should_fid:
            fid_start = time.perf_counter()
            fake_dir = None
            try:
                real_subset_dir = prepare_or_reuse_stage_valid_subset(
                    valid_root=valid_root,
                    output_dir=output_dir / "fid" / f"valid_subset_{resolution}_{fid_num_real}",
                    image_size=resolution,
                    num_images=fid_num_real,
                    seed=fid_subset_seed,
                )
                fid_step_dir = output_dir / "fid" / f"step{step:06d}"
                fake_dir = fid_step_dir / "fake"
                if fake_dir.exists():
                    shutil.rmtree(fake_dir)
                fid_generator = generator_ema if fid_use_ema and use_ema else generator
                export_generated_images(
                    fid_generator,
                    lambda count, sample_device: fid_generator.sample_z(count, sample_device),
                    fake_dir,
                    fid_num_fake,
                    fid_export_batch_size,
                    device,
                    lambda module, z_batch: G_cur(module, z_batch, alpha_current).float(),
                )
                fid_value = compute_fid(real_subset_dir, fake_dir, fid_batch_size, fid_device, fid_dims)
                fid_result = {
                    "step": step,
                    "fid/valid_subset": fid_value,
                    "fid/time_sec": time.perf_counter() - fid_start,
                }
                print(
                    f"step {step:06d} fid={fid_result['fid/valid_subset']:.4f} "
                    f"time={fid_result['fid/time_sec']:.2f}s"
                )
                logger.log(fid_result)
            except Exception as exc:
                if fid_fail_on_error:
                    raise
                warnings.warn(f"[fid] skipped step {step}: {exc}")
            finally:
                if fake_dir is not None and fake_dir.exists():
                    shutil.rmtree(fake_dir)
                if device.type == "cuda":
                    gc.collect()
                    torch.cuda.empty_cache()

        if step % ckpt_interval == 0 or step == total_steps:
            wandb_run_id = logger.run_id
            checkpoint_progressive_state = {
                "resolution": resolution,
                "prev_resolution": prev_resolution,
                "stage_mode": stage_mode,
                "stage_id": progressive_state["stage_id"],
                "stage_step": stage_step,
                "stage_seen_images": stage_seen_images,
                "stage_kimg": stage_kimg,
                "optimizer_step": stage_step,
                "micro_batch": micro_batch,
                "grad_accum_steps": grad_accum_steps,
                "effective_batch": effective_batch,
                "alpha_current": alpha_current,
                "consistency_weight_current": consistency_weight_current,
            }
            extra_state = {
                "progressive": checkpoint_progressive_state,
            }
            save_checkpoint(
                output_dir / "model.pth",
                step,
                generator,
                discriminator,
                g_optimizer,
                d_optimizer,
                cfg,
                wandb_run_id=wandb_run_id,
                generator_ema=generator_ema if use_ema else None,
                extra_state=extra_state,
            )


def main() -> None:
    """Single progressive stage training entrypoint."""

    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg.get("seed", 42)))
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
    training_cfg = cfg["training"]
    if bool(training_cfg.get("auto_resume", False)):
        raise ValueError("training.auto_resume is not supported in this progressive runner cycle.")
    model_cfg = cfg.get("model", {})
    stage_cfg = _stage_config(cfg)
    progressive_state = progressive_state_from_config(cfg)
    resolution = int(progressive_state["resolution"])
    prev_resolution = progressive_state["prev_resolution"]
    stage_mode = str(progressive_state["stage_mode"])
    alpha_spec = stage_cfg["alpha_spec"]

    generator = _build_generator(model_cfg).to(device)
    discriminator = _build_discriminator(model_cfg).to(device)
    _validate_generator_discriminator_profiles(generator, discriminator)
    generator_ema = copy.deepcopy(generator).eval()
    set_requires_grad(generator_ema, False)
    sync_ema_buffers(generator, generator_ema)
    total_params = count_parameters(generator)
    trainable_params = count_parameters(generator, trainable_only=True)
    d_params = count_parameters(discriminator)

    print("Generator: project02 generator")
    print(f"Progressive stage: resolution={resolution}, mode={stage_mode}, alpha_spec={alpha_spec}")
    print(f"Generator params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Discriminator params: {d_params:,}")
    if total_params >= 40_000_000:
        warnings.warn(f"Generator parameter count is {total_params:,}, exceeding the v2 target of <40M.")

    if str(training_cfg.get("loss", "default")) not in {"default", "r3gan"}:
        raise ValueError("project02.train only supports the configured adversarial loss.")

    amp_cfg = training_cfg.get("amp", {})
    amp_enabled = bool(amp_cfg.get("enabled", False)) and device.type == "cuda"
    amp_type = str(amp_cfg.get("dtype", "bf16"))
    dtype = amp_dtype(amp_type)
    scaler = make_scaler(amp_enabled and dtype == torch.float16)
    if scaler is not None:
        raise ValueError(
            "etmann GP backend does not support fp16 GradScaler. Use bf16 AMP or fp32."
        )
    print(f"AMP enabled: {amp_enabled} ({amp_type})")

    training_resume_value = training_cfg.get("resume")
    if isinstance(training_resume_value, bool):
        raise TypeError("training.resume must be null or a checkpoint path string.")
    resume_path = args.resume or training_resume_value
    init_from_cfg = _init_from_config(training_cfg)
    init_from_path = init_from_cfg["path"] if init_from_cfg["enabled"] else None
    if resume_path and init_from_cfg["enabled"]:
        raise ValueError("training.resume and training.init_from cannot be used together.")
    grad_accum_steps = int(training_cfg.get("grad_accum_steps", 1))
    if grad_accum_steps < 1:
        raise ValueError("training.grad_accum_steps must be >= 1.")

    consistency_cfg = training_cfg.get("consistency", {})
    consistency_weight_spec = consistency_cfg.get("weight", 0.0)
    consistency_enabled = stage_mode == "transition" and bool(consistency_cfg.get("enabled", False))
    teacher_checkpoint = consistency_cfg.get("teacher_checkpoint") if consistency_enabled else None
    teacher_generator = None
    if consistency_enabled:
        if prev_resolution is None:
            raise ValueError("Consistency requires a previous resolution.")
        if not teacher_checkpoint:
            raise ValueError("Transition consistency requires training.consistency.teacher_checkpoint.")
        teacher_generator = load_teacher_generator(
            model_cfg,
            teacher_checkpoint,
            device,
            bool(consistency_cfg.get("prefer_ema", True)),
            int(prev_resolution),
        )

    data_cfg = dict(cfg["data"])
    data_cfg["image_size"] = resolution
    data_cfg["batch_size"] = int(data_cfg.get("batch_size", training_cfg.get("batch_size", 2)))
    micro_batch = int(data_cfg["batch_size"])
    effective_batch = micro_batch * grad_accum_steps
    loader = build_dataloader(data_cfg, "train")
    batches = repeat_loader(loader)

    if init_from_cfg["enabled"]:
        init_state = read_progressive_state(init_from_path)
        expected_init_resolution = prev_resolution if stage_mode == "transition" else resolution
        if expected_init_resolution is None:
            raise ValueError("training.init_from requires a source resolution.")
        if int(init_state["resolution"]) != int(expected_init_resolution):
            raise ValueError(
                "Init checkpoint resolution does not match expected source stage: "
                f"checkpoint={init_state.get('resolution')!r}, expected={expected_init_resolution!r}"
            )
        load_previous_stage_init(
            init_from_path,
            generator,
            discriminator,
            generator_ema,
            int(expected_init_resolution),
            device,
        )
        sync_ema_buffers(generator, generator_ema)
        if prev_resolution is not None and int(expected_init_resolution) == int(prev_resolution):
            maybe_copy_to_rgb_from_previous(generator, int(prev_resolution), resolution)
            maybe_copy_to_rgb_from_previous(generator_ema, int(prev_resolution), resolution)
        print(
            "Initialized progressive stage from previous checkpoint: "
            f"{init_from_path} (load_resolution={expected_init_resolution})"
        )

    betas = (float(training_cfg.get("beta1", 0.0)), float(training_cfg.get("beta2", 0.99)))
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=float(training_cfg["g_lr"]), betas=betas)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=float(training_cfg["d_lr"]), betas=betas)
    ema_beta = float(training_cfg.get("ema_beta", 0.0))
    use_ema = ema_beta > 0.0

    start_step = 0
    stage_start_step = 0
    restored_stage_seen_images = None
    if resume_path:
        resume_mode = str(training_cfg.get("resume_mode", "strict"))
        if resume_mode != "strict":
            raise ValueError("Progressive trainer only supports strict resume.")
        start_step, extra_state = load_checkpoint(
            resume_path,
            generator,
            discriminator,
            g_optimizer,
            d_optimizer,
            generator_ema=generator_ema,
            map_location=device,
            return_extra_state=True,
        )
        sync_ema_buffers(generator, generator_ema)
        _validate_resume_state(extra_state, progressive_state)
        saved_progressive = extra_state.get("progressive", {})
        stage_start_step = start_step - int(saved_progressive.get("stage_step", start_step))
        restored_stage_seen_images = saved_progressive.get("stage_seen_images")
        print(f"Resumed progressive checkpoint from step {start_step}")

    output_dir = Path(training_cfg.get("output_dir", "checkpoints_progressive"))
    logger = build_logger(cfg, total_params, d_params, resume_path)

    fid_cfg = training_cfg.get("fid", {})
    fid_enabled = bool(fid_cfg.get("enabled", False))
    valid_root = data_cfg.get("valid_root")
    fid_num_real = int(fid_cfg.get("num_real_images", 2048))
    fid_num_fake = int(fid_cfg.get("num_fake_images", 2048))
    fid_subset_seed = int(fid_cfg.get("subset_seed", cfg.get("seed", 42)))
    fid_interval = int(fid_cfg.get("interval", 5000))
    fid_batch_size = int(fid_cfg.get("batch_size", 32))
    fid_export_batch_size = int(fid_cfg.get("export_batch_size", fid_batch_size))
    fid_dims = int(fid_cfg.get("dims", 2048))
    fid_device = str(fid_cfg.get("device", device))
    fid_fail_on_error = bool(fid_cfg.get("fail_on_error", False))
    fid_use_ema = bool(fid_cfg.get("use_ema", use_ema))
    if fid_enabled and valid_root is None:
        warnings.warn("[fid] training.fid.enabled is true, but data.valid_root is not configured; FID will be skipped.")

    total_steps = int(training_cfg["total_steps"])
    log_interval = int(training_cfg.get("log_interval", 100))
    ckpt_interval = int(training_cfg.get("ckpt_interval", 1000))
    r3_gamma = float(training_cfg.get("r3_gamma", training_cfg.get("gamma", 0.05)))
    print(f"R3GAN full R1/R2 gamma={r3_gamma}")
    stage_seen_images_total = (
        int(restored_stage_seen_images)
        if restored_stage_seen_images is not None
        else max(0, start_step - stage_start_step) * effective_batch
    )

    progressive = ProgressiveContext(
        resolution=resolution,
        prev_resolution=prev_resolution,
        stage_mode=stage_mode,
        alpha_spec=alpha_spec,
        progressive_state=progressive_state,
        start_step=start_step,
        total_steps=total_steps,
        stage_seen_images_total=stage_seen_images_total,
    )
    train_opts = TrainOptions(
        amp_enabled=amp_enabled,
        dtype=dtype,
        r3_gamma=r3_gamma,
        grad_accum_steps=grad_accum_steps,
        micro_batch=micro_batch,
        effective_batch=effective_batch,
        ema_beta=ema_beta,
        use_ema=use_ema,
        consistency_enabled=consistency_enabled,
        consistency_weight_spec=consistency_weight_spec,
        teacher_generator=teacher_generator,
        log_interval=log_interval,
        ckpt_interval=ckpt_interval,
        output_dir=output_dir,
    )
    fid_opts = FidOptions(
        enabled=fid_enabled,
        valid_root=valid_root,
        num_real=fid_num_real,
        num_fake=fid_num_fake,
        subset_seed=fid_subset_seed,
        interval=fid_interval,
        batch_size=fid_batch_size,
        export_batch_size=fid_export_batch_size,
        dims=fid_dims,
        device=fid_device,
        fail_on_error=fid_fail_on_error,
        use_ema=fid_use_ema,
    )
    _train_loop(
        generator=generator,
        discriminator=discriminator,
        generator_ema=generator_ema,
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
        batches=batches,
        logger=logger,
        device=device,
        cfg=cfg,
        progressive=progressive,
        train_opts=train_opts,
        fid_opts=fid_opts,
    )

    logger.finish()


if __name__ == "__main__":
    main()
