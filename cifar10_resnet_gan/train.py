import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torchvision.utils import save_image

from data.cifar10 import get_cifar10_loader
from models.discriminator import build_discriminator
from models.generator import build_generator


def load_config(path: str = "config/cifar10_gan.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Train CIFAR-10 ResNet GAN.")
    parser.add_argument("--config", type=str, default="config/cifar10_gan.yaml")
    return parser.parse_args()


def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch


def d_loss_fn(real_logits, fake_logits):
    # L_D = softplus(fake) + softplus(-real)  (non-saturating logistic loss)
    return F.softplus(fake_logits).mean() + F.softplus(-real_logits).mean()


def g_loss_fn(fake_logits):
    # L_G = softplus(-fake)
    return F.softplus(-fake_logits).mean()


def set_requires_grad(module, value):
    for p in module.parameters():
        p.requires_grad_(value)


def grad_norm(module):
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += p.grad.detach().data.norm(2).item() ** 2
    return total ** 0.5


def select_device(preferred: str) -> torch.device:
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def init_wandb(cfg: dict, config_path: str):
    wandb_cfg = cfg.get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        return None

    import wandb

    run = wandb.init(
        project=wandb_cfg["project"],
        entity=wandb_cfg.get("entity"),
        name=wandb_cfg.get("run_name"),
        mode=wandb_cfg.get("mode", "online"),
        tags=wandb_cfg.get("tags"),
        config={**cfg, "config_path": config_path},
    )
    return run


def main():
    args = parse_args()
    cfg = load_config(args.config)

    device = select_device(cfg["device"])
    print(f"Using device: {device}")
    torch.manual_seed(cfg["seed"])
    wandb_run = init_wandb(cfg, args.config)

    # --- Data ---
    loader = get_cifar10_loader(
        root=cfg["data"]["root"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"] and torch.cuda.is_available(),
    )
    data_iter = infinite_loader(loader)

    # --- Models ---
    G = build_generator(
        z_dim=cfg["model"]["z_dim"],
        base_channels=cfg["model"]["g_base_channels"],
    ).to(device)
    D = build_discriminator(base_channels=cfg["model"]["d_base_channels"]).to(device)

    g_params = sum(p.numel() for p in G.parameters())
    d_params = sum(p.numel() for p in D.parameters())
    print(f"Generator params: {g_params:,}")
    print(f"Discriminator params: {d_params:,}")

    # --- Optimizers ---
    betas = (cfg["training"]["beta1"], cfg["training"]["beta2"])
    g_optim = torch.optim.Adam(G.parameters(), lr=cfg["training"]["g_lr"], betas=betas)
    d_optim = torch.optim.Adam(D.parameters(), lr=cfg["training"]["d_lr"], betas=betas)

    # --- Setup output dirs and fixed sampling noise ---
    sample_dir = Path(cfg["training"]["sample_dir"])
    sample_dir.mkdir(parents=True, exist_ok=True)
    fixed_z = torch.randn(cfg["training"]["num_sample_images"], G.z_dim, device=device)

    total_iters = cfg["training"]["total_iters"]
    n_critic = cfg["training"]["n_critic"]
    log_interval = cfg["training"]["log_interval"]
    sample_interval = cfg["training"]["sample_interval"]
    ckpt_interval = cfg["training"]["ckpt_interval"]
    batch_size = cfg["training"]["batch_size"]

    G.train()
    D.train()

    d_loss_sum = 0.0
    g_loss_sum = 0.0
    d_real_prob_sum = 0.0
    d_fake_prob_sum = 0.0
    d_grad_norm_sum = 0.0
    g_grad_norm_sum = 0.0
    d_steps = 0
    g_steps = 0

    for it in range(1, total_iters + 1):
        # --- Train D for n_critic steps ---
        for _ in range(n_critic):
            real = next(data_iter).to(device, non_blocking=True)

            with torch.no_grad():
                z = G.sample_z(real.size(0), device)
                fake = G(z)

            real_logits = D(real)
            fake_logits = D(fake)
            d_loss = d_loss_fn(real_logits, fake_logits)

            d_optim.zero_grad(set_to_none=True)
            d_loss.backward()
            d_grad = grad_norm(D)
            d_optim.step()

            d_loss_sum += d_loss.item()
            d_real_prob_sum += torch.sigmoid(real_logits).mean().item()
            d_fake_prob_sum += torch.sigmoid(fake_logits).mean().item()
            d_grad_norm_sum += d_grad
            d_steps += 1

        # --- Train G once (freeze D so backward skips its parameter grads) ---
        set_requires_grad(D, False)

        z = G.sample_z(batch_size, device)
        fake = G(z)
        fake_logits = D(fake)
        g_loss = g_loss_fn(fake_logits)

        g_optim.zero_grad(set_to_none=True)
        g_loss.backward()
        g_grad = grad_norm(G)
        g_optim.step()

        set_requires_grad(D, True)

        g_loss_sum += g_loss.item()
        g_grad_norm_sum += g_grad
        g_steps += 1

        # --- Logging ---
        if it % log_interval == 0:
            metrics = {
                "loss/D": d_loss_sum / d_steps,
                "loss/G": g_loss_sum / g_steps,
                "output/D_real_sigmoid": d_real_prob_sum / d_steps,
                "output/D_fake_sigmoid": d_fake_prob_sum / d_steps,
                "grad_norm/D": d_grad_norm_sum / d_steps,
                "grad_norm/G": g_grad_norm_sum / g_steps,
                "training/n_critic": n_critic,
            }
            print(
                f"Iter {it:6d}/{total_iters} | "
                f"D loss: {metrics['loss/D']:.4f} | "
                f"G loss: {metrics['loss/G']:.4f} | "
                f"D(x): {metrics['output/D_real_sigmoid']:.4f} | "
                f"D(G(z)): {metrics['output/D_fake_sigmoid']:.4f} | "
                f"D grad: {metrics['grad_norm/D']:.4f} | "
                f"G grad: {metrics['grad_norm/G']:.4f}"
            )
            if wandb_run is not None:
                wandb_run.log(metrics, step=it)

            d_loss_sum = g_loss_sum = 0.0
            d_real_prob_sum = d_fake_prob_sum = 0.0
            d_grad_norm_sum = g_grad_norm_sum = 0.0
            d_steps = g_steps = 0

        # --- Save sample grid ---
        if it % sample_interval == 0:
            G.eval()
            with torch.no_grad():
                samples = G(fixed_z)
            G.train()
            grid_path = sample_dir / f"samples_iter{it:06d}.png"
            save_image(samples, grid_path, nrow=8, normalize=True, value_range=(-1, 1))
            print(f"  saved samples -> {grid_path}")
            if wandb_run is not None:
                import wandb

                wandb_run.log({"generated/images": wandb.Image(str(grid_path))}, step=it)

        # --- Save checkpoint ---
        if it % ckpt_interval == 0:
            torch.save(
                {
                    "iter": it,
                    "generator": G.state_dict(),
                    "discriminator": D.state_dict(),
                    "g_optim": g_optim.state_dict(),
                    "d_optim": d_optim.state_dict(),
                    "config": cfg,
                },
                cfg["training"]["ckpt_path"],
            )
            print(f"  saved checkpoint -> {cfg['training']['ckpt_path']}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
