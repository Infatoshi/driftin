"""Train DDPM on CIFAR-10.

Run: uv run python -m drifting_vs_diffusion.train_ddpm
"""

import os
import time
import csv
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .config import UNetConfig, DDPMConfig
from .models.unet import UNet
from .models.ema import EMA
from .training.ddpm_utils import DDPMSchedule
from .eval.sample import save_sample_grid


def get_cifar10(batch_size, num_workers=4):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # -> [-1, 1]
    ])
    dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    return loader


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Global performance flags
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    cfg = DDPMConfig()
    if args.steps:
        cfg.total_steps = args.steps
    if args.batch_size:
        cfg.batch_size = args.batch_size

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "samples"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)

    # Model
    unet_cfg = UNetConfig()
    model = UNet(
        in_ch=unet_cfg.in_ch, out_ch=unet_cfg.out_ch, base_ch=unet_cfg.base_ch,
        ch_mult=unet_cfg.ch_mult, num_res_blocks=unet_cfg.num_res_blocks,
        attn_resolutions=unet_cfg.attn_resolutions, dropout=unet_cfg.dropout,
        num_heads=unet_cfg.num_heads,
    ).to(memory_format=torch.channels_last).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"UNet parameters: {n_params:,}")

    # Compile
    model = torch.compile(model)
    print("  torch.compile enabled")

    # EMA (on the uncompiled model inside compile wrapper)
    ema = EMA(model, decay=cfg.ema_decay)

    # Fused optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, betas=(0.9, 0.999),
        weight_decay=0.0, fused=True,
    )

    # Noise schedule
    schedule = DDPMSchedule(T=cfg.T, beta_start=cfg.beta_start, beta_end=cfg.beta_end).to(device)

    # Data
    loader = get_cifar10(cfg.batch_size)
    data_iter = iter(loader)

    # Mixed precision
    scaler = torch.amp.GradScaler("cuda")

    # Logging
    log_path = os.path.join(out_dir, "loss_log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["step", "loss", "time_s"])

    print(f"Training DDPM for {cfg.total_steps} steps, batch_size={cfg.batch_size}")
    start_time = time.time()

    for step in range(1, cfg.total_steps + 1):
        # Get batch
        try:
            x_0, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x_0, _ = next(data_iter)
        x_0 = x_0.to(device, memory_format=torch.channels_last)

        # Random timesteps
        t = torch.randint(0, cfg.T, (x_0.shape[0],), device=device)

        # Forward diffusion
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            x_t, noise = schedule.q_sample(x_0, t)
            pred_noise = model(x_t, t)
            loss = F.mse_loss(pred_noise, noise)

        # Backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        # EMA update
        ema.update(model)

        # Logging
        if step % cfg.log_every == 0:
            elapsed = time.time() - start_time
            steps_per_sec = step / elapsed
            eta_h = (cfg.total_steps - step) / steps_per_sec / 3600
            print(
                f"step {step:>7d}/{cfg.total_steps} | "
                f"loss {loss.item():.4f} | "
                f"{steps_per_sec:.1f} it/s | "
                f"ETA {eta_h:.1f}h"
            )
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([step, f"{loss.item():.6f}", f"{elapsed:.1f}"])

        # Sample
        if step % cfg.sample_every == 0:
            model.eval()
            with torch.no_grad():
                # Use EMA for sampling
                samples = schedule.ddim_sample(ema.shadow, 64, (3, 32, 32), device, steps=50)
            save_sample_grid(samples, os.path.join(out_dir, "samples", f"ddpm_step{step:07d}.png"))
            model.train()
            print(f"  Saved sample grid at step {step}")

        # Checkpoint
        if step % cfg.save_every == 0:
            ckpt = {
                "step": step,
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "config": {"unet": unet_cfg, "ddpm": cfg},
            }
            path = os.path.join(out_dir, "checkpoints", f"ddpm_step{step:07d}.pt")
            torch.save(ckpt, path)
            # Also save as 'latest'
            torch.save(ckpt, os.path.join(out_dir, "checkpoints", "ddpm_latest.pt"))
            print(f"  Saved checkpoint at step {step}")

    # Final save
    elapsed = time.time() - start_time
    print(f"\nTraining complete. {cfg.total_steps} steps in {elapsed/3600:.1f} hours")
    torch.save({
        "step": cfg.total_steps,
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "config": {"unet": unet_cfg, "ddpm": cfg},
    }, os.path.join(out_dir, "checkpoints", "ddpm_final.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="outputs/ddpm")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()
    train(args)
