"""Train DDPM on CIFAR-10 with DDP on multi-GPU.

Launch: torchrun --nproc_per_node=8 -m drifting_vs_diffusion.train_ddpm_ddp
"""

import os
import time
import csv
import argparse
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

from .config import UNetConfig, UNetLargeConfig, DDPMConfig
from .models.unet import UNet
from .models.ema import EMA
from .training.ddpm_utils import DDPMSchedule
from .eval.sample import save_sample_grid


def setup_ddp():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup():
    dist.destroy_process_group()


def get_cifar10(batch_size, world_size, rank, num_workers=4):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    # Rank 0 downloads, others wait
    if rank == 0:
        datasets.CIFAR10(root="./data", train=True, download=True)
    dist.barrier()
    dataset = datasets.CIFAR10(root="./data", train=True, download=False, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(
        dataset, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    return loader, sampler


def train(args):
    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    is_main = (rank == 0)

    if is_main:
        print(f"DDP: {world_size} GPUs")

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
    if is_main:
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(os.path.join(out_dir, "samples"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)

    # Model
    unet_cfg = UNetLargeConfig() if args.large else UNetConfig()
    model = UNet(
        in_ch=unet_cfg.in_ch, out_ch=unet_cfg.out_ch, base_ch=unet_cfg.base_ch,
        ch_mult=unet_cfg.ch_mult, num_res_blocks=unet_cfg.num_res_blocks,
        attn_resolutions=unet_cfg.attn_resolutions, dropout=unet_cfg.dropout,
        num_heads=unet_cfg.num_heads,
    ).to(memory_format=torch.channels_last).to(device)

    if is_main:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"UNet parameters: {n_params:,}")

    # Compile then wrap in DDP
    model = torch.compile(model)
    model = DDP(model, device_ids=[local_rank])
    if is_main:
        print("  torch.compile + DDP enabled")

    # EMA (on the underlying compiled model, rank 0 only for sampling)
    ema = EMA(model.module, decay=cfg.ema_decay)

    # Fused optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, betas=(0.9, 0.999),
        weight_decay=0.0, fused=True,
    )

    # Noise schedule
    schedule = DDPMSchedule(T=cfg.T, beta_start=cfg.beta_start, beta_end=cfg.beta_end).to(device)

    # Data
    loader, sampler = get_cifar10(cfg.batch_size, world_size, rank)

    # Mixed precision
    scaler = torch.amp.GradScaler("cuda")

    # Logging
    log_path = os.path.join(out_dir, "loss_log.csv")
    if is_main:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["step", "loss", "time_s", "images_per_sec"])
        global_bs = cfg.batch_size * world_size
        print(f"Training DDPM for {cfg.total_steps} steps")
        print(f"  per-GPU batch={cfg.batch_size}, global batch={global_bs}")

    start_time = time.time()
    epoch = 0
    data_iter = iter(loader)

    for step in range(1, cfg.total_steps + 1):
        try:
            x_0, _ = next(data_iter)
        except StopIteration:
            epoch += 1
            sampler.set_epoch(epoch)
            data_iter = iter(loader)
            x_0, _ = next(data_iter)
        x_0 = x_0.to(device, memory_format=torch.channels_last)

        t = torch.randint(0, cfg.T, (x_0.shape[0],), device=device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            x_t, noise = schedule.q_sample(x_0, t)
            pred_noise = model(x_t, t)
            loss = F.mse_loss(pred_noise, noise)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        ema.update(model.module)

        if step % cfg.log_every == 0 and is_main:
            elapsed = time.time() - start_time
            steps_per_sec = step / elapsed
            imgs_per_sec = steps_per_sec * cfg.batch_size * world_size
            eta_h = (cfg.total_steps - step) / steps_per_sec / 3600
            print(
                f"step {step:>7d}/{cfg.total_steps} | "
                f"loss {loss.item():.4f} | "
                f"{steps_per_sec:.1f} it/s ({imgs_per_sec:.0f} img/s) | "
                f"ETA {eta_h:.1f}h"
            )
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([step, f"{loss.item():.6f}", f"{elapsed:.1f}", f"{imgs_per_sec:.0f}"])

        if step % cfg.sample_every == 0 and is_main:
            model.eval()
            with torch.no_grad():
                samples = schedule.ddim_sample(ema.shadow, 64, (3, 32, 32), device, steps=50)
            save_sample_grid(samples, os.path.join(out_dir, "samples", f"ddpm_step{step:07d}.png"))
            model.train()
            print(f"  Saved sample grid at step {step}")

        if step % cfg.save_every == 0 and is_main:
            ckpt = {
                "step": step,
                "model": model.module.state_dict(),
                "ema": ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "config": {"unet": unet_cfg, "ddpm": cfg},
            }
            path = os.path.join(out_dir, "checkpoints", f"ddpm_step{step:07d}.pt")
            torch.save(ckpt, path)
            torch.save(ckpt, os.path.join(out_dir, "checkpoints", "ddpm_latest.pt"))
            print(f"  Saved checkpoint at step {step}")

    if is_main:
        elapsed = time.time() - start_time
        print(f"\nTraining complete. {cfg.total_steps} steps in {elapsed/3600:.1f} hours")
        torch.save({
            "step": cfg.total_steps,
            "model": model.module.state_dict(),
            "ema": ema.state_dict(),
            "config": {"unet": unet_cfg, "ddpm": cfg},
        }, os.path.join(out_dir, "checkpoints", "ddpm_final.pt"))

    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="outputs/ddpm")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Per-GPU batch size (global = this * num_gpus)")
    parser.add_argument("--large", action="store_true",
                        help="Use large UNet (~152M params) instead of small (~38M)")
    args = parser.parse_args()
    train(args)
