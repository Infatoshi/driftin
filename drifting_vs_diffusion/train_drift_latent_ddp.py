"""Train drifting model in latent space for 256x256 generation.

Pipeline: z ~ N(0,1) -> UNet (32x32x4) -> VAE decode (256x256) -> DINOv3 features -> drift loss
Gradient flows through frozen DINOv3 + frozen VAE decoder back to UNet.

Launch:
    torchrun --nproc_per_node=8 -m drifting_vs_diffusion.train_drift_latent_ddp \
        --data-dir /path/to/imagenet/train \
        --batch-size 16 --steps 50000
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

from .config import UNetConfig, UNetLargeConfig, LatentDriftConfig
from .models.unet import UNet
from .models.ema import EMA
from .training.encoders import build_encoder
from .training.compute_v import compute_drift_multitemp_batched


def setup_ddp():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup():
    dist.destroy_process_group()


def all_gather_flat(tensor):
    """All-gather tensors across all ranks, concatenated along dim 0."""
    tensor = tensor.contiguous()
    world_size = dist.get_world_size()
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=0)


def build_vae_decoder(model_name, device):
    """Load frozen SD VAE decoder with gradient checkpointing."""
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(model_name, torch_dtype=torch.float32).to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    vae.enable_gradient_checkpointing()
    return vae


def get_dataset(data_dir, image_size=256, split="train"):
    """Build ImageNet-style dataset with standard transforms."""
    transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    return dataset


def multires_drift_loss_distributed(gen_groups, pos_groups, temps, rank, world_size):
    """Compute drift loss with globally-gathered features across GPUs.

    gen_groups: list of (features[B, L, C], C_j) -- has gradient
    pos_groups: list of (features[B, L, C], C_j) -- detached
    """
    total_loss = 0.0
    n_groups = len(gen_groups)

    for (gen_feat, C_j), (pos_feat, _) in zip(gen_groups, pos_groups):
        B_local = gen_feat.shape[0]

        with torch.no_grad():
            gen_det = gen_feat.detach()
            pos_det = pos_feat.detach()

            # All-gather across GPUs for better drift estimation
            global_gen = all_gather_flat(gen_det)
            global_pos = all_gather_flat(pos_det)

            # Transpose to [L, N, C] for batched drift computation
            gen_t = global_gen.transpose(0, 1).contiguous()
            pos_t = global_pos.transpose(0, 1).contiguous()

            V = compute_drift_multitemp_batched(gen_t, pos_t, temps=temps)

            # Extract this rank's portion of V
            V_local = V[:, rank * B_local : (rank + 1) * B_local, :]
            target = gen_det + V_local.transpose(0, 1)

        total_loss = total_loss + F.mse_loss(gen_feat, target)

    return total_loss / n_groups


def train(args):
    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    is_main = (rank == 0)

    if is_main:
        print(f"DDP: {world_size} GPUs")

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    cfg = LatentDriftConfig()
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.steps:
        cfg.total_steps = args.steps
    if args.encoder:
        cfg.encoder = args.encoder
    if args.temps:
        cfg.temperatures = [float(t) for t in args.temps.split(",")]
    if args.pool_size:
        cfg.pool_size = args.pool_size
    if args.encoder_size:
        cfg.encoder_input_size = args.encoder_size
    if args.image_size:
        cfg.image_size = args.image_size
    if args.vae:
        cfg.vae_model = args.vae

    out_dir = args.output_dir
    if is_main:
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(os.path.join(out_dir, "samples"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)

    # ---- UNet (generates in latent space) ----
    unet_cfg = UNetLargeConfig() if args.large else UNetConfig()
    # Override channels for latent space
    unet_cfg.in_ch = cfg.latent_channels
    unet_cfg.out_ch = cfg.latent_channels

    num_classes = args.num_classes

    model = UNet(
        in_ch=unet_cfg.in_ch, out_ch=unet_cfg.out_ch, base_ch=unet_cfg.base_ch,
        ch_mult=unet_cfg.ch_mult, num_res_blocks=unet_cfg.num_res_blocks,
        attn_resolutions=unet_cfg.attn_resolutions, dropout=unet_cfg.dropout,
        num_heads=unet_cfg.num_heads, num_classes=num_classes,
    ).to(device)
    model = model.to(memory_format=torch.channels_last)

    if is_main:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"UNet: {n_params:,} params ({n_params/1e6:.1f}M)")
        print(f"  in_ch={unet_cfg.in_ch}, out_ch={unet_cfg.out_ch}")
        print(f"  base_ch={unet_cfg.base_ch}, ch_mult={unet_cfg.ch_mult}")
        print(f"  num_classes={num_classes} ({'conditional' if num_classes > 0 else 'unconditional'})")

    model = torch.compile(model)
    model = DDP(model, device_ids=[local_rank])

    # ---- VAE decoder (frozen, grad flows through for generated images) ----
    if is_main:
        print(f"Loading VAE: {cfg.vae_model}")
    if rank == 0:
        vae = build_vae_decoder(cfg.vae_model, device)
    dist.barrier()
    if rank != 0:
        vae = build_vae_decoder(cfg.vae_model, device)
    dist.barrier()

    if is_main:
        n_vae = sum(p.numel() for p in vae.parameters())
        print(f"  VAE params: {n_vae:,} ({n_vae/1e6:.1f}M, frozen, grad-checkpointed)")

    # ---- Feature encoder (frozen, grad flows through for generated images) ----
    if is_main:
        print(f"Loading encoder: {cfg.encoder}")
    if rank == 0:
        feat_encoder = build_encoder(
            cfg.encoder, pool_size=cfg.pool_size, input_size=cfg.encoder_input_size,
        ).to(device)
    dist.barrier()
    if rank != 0:
        feat_encoder = build_encoder(
            cfg.encoder, pool_size=cfg.pool_size, input_size=cfg.encoder_input_size,
        ).to(device)
    dist.barrier()
    feat_encoder.eval()

    if is_main:
        n_enc = sum(p.numel() for p in feat_encoder.parameters())
        print(f"  Encoder params: {n_enc:,} ({n_enc/1e6:.1f}M, frozen)")
        print(f"  Input size: {cfg.encoder_input_size}x{cfg.encoder_input_size}")

    # ---- Dataset ----
    if is_main:
        print(f"Loading dataset from: {args.data_dir}")

    dataset = get_dataset(args.data_dir, image_size=cfg.image_size)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )

    if is_main:
        print(f"  Dataset size: {len(dataset):,}")
        global_bs = cfg.batch_size * world_size
        steps_per_epoch = len(dataset) // global_bs
        print(f"  Per-GPU batch={cfg.batch_size}, global batch={global_bs}")
        print(f"  Steps per epoch: {steps_per_epoch}")

    # ---- Optimizer ----
    ema = EMA(model.module, decay=cfg.ema_decay)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, betas=(0.9, 0.999),
        weight_decay=0.01, fused=True,
    )

    scaler = torch.amp.GradScaler("cuda")

    # ---- Logging ----
    log_path = os.path.join(out_dir, "loss_log.csv")
    if is_main:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["step", "loss", "time_s", "images_per_sec", "ms_per_step"])
        print(f"\nTraining latent drifting for {cfg.total_steps} steps")
        print(f"  temps={cfg.temperatures}, pool_size={cfg.pool_size}")
        print(f"  image_size={cfg.image_size}, latent_size={cfg.latent_size}")

    # ---- Resume from checkpoint ----
    start_step = 0
    if args.resume:
        ckpt_path = args.resume
        if is_main:
            print(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.module.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_step = ckpt.get("step", 0)
        if is_main:
            print(f"  Resumed at step {start_step}")

    # ---- Training loop ----
    start_time = time.time()
    epoch = 0
    data_iter = iter(loader)

    for step in range(start_step + 1, cfg.total_steps + 1):
        try:
            real_images, labels = next(data_iter)
        except StopIteration:
            epoch += 1
            sampler.set_epoch(epoch)
            data_iter = iter(loader)
            real_images, labels = next(data_iter)

        real_images = real_images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True) if num_classes > 0 else None
        B = real_images.shape[0]

        # --- Compute real features (no grad, frozen encoder) ---
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            real_groups_raw = feat_encoder(real_images)
        real_groups = [(f.float(), c) for f, c in real_groups_raw]

        # --- Generate in latent space ---
        z = torch.randn(B, cfg.latent_channels, cfg.latent_size, cfg.latent_size,
                        device=device).to(memory_format=torch.channels_last)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            gen_latents = model(z, class_labels=labels)

        # --- VAE decode to images (grad flows through) ---
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            gen_images = vae.decode(gen_latents.float()).sample  # [B, 3, 256, 256]

        # --- Extract features from generated images (grad flows through) ---
        gen_images_f32 = gen_images.float()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            gen_groups_raw = feat_encoder(gen_images_f32)
        gen_groups = [(f.float(), c) for f, c in gen_groups_raw]

        # --- Drift loss with all-gather ---
        loss = multires_drift_loss_distributed(
            gen_groups, real_groups,
            temps=tuple(cfg.temperatures),
            rank=rank, world_size=world_size,
        )

        # --- Backward + optimize ---
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        ema.update(model.module)

        # --- Logging ---
        if step % cfg.log_every == 0 and is_main:
            elapsed = time.time() - start_time
            steps_done = step - start_step
            steps_per_sec = steps_done / elapsed
            imgs_per_sec = steps_per_sec * cfg.batch_size * world_size
            ms_per_step = 1000.0 / steps_per_sec if steps_per_sec > 0 else 0
            eta_h = (cfg.total_steps - step) / steps_per_sec / 3600 if steps_per_sec > 0 else 0
            print(
                f"step {step:>7d}/{cfg.total_steps} | "
                f"loss {loss.item():.4f} | "
                f"{ms_per_step:.0f} ms/step | "
                f"{imgs_per_sec:.0f} img/s | "
                f"ETA {eta_h:.1f}h"
            )
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    step, f"{loss.item():.6f}", f"{elapsed:.1f}",
                    f"{imgs_per_sec:.0f}", f"{ms_per_step:.1f}",
                ])

        # --- Sample ---
        if step % cfg.sample_every == 0 and is_main:
            model.eval()
            with torch.no_grad():
                sample_z = torch.randn(
                    64, cfg.latent_channels, cfg.latent_size, cfg.latent_size,
                    device=device,
                ).to(memory_format=torch.channels_last)
                sample_labels = None
                if num_classes > 0:
                    sample_labels = torch.randint(0, num_classes, (64,), device=device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    sample_latents = ema.shadow(sample_z, class_labels=sample_labels)
                # Decode in chunks to avoid OOM
                chunk_size = 8
                decoded_chunks = []
                for i in range(0, sample_latents.shape[0], chunk_size):
                    chunk = sample_latents[i:i+chunk_size].float()
                    decoded = vae.decode(chunk).sample
                    decoded_chunks.append(decoded.clamp(-1, 1))
                samples = torch.cat(decoded_chunks, dim=0)

            from .eval.sample import save_sample_grid
            save_sample_grid(
                samples,
                os.path.join(out_dir, "samples", f"latent_drift_step{step:07d}.png"),
                nrow=8,
            )
            model.train()
            print(f"  Saved sample grid at step {step}")

        # --- Checkpoint ---
        if step % cfg.save_every == 0 and is_main:
            ckpt = {
                "step": step,
                "model": model.module.state_dict(),
                "ema": ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "config": {"unet": unet_cfg, "drift": cfg, "num_classes": num_classes},
            }
            path = os.path.join(out_dir, "checkpoints", f"latent_drift_step{step:07d}.pt")
            torch.save(ckpt, path)
            torch.save(ckpt, os.path.join(out_dir, "checkpoints", "latest.pt"))
            print(f"  Saved checkpoint at step {step}")

    # ---- Final save ----
    if is_main:
        elapsed = time.time() - start_time
        print(f"\nTraining complete. {cfg.total_steps} steps in {elapsed/3600:.1f} hours")
        torch.save({
            "step": cfg.total_steps,
            "model": model.module.state_dict(),
            "ema": ema.state_dict(),
            "config": {"unet": unet_cfg, "drift": cfg},
        }, os.path.join(out_dir, "checkpoints", "final.pt"))

    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latent-space drifting on ImageNet-256")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to ImageNet train directory (ImageFolder layout)")
    parser.add_argument("--output-dir", type=str, default="outputs/drift_latent_256")
    parser.add_argument("--encoder", type=str, default="dinov3",
                        choices=["dinov2-multires", "convnextv2", "mocov2",
                                 "dinov3", "dinov3-l", "dinov3-h+",
                                 "aimv2-l", "aimv2-h", "radio",
                                 "eva02", "siglip2", "clip"])
    parser.add_argument("--encoder-size", type=int, default=None,
                        help="Encoder input resolution (default 224)")
    parser.add_argument("--image-size", type=int, default=None,
                        help="Training image resolution (default 256)")
    parser.add_argument("--vae", type=str, default=None,
                        help="VAE model name (default: stabilityai/sd-vae-ft-mse)")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Per-GPU batch size")
    parser.add_argument("--temps", type=str, default=None,
                        help="Comma-separated temperatures")
    parser.add_argument("--pool-size", type=int, default=None)
    parser.add_argument("--large", action="store_true",
                        help="Use large UNet config (152M params)")
    parser.add_argument("--num-classes", type=int, default=0,
                        help="Number of classes for conditioning (0 = unconditional)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()
    train(args)
