"""Train drifting model on CIFAR-10 with DDP on multi-GPU.

Key: all_gather features across GPUs before computing drift field,
so the drift signal comes from the full global batch.

Launch: torchrun --nproc_per_node=8 -m drifting_vs_diffusion.train_drift_ddp
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
from torchvision.models import resnet18, ResNet18_Weights

from .config import UNetConfig, UNetLargeConfig, DriftConfig
from .models.unet import UNet
from .models.ema import EMA
from .training.compute_v import compute_drift_multitemp
from .eval.sample import drift_sample, save_sample_grid


def setup_ddp():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup():
    dist.destroy_process_group()


class FeatureExtractor(torch.nn.Module):
    """Frozen ResNet-18 feature extractor for drift loss in feature space."""

    def __init__(self):
        super().__init__()
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.conv1 = torch.nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        with torch.no_grad():
            pretrained_weight = resnet.conv1.weight.data
            self.conv1.weight.copy_(pretrained_weight[:, :, 2:5, 2:5])
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x + 1) / 2
        x = (x - mean) / std
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu(h)
        h = self.layer1(h)
        h = self.layer2(h)
        h = h.mean(dim=[2, 3])
        return h


class DINOv2Extractor(torch.nn.Module):
    """Frozen DINOv2 ViT-B/14 feature extractor.

    Resizes 32x32 -> 224x224, extracts CLS token -> 768D.
    """

    def __init__(self):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        # Register ImageNet normalization as buffers
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        """
        Args:
            x: [B, 3, 32, 32] in [-1, 1]
        Returns:
            [B, 768] CLS token features
        """
        # [-1,1] -> [0,1] -> ImageNet normalization
        x = (x + 1) / 2
        x = (x - self.mean) / self.std
        # Resize to 224x224 for DINOv2
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        # Extract CLS token
        features = self.model(x)  # [B, 768]
        return features


def all_gather_flat(tensor):
    """All-gather tensors across all ranks. Returns concatenated [world_size * N, D]."""
    tensor = tensor.contiguous()
    world_size = dist.get_world_size()
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=0)


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


def drifting_loss_distributed(gen_feats, pos_feats, temps, rank, world_size):
    """Compute drifting loss with globally-gathered features.

    1. All-gather gen_feats and pos_feats across GPUs (detached, for drift computation)
    2. Compute drift V using global batch
    3. Extract local V slice
    4. Loss = MSE(local_gen_feats, local_gen_feats.detach() + V_local)
    """
    B_local = gen_feats.shape[0]

    with torch.no_grad():
        gen_det = gen_feats.detach()
        pos_det = pos_feats.detach()

        # All-gather to get global batch
        global_gen = all_gather_flat(gen_det)   # [world_size * B, D]
        global_pos = all_gather_flat(pos_det)   # [world_size * B, D]

        # Compute drift on global batch
        V_global = compute_drift_multitemp(global_gen, global_pos, temps=temps)

        # Extract local slice
        V_local = V_global[rank * B_local : (rank + 1) * B_local]

        target = gen_det + V_local

    return F.mse_loss(gen_feats, target)


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

    cfg = DriftConfig()
    if args.steps:
        cfg.total_steps = args.steps
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.use_features:
        cfg.use_feature_encoder = True
    if args.temps:
        cfg.temperatures = [float(t) for t in args.temps.split(",")]

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

    # Compile then DDP
    model = torch.compile(model)
    model = DDP(model, device_ids=[local_rank])
    if is_main:
        print("  torch.compile + DDP enabled")

    # Feature encoder (optional)
    feat_encoder = None
    encoder_name = "none"
    if args.encoder == "dinov2":
        if rank == 0:
            # Rank 0 downloads model, others wait
            feat_encoder = DINOv2Extractor().to(device)
        dist.barrier()
        if rank != 0:
            feat_encoder = DINOv2Extractor().to(device)
        dist.barrier()
        feat_encoder.eval()
        cfg.use_feature_encoder = True
        encoder_name = "dinov2-vitb14 (768D)"
        if is_main:
            n_feat_params = sum(p.numel() for p in feat_encoder.parameters())
            print(f"Feature encoder: DINOv2 ViT-B/14, {n_feat_params:,} params (frozen)")
    elif cfg.use_feature_encoder:
        feat_encoder = FeatureExtractor().to(memory_format=torch.channels_last).to(device)
        feat_encoder.eval()
        encoder_name = "resnet18-layer2 (128D)"
        if is_main:
            n_feat_params = sum(p.numel() for p in feat_encoder.parameters())
            print(f"Feature encoder: ResNet-18, {n_feat_params:,} params (frozen)")

    # EMA
    ema = EMA(model.module, decay=cfg.ema_decay)

    # Fused optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, betas=(0.9, 0.999),
        weight_decay=0.0, fused=True,
    )

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
        mode = encoder_name if cfg.use_feature_encoder else "pixel-space"
        print(f"Training Drifting ({mode}) for {cfg.total_steps} steps")
        print(f"  per-GPU batch={cfg.batch_size}, global batch={global_bs}")
        print(f"  temps={cfg.temperatures}")

    start_time = time.time()
    epoch = 0
    data_iter = iter(loader)

    for step in range(1, cfg.total_steps + 1):
        try:
            real_images, _ = next(data_iter)
        except StopIteration:
            epoch += 1
            sampler.set_epoch(epoch)
            data_iter = iter(loader)
            real_images, _ = next(data_iter)
        real_images = real_images.to(device, memory_format=torch.channels_last)
        B = real_images.shape[0]

        # Generate images from noise (UNet in bf16)
        z = torch.randn(B, 3, 32, 32, device=device).to(memory_format=torch.channels_last)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            gen_images = model(z)

        # Feature extraction in float32
        gen_images_f32 = gen_images.float()
        if feat_encoder is not None:
            with torch.no_grad():
                pos_feats = feat_encoder(real_images)
            gen_feats = feat_encoder(gen_images_f32)
        else:
            pos_feats = real_images.flatten(1)
            gen_feats = gen_images_f32.flatten(1)

        # Compute drifting loss with global all-gather
        loss = drifting_loss_distributed(
            gen_feats, pos_feats,
            temps=tuple(cfg.temperatures),
            rank=rank, world_size=world_size,
        )

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
            samples = drift_sample(ema.shadow, 64, device)
            save_sample_grid(
                samples,
                os.path.join(out_dir, "samples", f"drift_step{step:07d}.png"),
            )
            model.train()
            print(f"  Saved sample grid at step {step}")

        if step % cfg.save_every == 0 and is_main:
            ckpt = {
                "step": step,
                "model": model.module.state_dict(),
                "ema": ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "config": {"unet": unet_cfg, "drift": cfg},
            }
            path = os.path.join(out_dir, "checkpoints", f"drift_step{step:07d}.pt")
            torch.save(ckpt, path)
            torch.save(ckpt, os.path.join(out_dir, "checkpoints", "drift_latest.pt"))
            print(f"  Saved checkpoint at step {step}")

    if is_main:
        elapsed = time.time() - start_time
        print(f"\nTraining complete. {cfg.total_steps} steps in {elapsed/3600:.1f} hours")
        torch.save({
            "step": cfg.total_steps,
            "model": model.module.state_dict(),
            "ema": ema.state_dict(),
            "config": {"unet": unet_cfg, "drift": cfg},
        }, os.path.join(out_dir, "checkpoints", "drift_final.pt"))

    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="outputs/drift")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Per-GPU batch size (global = this * num_gpus)")
    parser.add_argument("--use-features", action="store_true",
                        help="Use frozen ResNet-18 feature encoder")
    parser.add_argument("--temps", type=str, default=None,
                        help="Comma-separated temperatures, e.g. '0.02,0.05,0.2'")
    parser.add_argument("--large", action="store_true",
                        help="Use large UNet (~152M params) instead of small (~38M)")
    parser.add_argument("--encoder", type=str, default=None,
                        choices=["resnet18", "dinov2"],
                        help="Feature encoder: resnet18 (128D) or dinov2 (768D)")
    args = parser.parse_args()
    # --encoder implies --use-features
    if args.encoder:
        args.use_features = True
    train(args)
