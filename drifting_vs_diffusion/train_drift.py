"""Train drifting model on CIFAR-10.

Run: uv run python -m drifting_vs_diffusion.train_drift
"""

import os
import time
import csv
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights

from .config import UNetConfig, DriftConfig
from .models.unet import UNet
from .models.ema import EMA
from .training.compute_v import drifting_loss
from .eval.sample import drift_sample, save_sample_grid


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


class FeatureExtractor(torch.nn.Module):
    """Frozen ResNet-18 feature extractor for drift loss in feature space.

    Extracts features from layer2 (8x8 spatial, 128-dim for 32x32 input).
    Flattens to [B, D] vector.
    """

    def __init__(self):
        super().__init__()
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        # For 32x32 input: modify first conv to not downsample
        self.conv1 = torch.nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        # Copy what we can from pretrained (center crop the 7x7 kernel)
        with torch.no_grad():
            pretrained_weight = resnet.conv1.weight.data  # [64, 3, 7, 7]
            self.conv1.weight.copy_(pretrained_weight[:, :, 2:5, 2:5])
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        # Skip maxpool for 32x32 (would reduce too much)
        self.layer1 = resnet.layer1  # 32x32 -> 32x32, 64-dim
        self.layer2 = resnet.layer2  # 32x32 -> 16x16, 128-dim

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        """Extract features. Gradient flows through for generated images.

        Args:
            x: [B, 3, 32, 32] in [-1, 1]

        Returns:
            [B, D] flattened features
        """
        # Renormalize from [-1,1] to ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x + 1) / 2  # [-1,1] -> [0,1]
        x = (x - mean) / std

        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu(h)
        h = self.layer1(h)   # [B, 64, 32, 32]
        h = self.layer2(h)   # [B, 128, 16, 16]

        # Global average pool -> [B, 128]
        h = h.mean(dim=[2, 3])
        return h


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

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
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "samples"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)

    # Model (same UNet as DDPM, but used as noise -> image)
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

    # Feature encoder (optional)
    feat_encoder = None
    if cfg.use_feature_encoder:
        feat_encoder = FeatureExtractor().to(memory_format=torch.channels_last).to(device)
        feat_encoder.eval()
        n_feat_params = sum(p.numel() for p in feat_encoder.parameters())
        print(f"Feature encoder parameters: {n_feat_params:,} (frozen)")

    # EMA
    ema = EMA(model, decay=cfg.ema_decay)

    # Fused optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, betas=(0.9, 0.999),
        weight_decay=0.0, fused=True,
    )

    # Data
    loader = get_cifar10(cfg.batch_size)
    data_iter = iter(loader)

    # Logging
    log_path = os.path.join(out_dir, "loss_log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["step", "loss", "time_s"])

    # Mixed precision
    scaler = torch.amp.GradScaler("cuda")

    mode = "feature-space" if cfg.use_feature_encoder else "pixel-space"
    print(f"Training Drifting ({mode}) for {cfg.total_steps} steps")
    print(f"  batch_size={cfg.batch_size}, temps={cfg.temperatures}")
    start_time = time.time()

    for step in range(1, cfg.total_steps + 1):
        # Get real batch
        try:
            real_images, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            real_images, _ = next(data_iter)
        real_images = real_images.to(device, memory_format=torch.channels_last)
        B = real_images.shape[0]

        # Generate images from noise (UNet in bf16)
        z = torch.randn(B, 3, 32, 32, device=device).to(memory_format=torch.channels_last)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            gen_images = model(z)  # t=None -> t=0 internally

        # Extract features or use raw pixels
        # Feature extraction and drift computation stay in float32 for numerical stability
        gen_images_f32 = gen_images.float()
        if feat_encoder is not None:
            with torch.no_grad():
                pos_feats = feat_encoder(real_images)     # [B, D], no grad
            gen_feats = feat_encoder(gen_images_f32)      # [B, D], grad flows to model
        else:
            pos_feats = real_images.flatten(1)             # [B, 3072]
            gen_feats = gen_images_f32.flatten(1)          # [B, 3072]

        # Compute drifting loss (float32 -- softmax/cdist need precision)
        loss = drifting_loss(gen_feats, pos_feats, temps=tuple(cfg.temperatures))

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
            samples = drift_sample(ema.shadow, 64, device)
            save_sample_grid(
                samples,
                os.path.join(out_dir, "samples", f"drift_step{step:07d}.png"),
            )
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
                "config": {"unet": unet_cfg, "drift": cfg},
            }
            path = os.path.join(out_dir, "checkpoints", f"drift_step{step:07d}.pt")
            torch.save(ckpt, path)
            torch.save(ckpt, os.path.join(out_dir, "checkpoints", "drift_latest.pt"))
            print(f"  Saved checkpoint at step {step}")

    elapsed = time.time() - start_time
    print(f"\nTraining complete. {cfg.total_steps} steps in {elapsed/3600:.1f} hours")
    torch.save({
        "step": cfg.total_steps,
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "config": {"unet": unet_cfg, "drift": cfg},
    }, os.path.join(out_dir, "checkpoints", "drift_final.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="outputs/drift")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--use-features", action="store_true",
                        help="Use frozen ResNet-18 feature encoder")
    parser.add_argument("--temps", type=str, default=None,
                        help="Comma-separated temperatures, e.g. '0.02,0.05,0.2'")
    args = parser.parse_args()
    train(args)
