"""Sampling and visualization utilities."""

import os
import torch
from torchvision.utils import make_grid, save_image


def drift_sample(model, n, device):
    """Generate samples from drifting model (1 forward pass)."""
    z = torch.randn(n, 3, 32, 32, device=device)
    with torch.no_grad():
        x = model(z)
    return x.clamp(-1, 1)


def save_sample_grid(samples, path, nrow=8):
    """Save a grid of samples to disk. Expects samples in [-1, 1]."""
    grid = make_grid(samples, nrow=nrow, normalize=True, value_range=(-1, 1), padding=2)
    save_image(grid, path)


def generate_fid_images(model, n_images, output_dir, device, sample_fn, batch_size=256):
    """Generate images for FID computation and save as individual PNGs.

    Args:
        model: the generative model
        n_images: total number of images to generate
        output_dir: directory to save images
        device: torch device
        sample_fn: callable(model, batch_size, device) -> [B, 3, 32, 32] in [-1, 1]
        batch_size: batch size for generation
    """
    os.makedirs(output_dir, exist_ok=True)
    idx = 0
    while idx < n_images:
        bs = min(batch_size, n_images - idx)
        samples = sample_fn(model, bs, device)
        # Convert to [0, 1] for saving
        samples = (samples + 1) / 2

        for i in range(bs):
            save_image(samples[i], os.path.join(output_dir, f"{idx:05d}.png"))
            idx += 1

        if idx % 1000 == 0:
            print(f"  Generated {idx}/{n_images} images")


def save_cifar10_images(dataset, output_dir, n_images=10000):
    """Save CIFAR-10 images as individual PNGs for FID reference."""
    os.makedirs(output_dir, exist_ok=True)
    for i in range(min(n_images, len(dataset))):
        img, _ = dataset[i]
        # img is already a tensor in [-1, 1] if transformed
        img = (img + 1) / 2
        save_image(img, os.path.join(output_dir, f"{i:05d}.png"))
        if (i + 1) % 1000 == 0:
            print(f"  Saved {i + 1}/{n_images} reference images")
