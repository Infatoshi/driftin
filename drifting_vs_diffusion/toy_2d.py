"""Toy 2D experiment: validate drifting on swiss roll and checkerboard.

Uses the exact compute_drift from the paper's reference notebook.
Run: uv run python -m drifting_vs_diffusion.toy_2d
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .training.compute_v import compute_drift


# --- Data distributions ---

def sample_swiss_roll(n, noise=0.5):
    t = 1.5 * math.pi * (1 + 2 * torch.rand(n))
    x = t * torch.cos(t) + noise * torch.randn(n)
    y = t * torch.sin(t) + noise * torch.randn(n)
    return torch.stack([x, y], dim=1) / 8.0  # scale to ~[-3, 3]


def sample_checkerboard(n, size=4.0):
    x = torch.rand(n) * size - size / 2
    y = torch.rand(n) * size - size / 2
    mask = ((torch.floor(x) + torch.floor(y)) % 2 == 0).float()
    # Resample y for rejected points
    while mask.sum() < n:
        idx = mask == 0
        x[idx] = torch.rand(idx.sum()) * size - size / 2
        y[idx] = torch.rand(idx.sum()) * size - size / 2
        mask = ((torch.floor(x) + torch.floor(y)) % 2 == 0).float()
    return torch.stack([x, y], dim=1)


def sample_mog(n, num_modes=8, radius=5.0, std=0.3):
    """Mixture of Gaussians in a ring."""
    angles = torch.linspace(0, 2 * math.pi, num_modes + 1)[:num_modes]
    centers = torch.stack([radius * torch.cos(angles), radius * torch.sin(angles)], dim=1)
    mode_idx = torch.randint(0, num_modes, (n,))
    samples = centers[mode_idx] + std * torch.randn(n, 2)
    return samples


# --- Simple MLP generator ---

class ToyGenerator(nn.Module):
    def __init__(self, hidden=256, depth=4):
        super().__init__()
        layers = [nn.Linear(2, hidden), nn.ReLU()]
        for _ in range(depth - 2):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers += [nn.Linear(hidden, 2)]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


# --- Drifting loss for 2D ---

def toy_drifting_loss(gen, pos, temps=(0.05, 0.1, 0.2)):
    """Simplified drifting loss for 2D toy data."""
    D = gen.shape[-1]
    with torch.no_grad():
        V_total = torch.zeros_like(gen)
        gen_det = gen.detach()
        for temp in temps:
            V = compute_drift(gen_det, pos, temp=temp)
            # Normalize drift
            lam = torch.sqrt((V.float().pow(2).sum(dim=-1) / D).mean()).detach()
            V = V / (lam + 1e-8)
            V_total += V
        target = (gen_det + V_total).detach()
    return F.mse_loss(gen, target)


def run_experiment(
    distribution="mog",
    num_steps=5000,
    N=512,
    lr=1e-3,
    viz_every=500,
    save_dir="outputs/toy_2d",
):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, Distribution: {distribution}")

    # Choose distribution
    sample_fn = {
        "mog": sample_mog,
        "swiss_roll": sample_swiss_roll,
        "checkerboard": sample_checkerboard,
    }[distribution]

    generator = ToyGenerator().to(device)
    optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
    losses = []

    for step in range(1, num_steps + 1):
        z = torch.randn(N, 2, device=device)
        pos = sample_fn(N).to(device)
        gen = generator(z)

        loss = toy_drifting_loss(gen, pos)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if step % 100 == 0:
            print(f"  step {step:>5d} | loss {loss.item():.6f}")

        if step % viz_every == 0 or step == 1:
            with torch.no_grad():
                z_viz = torch.randn(2048, 2, device=device)
                gen_viz = generator(z_viz).cpu().numpy()
                pos_viz = sample_fn(2048).numpy()

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].scatter(pos_viz[:, 0], pos_viz[:, 1], s=1, alpha=0.4, c="blue", label="target")
            axes[0].scatter(gen_viz[:, 0], gen_viz[:, 1], s=1, alpha=0.4, c="red", label="generated")
            lim = max(abs(pos_viz).max(), abs(gen_viz).max()) * 1.2
            axes[0].set_xlim(-lim, lim)
            axes[0].set_ylim(-lim, lim)
            axes[0].set_aspect("equal")
            axes[0].legend()
            axes[0].set_title(f"{distribution} - Step {step}")

            axes[1].plot(losses)
            axes[1].set_xlabel("Step")
            axes[1].set_ylabel("Loss")
            axes[1].set_title("Training Loss")

            path = os.path.join(save_dir, f"{distribution}_step{step:05d}.png")
            fig.savefig(path, dpi=100, bbox_inches="tight")
            plt.close(fig)

    # Final plot
    with torch.no_grad():
        z_final = torch.randn(4096, 2, device=device)
        gen_final = generator(z_final).cpu().numpy()
        pos_final = sample_fn(4096).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(pos_final[:, 0], pos_final[:, 1], s=1, alpha=0.4, c="blue", label="target")
    axes[0].scatter(gen_final[:, 0], gen_final[:, 1], s=1, alpha=0.4, c="red", label="generated")
    lim = max(abs(pos_final).max(), abs(gen_final).max()) * 1.2
    axes[0].set_xlim(-lim, lim)
    axes[0].set_ylim(-lim, lim)
    axes[0].set_aspect("equal")
    axes[0].legend()
    axes[0].set_title(f"{distribution} - Final (Step {num_steps})")
    axes[1].plot(losses)
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Training Loss")
    path = os.path.join(save_dir, f"{distribution}_final.png")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    print(f"Final loss: {losses[-1]:.6f}")
    print(f"Saved to {save_dir}/")
    return losses


if __name__ == "__main__":
    for dist in ["mog", "swiss_roll", "checkerboard"]:
        print(f"\n{'='*60}")
        print(f"Running toy experiment: {dist}")
        print(f"{'='*60}")
        run_experiment(distribution=dist, num_steps=5000)
