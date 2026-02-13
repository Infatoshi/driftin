"""Generate and visualize samples from latent-space drifting model.

Usage:
    python -m drifting_vs_diffusion.eval.sample_latent \
        --checkpoint outputs/drift_latent_256/checkpoints/final.pt \
        --output samples_256.png --n 64
"""

import argparse
import torch
from torchvision.utils import make_grid, save_image


def load_model(ckpt_path, device):
    """Load UNet from checkpoint."""
    from ..models.unet import UNet

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    unet_cfg = ckpt["config"]["unet"]

    model = UNet(
        in_ch=unet_cfg.in_ch, out_ch=unet_cfg.out_ch, base_ch=unet_cfg.base_ch,
        ch_mult=unet_cfg.ch_mult, num_res_blocks=unet_cfg.num_res_blocks,
        attn_resolutions=unet_cfg.attn_resolutions, dropout=0.0,
        num_heads=unet_cfg.num_heads,
    ).to(device)

    # Load EMA weights if available, else model weights
    if "ema" in ckpt:
        model.load_state_dict(ckpt["ema"])
    else:
        model.load_state_dict(ckpt["model"])

    model.eval()
    return model


def load_vae(vae_model, device):
    """Load frozen VAE decoder."""
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(vae_model, torch_dtype=torch.float32).to(device)
    vae.eval()
    return vae


@torch.no_grad()
def generate_samples(model, vae, n, device, latent_ch=4, latent_size=32, chunk_size=8):
    """Generate n 256x256 images: noise -> UNet -> VAE decode."""
    all_samples = []

    for i in range(0, n, chunk_size):
        bs = min(chunk_size, n - i)
        z = torch.randn(bs, latent_ch, latent_size, latent_size, device=device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            latents = model(z)

        images = vae.decode(latents.float()).sample
        all_samples.append(images.clamp(-1, 1).cpu())

    return torch.cat(all_samples, dim=0)


def save_grid(samples, path, nrow=8):
    """Save sample grid. Expects [-1, 1] range."""
    grid = make_grid(samples, nrow=nrow, normalize=True, value_range=(-1, 1), padding=2)
    save_image(grid, path)
    print(f"Saved {samples.shape[0]} samples to {path}")
    print(f"  Resolution: {samples.shape[2]}x{samples.shape[3]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="samples_latent.png")
    parser.add_argument("--n", type=int, default=64, help="Number of samples")
    parser.add_argument("--nrow", type=int, default=8)
    parser.add_argument("--vae", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--compile", action="store_true", help="torch.compile the UNet")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_model(args.checkpoint, device)
    if args.compile:
        model = torch.compile(model)

    print(f"Loading VAE: {args.vae}")
    vae = load_vae(args.vae, device)

    # Infer latent config from model
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    drift_cfg = ckpt["config"].get("drift", None)
    latent_ch = drift_cfg.latent_channels if drift_cfg and hasattr(drift_cfg, 'latent_channels') else 4
    latent_size = drift_cfg.latent_size if drift_cfg and hasattr(drift_cfg, 'latent_size') else 32

    print(f"Generating {args.n} samples (latent {latent_ch}x{latent_size}x{latent_size})...")
    samples = generate_samples(model, vae, args.n, device,
                               latent_ch=latent_ch, latent_size=latent_size)

    save_grid(samples, args.output, nrow=args.nrow)


if __name__ == "__main__":
    main()
