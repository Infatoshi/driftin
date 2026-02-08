"""Generate side-by-side comparison after both models are trained.

Run: uv run python -m drifting_vs_diffusion.compare
"""

import os
import time
import argparse
import torch
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image

from .config import UNetConfig, DDPMConfig
from .models.unet import UNet
from .training.ddpm_utils import DDPMSchedule
from .eval.sample import drift_sample, save_sample_grid, generate_fid_images, save_cifar10_images
from .eval.fid import compute_fid


def load_model(ckpt_path, device):
    """Load a trained model from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    unet_cfg = UNetConfig()
    model = UNet(
        in_ch=unet_cfg.in_ch, out_ch=unet_cfg.out_ch, base_ch=unet_cfg.base_ch,
        ch_mult=unet_cfg.ch_mult, num_res_blocks=unet_cfg.num_res_blocks,
        attn_resolutions=unet_cfg.attn_resolutions, dropout=0.0,  # no dropout at eval
        num_heads=unet_cfg.num_heads,
    ).to(device)

    # Load EMA weights (strip _orig_mod. prefix from torch.compile if present)
    state_dict = ckpt.get("ema", ckpt["model"])
    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k.replace("_orig_mod.", "")] = v
    model.load_state_dict(cleaned)
    model.eval()
    return model


def benchmark_sampling(model, sample_fn, n=1000, device="cuda", warmup=10, **kwargs):
    """Measure sampling throughput."""
    # Warmup
    for _ in range(warmup):
        sample_fn(model, 16, device, **kwargs)
    torch.cuda.synchronize()

    start = time.time()
    sample_fn(model, n, device, **kwargs)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    return n / elapsed  # images/sec


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # Load models
    print("Loading models...")
    drift_model = load_model(args.drift_ckpt, device)
    ddpm_model = load_model(args.ddpm_ckpt, device)

    schedule = DDPMSchedule(T=1000).to(device)

    # --- Visual comparison ---
    print("\nGenerating sample grids...")
    drift_samples = drift_sample(drift_model, 64, device)
    save_sample_grid(drift_samples, os.path.join(out_dir, "drift_samples_1nfe.png"))

    ddpm_50 = schedule.ddim_sample(ddpm_model, 64, (3, 32, 32), device, steps=50)
    save_sample_grid(ddpm_50, os.path.join(out_dir, "ddpm_samples_ddim50.png"))

    ddpm_100 = schedule.ddim_sample(ddpm_model, 64, (3, 32, 32), device, steps=100)
    save_sample_grid(ddpm_100, os.path.join(out_dir, "ddpm_samples_ddim100.png"))

    ddpm_1000 = schedule.ddpm_sample(ddpm_model, 64, (3, 32, 32), device)
    save_sample_grid(ddpm_1000, os.path.join(out_dir, "ddpm_samples_1000.png"))
    print("  Saved sample grids")

    # --- Side-by-side grid ---
    combined = torch.cat([drift_samples[:32], ddpm_50[:32]], dim=0)
    grid = make_grid(combined, nrow=8, normalize=True, value_range=(-1, 1), padding=2)
    save_image(grid, os.path.join(out_dir, "side_by_side.png"))

    # --- Timing comparison ---
    print("\nBenchmarking sampling speed...")

    def drift_fn(m, n, d):
        return drift_sample(m, n, d)

    def ddim_50_fn(m, n, d):
        return schedule.ddim_sample(m, n, (3, 32, 32), d, steps=50)

    def ddim_100_fn(m, n, d):
        return schedule.ddim_sample(m, n, (3, 32, 32), d, steps=100)

    drift_ips = benchmark_sampling(drift_model, drift_fn, n=1024, device=device)
    ddim50_ips = benchmark_sampling(ddpm_model, ddim_50_fn, n=256, device=device)
    ddim100_ips = benchmark_sampling(ddpm_model, ddim_100_fn, n=256, device=device)

    print(f"  Drifting (1 NFE):   {drift_ips:.1f} images/sec")
    print(f"  DDIM (50 NFE):      {ddim50_ips:.1f} images/sec")
    print(f"  DDIM (100 NFE):     {ddim100_ips:.1f} images/sec")
    print(f"  Speedup (1 vs 50):  {drift_ips/ddim50_ips:.1f}x")
    print(f"  Speedup (1 vs 100): {drift_ips/ddim100_ips:.1f}x")

    # --- FID computation ---
    if args.compute_fid:
        print("\nComputing FID scores (generating 10k images each)...")
        n_fid = args.n_fid

        # Reference images
        ref_dir = os.path.join(out_dir, "fid_ref")
        if not os.path.exists(ref_dir) or len(os.listdir(ref_dir)) < n_fid:
            print("  Saving reference CIFAR-10 images...")
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            test_dataset = datasets.CIFAR10(
                root="./data", train=False, download=True, transform=transform
            )
            save_cifar10_images(test_dataset, ref_dir, n_images=n_fid)

        # Drift samples
        drift_fid_dir = os.path.join(out_dir, "fid_drift")
        print("  Generating drift samples...")
        generate_fid_images(drift_model, n_fid, drift_fid_dir, device, drift_fn)

        # DDPM DDIM-50 samples
        ddim50_fid_dir = os.path.join(out_dir, "fid_ddim50")
        print("  Generating DDIM-50 samples...")
        generate_fid_images(ddpm_model, n_fid, ddim50_fid_dir, device, ddim_50_fn)

        # Compute FID
        print("  Computing FID scores...")
        fid_drift = compute_fid(ref_dir, drift_fid_dir)
        fid_ddim50 = compute_fid(ref_dir, ddim50_fid_dir)
        print(f"\n  FID (Drifting, 1 NFE):  {fid_drift}")
        print(f"  FID (DDIM, 50 NFE):     {fid_ddim50}")

    # Summary
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"Drifting (1 NFE):  {drift_ips:.1f} img/s")
    print(f"DDIM-50 (50 NFE):  {ddim50_ips:.1f} img/s")
    print(f"DDIM-100 (100 NFE): {ddim100_ips:.1f} img/s")
    if args.compute_fid:
        print(f"FID Drift:  {fid_drift}")
        print(f"FID DDIM50: {fid_ddim50}")
    print(f"All outputs saved to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--drift-ckpt", type=str, default="outputs/drift/checkpoints/drift_final.pt")
    parser.add_argument("--ddpm-ckpt", type=str, default="outputs/ddpm/checkpoints/ddpm_final.pt")
    parser.add_argument("--output-dir", type=str, default="outputs/comparison")
    parser.add_argument("--compute-fid", action="store_true")
    parser.add_argument("--n-fid", type=int, default=10000)
    args = parser.parse_args()
    main(args)
