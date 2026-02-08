"""Benchmark single-step UNet inference at different resolutions on 3090."""

import torch
import time

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

device = torch.device("cuda")

from drifting_vs_diffusion.models.unet import UNet

resolutions = [32, 64, 128, 256]

for res in resolutions:
    # Adjust attn_resolutions to be valid for this image size
    attn_res = tuple(r for r in [16, 8] if r <= res // 2)
    if not attn_res:
        attn_res = (res // 2,)

    model = UNet(
        in_ch=3, out_ch=3, base_ch=128,
        ch_mult=(1, 2, 2, 2), num_res_blocks=2,
        attn_resolutions=attn_res, dropout=0.0, num_heads=4,
    ).to(device).eval()

    n_params = sum(p.numel() for p in model.parameters())

    # Warmup
    try:
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(3):
                z = torch.randn(1, 3, res, res, device=device)
                t_dummy = torch.zeros(1, device=device, dtype=torch.long)
                _ = model(z, t_dummy)
        torch.cuda.synchronize()
    except RuntimeError as e:
        print(f"{res}x{res}: OOM or error: {e}")
        del model
        torch.cuda.empty_cache()
        continue

    # Benchmark
    times = []
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(30):
            z = torch.randn(1, 3, res, res, device=device)
            t_dummy = torch.zeros(1, device=device, dtype=torch.long)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(z, t_dummy)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    avg = sum(times[5:]) / len(times[5:])
    fps = 1.0 / avg
    mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"{res:>4d}x{res:<4d} | {avg*1000:>7.1f} ms | {fps:>6.1f} FPS | {mem:.1f} GB | {n_params/1e6:.1f}M params")

    del model
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

# Also test latent-space scenario:
# Game at 512x512, but drift operates on 64x64 latent (like SD's VAE)
print()
print("--- Latent-space scenario (drift on 64x64, VAE decodes to 512x512) ---")
model = UNet(
    in_ch=4, out_ch=4, base_ch=128,  # 4ch latent like SD
    ch_mult=(1, 2, 2, 2), num_res_blocks=2,
    attn_resolutions=(16, 8), dropout=0.0, num_heads=4,
).to(device).eval()
n_params = sum(p.numel() for p in model.parameters())

with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    for _ in range(5):
        z = torch.randn(1, 4, 64, 64, device=device)
        t_dummy = torch.zeros(1, device=device, dtype=torch.long)
        _ = model(z, t_dummy)
torch.cuda.synchronize()

times = []
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    for _ in range(30):
        z = torch.randn(1, 4, 64, 64, device=device)
        t_dummy = torch.zeros(1, device=device, dtype=torch.long)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(z, t_dummy)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

avg = sum(times[5:]) / len(times[5:])
fps = 1.0 / avg
print(f"Drift UNet (4ch, 64x64 latent): {avg*1000:.1f} ms = {fps:.0f} FPS")
print(f"+ VAE decode ~5-10ms -> effective {1000/(avg*1000 + 7.5):.0f} FPS at 512x512")
print(f"  ({n_params/1e6:.1f}M params)")
