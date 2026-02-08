"""Benchmark inference: DDPM (DDIM 50 steps) vs Drift (1 step)."""

import torch
import time
from drifting_vs_diffusion.config import UNetConfig
from drifting_vs_diffusion.models.unet import UNet
from drifting_vs_diffusion.training.ddpm_utils import DDPMSchedule

device = torch.device("cuda")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

cfg = UNetConfig()
model = UNet(
    in_ch=cfg.in_ch, out_ch=cfg.out_ch, base_ch=cfg.base_ch,
    ch_mult=cfg.ch_mult, num_res_blocks=cfg.num_res_blocks,
    attn_resolutions=cfg.attn_resolutions, dropout=0.0,
    num_heads=cfg.num_heads,
).to(device).eval()

n_params = sum(p.numel() for p in model.parameters())
print(f"UNet: {n_params:,} params")
print()

schedule = DDPMSchedule(T=1000).to(device)

# Warmup
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    for _ in range(5):
        z = torch.randn(1, 3, 32, 32, device=device)
        _ = model(z)
torch.cuda.synchronize()

# --- Drift: single forward pass ---
for batch in [1, 8, 64]:
    torch.cuda.synchronize()
    times = []
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(50):
            z = torch.randn(batch, 3, 32, 32, device=device)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model(z)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    avg = sum(times[5:]) / len(times[5:])  # skip first 5 for warmup
    per_img = avg / batch * 1000
    print(f"Drift (1 step)  batch={batch:>2d}: {avg*1000:>7.2f} ms total, {per_img:.2f} ms/image")

print()

# --- DDPM: DDIM 50 steps ---
for batch in [1, 8, 64]:
    torch.cuda.synchronize()
    times = []
    with torch.no_grad():
        for _ in range(10):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            samples = schedule.ddim_sample(model, batch, (3, 32, 32), device, steps=50)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    avg = sum(times[2:]) / len(times[2:])
    per_img = avg / batch * 1000
    print(f"DDPM (50 steps) batch={batch:>2d}: {avg*1000:>7.2f} ms total, {per_img:.2f} ms/image")

print()

# Ratio
drift_1 = []
ddpm_1 = []
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    for _ in range(50):
        z = torch.randn(1, 3, 32, 32, device=device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(z)
        torch.cuda.synchronize()
        drift_1.append(time.perf_counter() - t0)
with torch.no_grad():
    for _ in range(10):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = schedule.ddim_sample(model, 1, (3, 32, 32), device, steps=50)
        torch.cuda.synchronize()
        ddpm_1.append(time.perf_counter() - t0)

d = sum(drift_1[5:]) / len(drift_1[5:])
p = sum(ddpm_1[2:]) / len(ddpm_1[2:])
print(f"Speedup: DDPM takes {p/d:.1f}x longer than Drift for 1 image")
