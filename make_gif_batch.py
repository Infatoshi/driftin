"""Generate a batch GIF: 8 images being generated simultaneously.

DDPM side: all 8 denoise together over 50 steps.
Drift side: all 8 appear instantly in 1 step.
"""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

from drifting_vs_diffusion.config import UNetConfig
from drifting_vs_diffusion.models.unet import UNet
from drifting_vs_diffusion.training.ddpm_utils import DDPMSchedule

DEVICE = torch.device("cuda")
UPSCALE = 4  # 32 -> 128 each
CELL = 32 * UPSCALE  # 128px per image
N_IMGS = 8  # 2x4 grid

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


def load_model(ckpt_path):
    cfg = UNetConfig()
    model = UNet(
        in_ch=cfg.in_ch, out_ch=cfg.out_ch, base_ch=cfg.base_ch,
        ch_mult=cfg.ch_mult, num_res_blocks=cfg.num_res_blocks,
        attn_resolutions=cfg.attn_resolutions, dropout=0.0,
        num_heads=cfg.num_heads,
    ).to(DEVICE).eval()
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    sd = ckpt.get("ema", ckpt["model"])
    clean = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(clean)
    return model


def tensor_to_pil(t):
    t = t.clamp(-1, 1)
    t = (t + 1) / 2
    arr = (t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    img = Image.fromarray(arr)
    return img.resize((CELL, CELL), Image.NEAREST)


def make_grid(tensors, ncol=4):
    """Arrange PIL images in a grid."""
    pils = [tensor_to_pil(t) for t in tensors]
    nrow = (len(pils) + ncol - 1) // ncol
    w = ncol * CELL + (ncol - 1) * 4
    h = nrow * CELL + (nrow - 1) * 4
    grid = Image.new("RGB", (w, h), (15, 15, 15))
    for i, pil in enumerate(pils):
        r, c = divmod(i, ncol)
        grid.paste(pil, (c * (CELL + 4), r * (CELL + 4)))
    return grid


def get_font(size):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def ddim_intermediates(model, schedule, z, n_steps=50):
    intermediates = [z.clone()]
    T = schedule.T
    step_seq = list(range(0, T, T // n_steps))
    step_seq_next = [-1] + step_seq[:-1]
    alphas_bar = schedule.alpha_bar
    x = z.clone()
    with torch.no_grad():
        for i in reversed(range(len(step_seq))):
            t_cur = step_seq[i]
            t_next = step_seq_next[i]
            t_batch = torch.full((x.shape[0],), t_cur, device=x.device, dtype=torch.long)
            pred_noise = model(x, t_batch)
            ab_cur = alphas_bar[t_cur]
            x0_pred = (x - (1 - ab_cur).sqrt() * pred_noise) / ab_cur.sqrt()
            x0_pred = x0_pred.clamp(-1, 1)
            if t_next < 0:
                x = x0_pred
            else:
                ab_next = alphas_bar[t_next]
                x = ab_next.sqrt() * x0_pred + (1 - ab_next).sqrt() * pred_noise
            intermediates.append(x.clone())
    return intermediates


def make_frame(ddpm_grid, drift_grid, step, total, elapsed_ms, drift_ms, done=False):
    GAP = 40
    MARGIN = 20
    gw, gh = ddpm_grid.size
    W = MARGIN * 2 + gw * 2 + GAP
    H = gh + 130
    frame = Image.new("RGB", (W, H), (15, 15, 15))
    draw = ImageDraw.Draw(frame)

    font_title = get_font(20)
    font_label = get_font(15)
    font_time = get_font(13)
    font_big = get_font(24)

    draw.text((W // 2, 10), "Batch Generation: DDPM vs Drifting",
              fill=(230, 230, 230), font=font_title, anchor="mt")

    y_img = 48
    # DDPM
    x_ddpm = MARGIN
    draw.rectangle([x_ddpm - 2, y_img - 2, x_ddpm + gw + 1, y_img + gh + 1],
                   outline=(100, 200, 255), width=2)
    frame.paste(ddpm_grid, (x_ddpm, y_img))
    draw.text((x_ddpm + gw // 2, y_img - 6), "DDPM (50 steps)",
              fill=(100, 200, 255), font=font_label, anchor="mb")
    step_text = f"Step {step}/{total}" if not done else f"Done! ({total} steps)"
    draw.text((x_ddpm + gw // 2, y_img + gh + 6), step_text,
              fill=(180, 180, 180), font=font_time, anchor="mt")
    draw.text((x_ddpm + gw // 2, y_img + gh + 24), f"{elapsed_ms:.0f} ms",
              fill=(180, 180, 180), font=font_time, anchor="mt")

    # Drift
    x_drift = x_ddpm + gw + GAP
    draw.rectangle([x_drift - 2, y_img - 2, x_drift + gw + 1, y_img + gh + 1],
                   outline=(120, 230, 140), width=2)
    frame.paste(drift_grid, (x_drift, y_img))
    draw.text((x_drift + gw // 2, y_img - 6), "Drifting (1 step)",
              fill=(120, 230, 140), font=font_label, anchor="mb")
    draw.text((x_drift + gw // 2, y_img + gh + 6), "Done! (1 step)",
              fill=(180, 180, 180), font=font_time, anchor="mt")
    draw.text((x_drift + gw // 2, y_img + gh + 24), f"{drift_ms:.1f} ms",
              fill=(180, 180, 180), font=font_time, anchor="mt")

    if done:
        speedup = elapsed_ms / drift_ms
        draw.text((W // 2, H - 12),
                  f"Drifting: {speedup:.0f}x faster",
                  fill=(120, 230, 140), font=font_big, anchor="mb")

    return frame


def main():
    print("Loading models...")
    ddpm = load_model("outputs/ddpm_h100/checkpoints/ddpm_final.pt")
    drift = load_model("outputs/drift_dinov2/checkpoints/drift_final.pt")
    schedule = DDPMSchedule(T=1000).to(DEVICE)

    torch.manual_seed(42)
    z = torch.randn(N_IMGS, 3, 32, 32, device=DEVICE)

    # Benchmark
    print("Benchmarking...")
    with torch.no_grad():
        for _ in range(3):
            _ = drift(z)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.no_grad():
        drift_out = drift(z)
    end.record()
    torch.cuda.synchronize()
    drift_ms = start.elapsed_time(end)

    with torch.no_grad():
        for _ in range(2):
            _ = ddim_intermediates(ddpm, schedule, z.clone())
    torch.cuda.synchronize()
    start.record()
    with torch.no_grad():
        intermediates = ddim_intermediates(ddpm, schedule, z.clone())
    end.record()
    torch.cuda.synchronize()
    ddpm_ms = start.elapsed_time(end)
    per_step = ddpm_ms / 50

    print(f"Drift: {drift_ms:.1f} ms for {N_IMGS} images")
    print(f"DDPM: {ddpm_ms:.1f} ms for {N_IMGS} images")

    drift_grid = make_grid(drift_out)

    print("Generating frames...")
    frames = []

    # Initial noise frames
    for _ in range(3):
        f = make_frame(make_grid(intermediates[0]), drift_grid,
                       0, 50, 0, drift_ms)
        frames.append(f)

    # Denoising frames
    for step in range(1, 51):
        f = make_frame(make_grid(intermediates[step]), drift_grid,
                       step, 50, step * per_step, drift_ms,
                       done=(step == 50))
        frames.append(f)

    # Hold final
    for _ in range(20):
        frames.append(frames[-1])

    gif_path = "outputs/drifting_vs_diffusion_batch.gif"
    frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                   duration=80, loop=0)
    print(f"Saved: {gif_path} ({len(frames)} frames, {os.path.getsize(gif_path)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
