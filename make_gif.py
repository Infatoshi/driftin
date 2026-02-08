"""Generate a side-by-side GIF: DDPM (50 denoising steps) vs Drift (1 step).

Shows the dramatic speed difference in real time -- drift produces the image
instantly while DDPM is still iterating through noise.
"""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

from drifting_vs_diffusion.config import UNetConfig
from drifting_vs_diffusion.models.unet import UNet
from drifting_vs_diffusion.training.ddpm_utils import DDPMSchedule

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = "outputs/gif_frames"
UPSCALE = 8  # 32 -> 256
IMG_PX = 32 * UPSCALE  # 256

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


def load_model(ckpt_path):
    """Load model from checkpoint, stripping _orig_mod. prefix."""
    cfg = UNetConfig()
    model = UNet(
        in_ch=cfg.in_ch, out_ch=cfg.out_ch, base_ch=cfg.base_ch,
        ch_mult=cfg.ch_mult, num_res_blocks=cfg.num_res_blocks,
        attn_resolutions=cfg.attn_resolutions, dropout=0.0,
        num_heads=cfg.num_heads,
    ).to(DEVICE).eval()

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    # Use EMA weights if available
    sd = ckpt.get("ema", ckpt["model"])
    # Strip _orig_mod. prefix
    clean = {}
    for k, v in sd.items():
        clean_key = k.replace("_orig_mod.", "")
        clean[clean_key] = v
    model.load_state_dict(clean)
    return model


def tensor_to_pil(t, upscale=UPSCALE):
    """Convert [3, 32, 32] tensor in [-1,1] to upscaled PIL image."""
    t = t.clamp(-1, 1)
    t = (t + 1) / 2  # -> [0, 1]
    arr = (t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    img = Image.fromarray(arr)
    return img.resize((IMG_PX, IMG_PX), Image.NEAREST)


def get_font(size):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def make_frame(ddpm_img, drift_img, step, total_steps, elapsed_ms, drift_ms,
               ddpm_done=False):
    """Compose a single frame: DDPM left, Drift right, with labels."""
    W = IMG_PX * 2 + 60 + 40  # two images + gap + margins
    H = IMG_PX + 120  # image + top/bottom labels
    frame = Image.new("RGB", (W, H), (15, 15, 15))
    draw = ImageDraw.Draw(frame)

    font_title = get_font(22)
    font_label = get_font(16)
    font_time = get_font(14)
    font_big = get_font(28)

    # Title
    draw.text((W // 2, 12), "DDPM vs Drifting -- Same UNet (38M params)",
              fill=(230, 230, 230), font=font_title, anchor="mt")

    # Left: DDPM
    x_ddpm = 20
    y_img = 50
    # Border color: blue for DDPM
    color_ddpm = (100, 200, 255)
    draw.rectangle([x_ddpm - 2, y_img - 2, x_ddpm + IMG_PX + 1, y_img + IMG_PX + 1],
                   outline=color_ddpm, width=2)
    frame.paste(ddpm_img, (x_ddpm, y_img))

    # DDPM label
    draw.text((x_ddpm + IMG_PX // 2, y_img - 8), "DDPM (50 DDIM steps)",
              fill=color_ddpm, font=font_label, anchor="mb")

    # Step counter
    step_text = f"Step {step}/{total_steps}"
    if ddpm_done:
        step_text = f"Done! ({total_steps} steps)"
    draw.text((x_ddpm + IMG_PX // 2, y_img + IMG_PX + 8),
              step_text, fill=(180, 180, 180), font=font_time, anchor="mt")
    draw.text((x_ddpm + IMG_PX // 2, y_img + IMG_PX + 28),
              f"{elapsed_ms:.0f} ms", fill=(180, 180, 180), font=font_time, anchor="mt")

    # Right: Drift
    x_drift = x_ddpm + IMG_PX + 60
    color_drift = (120, 230, 140)
    draw.rectangle([x_drift - 2, y_img - 2, x_drift + IMG_PX + 1, y_img + IMG_PX + 1],
                   outline=color_drift, width=2)
    frame.paste(drift_img, (x_drift, y_img))

    # Drift label
    draw.text((x_drift + IMG_PX // 2, y_img - 8), "Drifting (1 step)",
              fill=color_drift, font=font_label, anchor="mb")

    # Drift timing
    draw.text((x_drift + IMG_PX // 2, y_img + IMG_PX + 8),
              "Done! (1 step)", fill=(180, 180, 180), font=font_time, anchor="mt")
    draw.text((x_drift + IMG_PX // 2, y_img + IMG_PX + 28),
              f"{drift_ms:.1f} ms", fill=(180, 180, 180), font=font_time, anchor="mt")

    # Speed comparison at bottom
    if ddpm_done:
        speedup = elapsed_ms / drift_ms
        draw.text((W // 2, H - 10),
                  f"Drifting: {speedup:.0f}x faster",
                  fill=(120, 230, 140), font=font_big, anchor="mb")

    return frame


def generate_ddpm_intermediates(model, schedule, z, n_ddim_steps=50):
    """Run DDIM sampling and capture every intermediate step."""
    intermediates = [z.clone()]  # step 0 = pure noise

    T = schedule.T
    # DDIM sub-sequence
    step_seq = list(range(0, T, T // n_ddim_steps))
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

    return intermediates  # len = n_ddim_steps + 1


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading models...")
    ddpm_model = load_model("outputs/ddpm_h100/checkpoints/ddpm_final.pt")
    drift_model = load_model("outputs/drift_dinov2/checkpoints/drift_final.pt")

    schedule = DDPMSchedule(T=1000).to(DEVICE)

    # Measure actual inference times
    print("Benchmarking...")
    # Drift timing
    z_drift = torch.randn(1, 3, 32, 32, device=DEVICE)
    with torch.no_grad():
        for _ in range(5):
            _ = drift_model(z_drift)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.no_grad():
        drift_out = drift_model(z_drift)
    end.record()
    torch.cuda.synchronize()
    drift_ms = start.elapsed_time(end)
    print(f"Drift: {drift_ms:.1f} ms")

    # DDPM timing
    z_ddpm = torch.randn(1, 3, 32, 32, device=DEVICE)
    with torch.no_grad():
        for _ in range(2):
            _ = generate_ddpm_intermediates(ddpm_model, schedule, z_ddpm.clone())
    torch.cuda.synchronize()

    start.record()
    with torch.no_grad():
        intermediates = generate_ddpm_intermediates(ddpm_model, schedule, z_ddpm.clone())
    end.record()
    torch.cuda.synchronize()
    ddpm_ms = start.elapsed_time(end)
    print(f"DDPM (50 DDIM steps): {ddpm_ms:.1f} ms")
    per_step_ms = ddpm_ms / 50

    # Convert to PIL
    drift_pil = tensor_to_pil(drift_out[0])
    ddpm_pils = [tensor_to_pil(x[0]) for x in intermediates]

    # Generate frames
    # We want the GIF to play at roughly 10x slow-mo so people can see the DDPM process
    # Each DDIM step takes ~8ms real time, we show each as one GIF frame
    # Frame duration = 80ms (12.5 fps) to give ~10x slow-mo feel
    print("Generating frames...")
    frames = []

    n_steps = 50

    # Frame 0: both start as noise, drift already done
    # Show 3 frames of pure noise on DDPM side before it "starts"
    for i in range(3):
        f = make_frame(ddpm_pils[0], drift_pil,
                       step=0, total_steps=n_steps,
                       elapsed_ms=0, drift_ms=drift_ms)
        frames.append(f)

    # Frames 1-50: DDPM denoising, one frame per DDIM step
    for step in range(1, n_steps + 1):
        elapsed = step * per_step_ms
        f = make_frame(ddpm_pils[step], drift_pil,
                       step=step, total_steps=n_steps,
                       elapsed_ms=elapsed, drift_ms=drift_ms,
                       ddpm_done=(step == n_steps))
        frames.append(f)

    # Hold on final frame
    for _ in range(20):
        f = make_frame(ddpm_pils[-1], drift_pil,
                       step=n_steps, total_steps=n_steps,
                       elapsed_ms=ddpm_ms, drift_ms=drift_ms,
                       ddpm_done=True)
        frames.append(f)

    # Save GIF
    gif_path = "outputs/drifting_vs_diffusion.gif"
    frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                   duration=80, loop=0)
    print(f"Saved GIF: {gif_path} ({len(frames)} frames)")

    # Also save the final comparison as a static image
    static_path = "outputs/drifting_vs_diffusion_final.png"
    frames[-1].save(static_path)
    print(f"Saved static: {static_path}")


if __name__ == "__main__":
    main()
