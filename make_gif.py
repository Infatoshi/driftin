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
               drift_done=True, ddpm_done=False):
    """Compose a single frame: DDPM top, Drift bottom, stacked vertically."""
    MARGIN = 20
    ROW_LABEL_H = 28      # label above each image
    ROW_INFO_H = 44       # step + ms below each image
    GAP = 16              # vertical gap between rows
    TITLE_H = 40
    FOOTER_H = 44

    W = MARGIN * 2 + IMG_PX
    H = (TITLE_H + (ROW_LABEL_H + IMG_PX + ROW_INFO_H) * 2 + GAP + FOOTER_H)
    frame = Image.new("RGB", (W, H), (15, 15, 15))
    draw = ImageDraw.Draw(frame)

    font_title = get_font(20)
    font_label = get_font(15)
    font_time = get_font(13)
    font_big = get_font(26)

    cx = W // 2

    # Title
    draw.text((cx, 10), "Same UNet (38M params)",
              fill=(230, 230, 230), font=font_title, anchor="mt")

    # --- Top row: DDPM ---
    y_ddpm_label = TITLE_H
    y_ddpm_img = y_ddpm_label + ROW_LABEL_H
    color_ddpm = (100, 200, 255)

    draw.text((cx, y_ddpm_label + 4), "DDPM (50 DDIM steps)",
              fill=color_ddpm, font=font_label, anchor="mt")
    draw.rectangle([MARGIN - 2, y_ddpm_img - 2,
                    MARGIN + IMG_PX + 1, y_ddpm_img + IMG_PX + 1],
                   outline=color_ddpm, width=2)
    frame.paste(ddpm_img, (MARGIN, y_ddpm_img))

    step_text = f"Step {step}/{total_steps}"
    if ddpm_done:
        step_text = f"Done! ({total_steps} steps)"
    draw.text((cx, y_ddpm_img + IMG_PX + 6),
              step_text, fill=(180, 180, 180), font=font_time, anchor="mt")
    draw.text((cx, y_ddpm_img + IMG_PX + 24),
              f"{elapsed_ms:.0f} ms", fill=(180, 180, 180), font=font_time, anchor="mt")

    # --- Bottom row: Drift ---
    y_drift_label = y_ddpm_img + IMG_PX + ROW_INFO_H + GAP
    y_drift_img = y_drift_label + ROW_LABEL_H
    color_drift = (120, 230, 140)

    draw.text((cx, y_drift_label + 4), "Drifting (1 step)",
              fill=color_drift, font=font_label, anchor="mt")
    draw.rectangle([MARGIN - 2, y_drift_img - 2,
                    MARGIN + IMG_PX + 1, y_drift_img + IMG_PX + 1],
                   outline=color_drift, width=2)
    frame.paste(drift_img, (MARGIN, y_drift_img))

    if drift_done:
        draw.text((cx, y_drift_img + IMG_PX + 6),
                  "Done! (1 step)", fill=(120, 230, 140), font=font_time, anchor="mt")
        draw.text((cx, y_drift_img + IMG_PX + 24),
                  f"{drift_ms:.1f} ms", fill=(120, 230, 140), font=font_time, anchor="mt")
    else:
        draw.text((cx, y_drift_img + IMG_PX + 6),
                  "Waiting...", fill=(180, 180, 180), font=font_time, anchor="mt")

    # Footer: speedup
    if ddpm_done:
        speedup = elapsed_ms / drift_ms
        draw.text((cx, H - 8),
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

    # Convert noise to PIL for the "before" state
    noise_pil = tensor_to_pil(intermediates[0][0])

    print("Generating frames...")
    frames = []
    n_steps = 50

    # Phase 1: Both start as noise (3 frames)
    for _ in range(3):
        f = make_frame(ddpm_pils[0], noise_pil,
                       step=0, total_steps=n_steps,
                       elapsed_ms=0, drift_ms=drift_ms,
                       drift_done=False)
        frames.append(f)

    # Phase 2: Frame 1 -- drift SNAPS to final image instantly, DDPM begins denoising
    # This single-frame transition is the whole point
    f = make_frame(ddpm_pils[1], drift_pil,
                   step=1, total_steps=n_steps,
                   elapsed_ms=per_step_ms, drift_ms=drift_ms,
                   drift_done=True)
    frames.append(f)

    # Phase 3: Frames 2-50 -- DDPM continues denoising, drift side is FROZEN
    # We render drift_pil into a single static image and reuse the exact same
    # PIL object so there is zero pixel variation across frames
    for step in range(2, n_steps + 1):
        elapsed = step * per_step_ms
        f = make_frame(ddpm_pils[step], drift_pil,
                       step=step, total_steps=n_steps,
                       elapsed_ms=elapsed, drift_ms=drift_ms,
                       drift_done=True,
                       ddpm_done=(step == n_steps))
        frames.append(f)

    # Hold on final frame
    for _ in range(20):
        frames.append(frames[-1].copy())

    # Build a single global palette from the final frame to prevent per-frame
    # requantization (which causes pixel flicker on the static drift side)
    print("Quantizing to global palette...")
    palette_img = frames[-1].quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    quantized = []
    for f in frames:
        q = f.quantize(palette=palette_img, dither=Image.Dither.NONE)
        quantized.append(q)

    gif_path = "outputs/drifting_vs_diffusion.gif"
    quantized[0].save(gif_path, save_all=True, append_images=quantized[1:],
                      duration=80, loop=0)
    print(f"Saved GIF: {gif_path} ({len(frames)} frames)")

    # Also save the final comparison as a static image
    static_path = "outputs/drifting_vs_diffusion_final.png"
    frames[-1].save(static_path)
    print(f"Saved static: {static_path}")


if __name__ == "__main__":
    main()
