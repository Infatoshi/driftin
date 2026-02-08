"""Build a labeled comparison grid: DDPM vs Drift+ResNet vs Drift+DINOv2.

Output: 1024x1024 PNG with 3 rows x 3 cols, labeled.
"""

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random

# Paths to sample grids
GRIDS = [
    ("DDPM\n50k steps", "outputs/ddpm_h100/samples/ddpm_step0050000.png"),
    ("Drift + ResNet-18\n50k steps, bs=4096", "outputs/drift_bs512_feat/samples/drift_step0050000.png"),
    ("Drift + DINOv2\n50k steps, bs=1024", "outputs/drift_dinov2/samples/drift_step0050000.png"),
]

CELL = 32
PAD = 2
CANVAS = 1024

def extract_cell(grid_img, row, col):
    x = col * (CELL + PAD) + PAD
    y = row * (CELL + PAD) + PAD
    return grid_img.crop((x, y, x + CELL, y + CELL))

def pick_good_cells(grid_img, n=3):
    candidates = []
    for r in range(8):
        for c in range(8):
            cell = extract_cell(grid_img, r, c)
            arr = np.array(cell, dtype=np.float32)
            var = arr.var()
            brightness = arr.mean()
            # Prefer high variance, not too dark, not too bright
            score = var * (1.0 if 40 < brightness < 220 else 0.3)
            candidates.append((score, r, c))
    candidates.sort(reverse=True)
    chosen = random.sample(candidates[:15], min(n, len(candidates[:15])))
    return [(r, c) for _, r, c in chosen]

# Layout
IMG_SZ = 270          # each upscaled image
COLS = 3
ROWS = 3
H_GAP = 12           # horizontal gap between images
V_GAP = 12           # vertical gap between rows
LEFT_W = 0           # no left label column -- labels go above
TOP_H = 52           # title
ROW_LABEL_H = 52     # label above each row

# Compute
content_w = COLS * IMG_SZ + (COLS - 1) * H_GAP
content_h = ROWS * (ROW_LABEL_H + IMG_SZ) + (ROWS - 1) * V_GAP + TOP_H
side_pad = (CANVAS - content_w) // 2
vert_pad = (CANVAS - content_h) // 2

canvas = Image.new("RGB", (CANVAS, CANVAS), (15, 15, 15))  # dark background
draw = ImageDraw.Draw(canvas)

try:
    font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
    font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    font_sub = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
except OSError:
    font_title = ImageFont.load_default()
    font_label = font_title
    font_sub = font_title

# Title
draw.text((CANVAS // 2, vert_pad // 2 + 4), "Drifting vs Diffusion -- CIFAR-10 (32x32)",
          fill=(230, 230, 230), font=font_title, anchor="mm")

# Colors for each method
ROW_COLORS = [
    (100, 200, 255),  # blue for DDPM
    (255, 180, 80),   # orange for ResNet
    (120, 230, 140),  # green for DINOv2
]

random.seed(42)

for row_idx, (label, path) in enumerate(GRIDS):
    grid_img = Image.open(path).convert("RGB")
    cells = pick_good_cells(grid_img, n=3)

    y_start = vert_pad + TOP_H + row_idx * (ROW_LABEL_H + IMG_SZ + V_GAP)
    color = ROW_COLORS[row_idx]

    # Row label (centered above the row of images)
    lines = label.split("\n")
    draw.text((CANVAS // 2, y_start + 4), lines[0],
              fill=color, font=font_label, anchor="mt")
    if len(lines) > 1:
        draw.text((CANVAS // 2, y_start + 26), lines[1],
                  fill=(160, 160, 160), font=font_sub, anchor="mt")

    for col_idx, (r, c) in enumerate(cells):
        cell = extract_cell(grid_img, r, c)
        big = cell.resize((IMG_SZ, IMG_SZ), Image.LANCZOS)

        x = side_pad + col_idx * (IMG_SZ + H_GAP)
        y = y_start + ROW_LABEL_H

        # Colored border
        bw = 2
        draw.rectangle([x - bw, y - bw, x + IMG_SZ + bw - 1, y + IMG_SZ + bw - 1],
                       outline=color, width=bw)
        canvas.paste(big, (x, y))

out_path = "outputs/comparison_grid.png"
canvas.save(out_path, quality=95)
print(f"Saved {out_path} ({canvas.size[0]}x{canvas.size[1]})")
