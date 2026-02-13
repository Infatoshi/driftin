"""Shared UNet architecture for both DDPM and Drifting models.

~35M parameters with base_ch=128, ch_mult=(1,2,2,2).
For DDPM:  forward(x_noisy, t) -> predicted_noise
For Drift: forward(z_noise, t=0) -> generated_image
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim, max_period=10000):
    """Sinusoidal timestep embedding. t: [B] long or float -> [B, dim]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class GroupNorm32(nn.GroupNorm):
    """GroupNorm that converts to float32 for numerical stability."""
    def forward(self, x):
        return super().forward(x.float()).to(x.dtype)


class ResBlock(nn.Module):
    """ResBlock with timestep conditioning via scale+shift after second GroupNorm."""

    def __init__(self, in_ch, out_ch, time_dim, dropout=0.1):
        super().__init__()
        self.norm1 = GroupNorm32(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch * 2),
        )
        self.norm2 = GroupNorm32(32, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

        # Zero-init last conv for residual-friendly start
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x, temb):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)

        # Time conditioning: scale + shift on second norm
        t_out = self.time_proj(temb)[:, :, None, None]
        scale, shift = t_out.chunk(2, dim=1)

        h = self.norm2(h) * (1 + scale) + shift
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.skip(x)


class SelfAttention(nn.Module):
    """Multi-head self-attention with pre-norm."""

    def __init__(self, ch, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = GroupNorm32(32, ch)
        self.qkv = nn.Conv1d(ch, ch * 3, 1)
        self.out = nn.Conv1d(ch, ch, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).reshape(B, C, H * W)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        q, k, v = qkv.unbind(1)  # each [B, heads, head_dim, HW]
        q = q.permute(0, 1, 3, 2)  # [B, heads, HW, head_dim]
        k = k.permute(0, 1, 3, 2)
        v = v.permute(0, 1, 3, 2)
        h = F.scaled_dot_product_attention(q, k, v)
        h = h.permute(0, 1, 3, 2).reshape(B, C, H * W)
        h = self.out(h).reshape(B, C, H, W)
        return x + h


class ResAttnBlock(nn.Module):
    """ResBlock + optional SelfAttention."""

    def __init__(self, in_ch, out_ch, time_dim, dropout, use_attn, num_heads):
        super().__init__()
        self.res = ResBlock(in_ch, out_ch, time_dim, dropout)
        self.attn = SelfAttention(out_ch, num_heads) if use_attn else None

    def forward(self, x, temb):
        x = self.res(x, temb)
        if self.attn is not None:
            x = self.attn(x)
        return x


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNet(nn.Module):
    """
    Standard UNet for 32x32 images.

    Architecture:
    - Channel progression: base_ch * ch_mult at each resolution level
    - Spatial: 32 -> 16 -> 8 -> 4 (with 3 downsamples)
    - Self-attention at specified resolutions
    - Timestep conditioning via AdaGN (scale+shift)
    - Skip connections between encoder and decoder
    """

    def __init__(
        self,
        in_ch=3,
        out_ch=3,
        base_ch=128,
        ch_mult=(1, 2, 2, 2),
        num_res_blocks=2,
        attn_resolutions=(16,),
        dropout=0.1,
        num_heads=4,
        num_classes=0,
    ):
        super().__init__()
        self.base_ch = base_ch
        self.ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks
        self.num_classes = num_classes
        time_dim = base_ch * 4

        # Time embedding MLP
        self.time_embed = nn.Sequential(
            nn.Linear(base_ch, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Class conditioning (added to time embedding)
        if num_classes > 0:
            self.class_embed = nn.Embedding(num_classes, time_dim)

        # Input conv
        self.conv_in = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        # --- Down path ---
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        skip_chs = [base_ch]
        ch = base_ch
        res = 32

        for level, mult in enumerate(ch_mult):
            out = base_ch * mult
            use_attn = res in attn_resolutions
            for _ in range(num_res_blocks):
                self.down_blocks.append(
                    ResAttnBlock(ch, out, time_dim, dropout, use_attn, num_heads)
                )
                ch = out
                skip_chs.append(ch)

            if level < len(ch_mult) - 1:
                self.down_samples.append(Downsample(ch))
                skip_chs.append(ch)
                res //= 2

        # --- Middle ---
        self.mid_block1 = ResAttnBlock(ch, ch, time_dim, dropout, True, num_heads)
        self.mid_block2 = ResAttnBlock(ch, ch, time_dim, dropout, False, num_heads)

        # --- Up path ---
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()

        for level in reversed(range(len(ch_mult))):
            mult = ch_mult[level]
            out = base_ch * mult
            use_attn = res in attn_resolutions
            for _ in range(num_res_blocks + 1):
                skip_ch = skip_chs.pop()
                self.up_blocks.append(
                    ResAttnBlock(ch + skip_ch, out, time_dim, dropout, use_attn, num_heads)
                )
                ch = out

            if level > 0:
                self.up_samples.append(Upsample(ch))
                res *= 2

        # Output
        self.norm_out = GroupNorm32(32, ch)
        self.conv_out = nn.Conv2d(ch, out_ch, 3, padding=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x, t=None, class_labels=None):
        """
        Args:
            x: [B, C, H, W] input image (noisy for DDPM, noise for drift)
            t: [B] integer timesteps. If None, uses t=0 (for drift model).
            class_labels: [B] integer class labels. Only used when num_classes > 0.

        Returns:
            [B, C, H, W] predicted noise (DDPM) or generated image (drift)
        """
        B = x.shape[0]
        if t is None:
            t = torch.zeros(B, device=x.device, dtype=torch.long)

        # Time embedding
        temb = timestep_embedding(t, self.base_ch)
        temb = self.time_embed(temb)

        # Class conditioning
        if self.num_classes > 0 and class_labels is not None:
            temb = temb + self.class_embed(class_labels)

        # Input
        h = self.conv_in(x)
        skips = [h]

        # Down
        block_idx = 0
        ds_idx = 0
        for level in range(len(self.ch_mult)):
            for _ in range(self.num_res_blocks):
                h = self.down_blocks[block_idx](h, temb)
                skips.append(h)
                block_idx += 1
            if level < len(self.ch_mult) - 1:
                h = self.down_samples[ds_idx](h)
                skips.append(h)
                ds_idx += 1

        # Middle
        h = self.mid_block1(h, temb)
        h = self.mid_block2(h, temb)

        # Up
        block_idx = 0
        us_idx = 0
        for level in reversed(range(len(self.ch_mult))):
            for _ in range(self.num_res_blocks + 1):
                h = torch.cat([h, skips.pop()], dim=1)
                h = self.up_blocks[block_idx](h, temb)
                block_idx += 1
            if level > 0:
                h = self.up_samples[us_idx](h)
                us_idx += 1

        # Output
        h = F.silu(self.norm_out(h))
        h = self.conv_out(h)
        return h
