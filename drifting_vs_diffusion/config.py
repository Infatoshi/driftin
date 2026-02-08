from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class UNetConfig:
    """Small config (~38M params) for single-GPU / 3090."""
    in_ch: int = 3
    out_ch: int = 3
    base_ch: int = 128
    ch_mult: Tuple[int, ...] = (1, 2, 2, 2)
    num_res_blocks: int = 2
    attn_resolutions: Tuple[int, ...] = (16,)
    dropout: float = 0.1
    num_heads: int = 4
    image_size: int = 32


@dataclass
class UNetLargeConfig:
    """Large config (~152M params) for 8xH100.

    base_ch=256, attn at 16x16 and 8x8, 8 heads.
    Channel progression: 256 -> 512 -> 512 -> 512.
    """
    in_ch: int = 3
    out_ch: int = 3
    base_ch: int = 256
    ch_mult: Tuple[int, ...] = (1, 2, 2, 2)
    num_res_blocks: int = 3
    attn_resolutions: Tuple[int, ...] = (16, 8)
    dropout: float = 0.1
    num_heads: int = 8
    image_size: int = 32


@dataclass
class DDPMConfig:
    T: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    # Training
    batch_size: int = 128
    lr: float = 2e-4
    ema_decay: float = 0.9999
    max_grad_norm: float = 1.0
    total_steps: int = 200_000
    # Logging
    log_every: int = 100
    sample_every: int = 10_000
    save_every: int = 10_000


@dataclass
class DriftConfig:
    # Training
    batch_size: int = 128
    lr: float = 2e-4
    ema_decay: float = 0.9999
    max_grad_norm: float = 2.0
    total_steps: int = 200_000
    temperatures: List[float] = field(default_factory=lambda: [0.02, 0.05, 0.2])
    # Feature encoder
    use_feature_encoder: bool = False  # Start with raw pixels
    # Logging
    log_every: int = 100
    sample_every: int = 10_000
    save_every: int = 10_000
