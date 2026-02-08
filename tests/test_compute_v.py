import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drifting_vs_diffusion.training.compute_v import compute_drift, drifting_loss


def test_compute_drift_preserves_shape():
    gen = torch.randn(8, 16)
    pos = torch.randn(8, 16)

    drift = compute_drift(gen, pos, temp=0.05)

    assert drift.shape == gen.shape


def test_drifting_loss_returns_scalar_and_backward():
    gen = torch.randn(8, 16, requires_grad=True)
    pos = torch.randn(8, 16)

    loss = drifting_loss(gen, pos, temps=(0.02, 0.05))

    assert loss.ndim == 0
    assert loss.item() >= 0.0

    loss.backward()
    assert gen.grad is not None
