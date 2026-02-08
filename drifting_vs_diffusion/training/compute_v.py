"""Drift field computation for the drifting model.

Based on the paper's Algorithm 2 with proper dimensionality-aware temperature
scaling (tau_tilde = tau * D) and feature normalization (Appendix A.6).
"""

import math
import torch
import torch.nn.functional as F


def normalize_features(x, y_all, D):
    """Feature normalization so avg pairwise distance ~ sqrt(D).

    Args:
        x: [N, D] generated features
        y_all: [M, D] all reference features (pos + neg)
        D: feature dimensionality

    Returns:
        x_norm: [N, D] normalized features
        S: scalar normalization factor (detached)
    """
    dists = torch.cdist(x.float(), y_all.float(), p=2)
    mean_dist = dists.mean()
    S = (mean_dist / math.sqrt(D)).detach().clamp(min=1e-8)
    return x / S, S


def compute_drift(gen, pos, temp=0.05):
    """Compute drift field V with dimensionality-aware temperature.

    Uses feature normalization + tau_tilde = tau * D scaling so that
    the kernel values are well-behaved regardless of feature dimension.

    Args:
        gen: [N, D] generated features (detached, no grad needed)
        pos: [N_pos, D] positive (real) features
        temp: base temperature (will be scaled by D)

    Returns:
        V: [N, D] drift vectors (in normalized feature space)
    """
    N, D = gen.shape

    # Normalize features so avg pairwise distance ~ sqrt(D)
    y_neg = gen  # negatives = generated batch
    y_all = torch.cat([pos, y_neg], dim=0)
    gen_n, S = normalize_features(gen, y_all, D)
    pos_n = pos / S
    neg_n = gen_n  # normalized negatives

    # Concatenate targets: [pos, neg]
    targets_n = torch.cat([pos_n, neg_n], dim=0)  # [N_pos + N, D]
    N_pos = pos_n.shape[0]

    gen_f = gen_n.float()
    targets_f = targets_n.float()

    # Distances from gen to all targets
    dist = torch.cdist(gen_f, targets_f)  # [N, N_pos + N]

    # Self-exclusion: mask diagonal of gen-to-neg block
    neg_start = N_pos
    for i in range(N):
        dist[i, neg_start + i] = 1e6

    # Dimensionality-aware temperature
    tau_tilde = temp * D

    logit = -dist / tau_tilde
    # Use softmax-style normalization for numerical stability
    A_row = torch.softmax(logit, dim=1)   # normalize over targets
    A_col = torch.softmax(logit, dim=0)   # normalize over gen samples
    A = torch.sqrt(A_row * A_col + 1e-30)

    A_pos = A[:, :N_pos]       # [N, N_pos]
    A_neg = A[:, N_pos:]       # [N, N]

    W_pos = A_pos * A_neg.sum(dim=-1, keepdim=True)  # [N, N_pos]
    W_neg = A_neg * A_pos.sum(dim=-1, keepdim=True)  # [N, N]

    drift_pos = W_pos @ targets_f[:N_pos]  # [N, D]
    drift_neg = W_neg @ targets_f[N_pos:]  # [N, D]

    V = (drift_pos - drift_neg).to(gen.dtype)
    # Un-normalize V back to original feature scale
    V = V * S
    return V


def normalize_drift(V, D):
    """Per-dimension drift normalization (Appendix A.6).

    Scales V so that E[||V||^2 / D] ~ 1.
    """
    lambda_j = torch.sqrt((V.float().pow(2).sum(dim=-1) / D).mean()).detach()
    return V / (lambda_j + 1e-8)


def compute_drift_multitemp(gen, pos, temps=(0.02, 0.05, 0.2)):
    """Compute drift field aggregated over multiple temperatures.

    Each temperature's drift is independently normalized before summing.
    """
    D = gen.shape[-1]
    V_total = torch.zeros_like(gen)
    for temp in temps:
        V = compute_drift(gen, pos, temp=temp)
        V = normalize_drift(V, D)
        V_total += V
    return V_total


def drifting_loss(gen_feats, pos_feats, temps=(0.02, 0.05, 0.2)):
    """Compute drifting loss.

    Gradient flows through gen_feats back to the generator.
    V is computed on detached features and target is detached.

    Args:
        gen_feats: [N, D] generated features (WITH gradient to generator)
        pos_feats: [N_pos, D] positive (real) features (detached)
        temps: temperatures for multi-scale drift

    Returns:
        loss: scalar MSE between gen_feats and (gen_feats_detached + V)
    """
    with torch.no_grad():
        V = compute_drift_multitemp(gen_feats.detach(), pos_feats.detach(), temps=temps)
        target = gen_feats.detach() + V

    return F.mse_loss(gen_feats, target)
