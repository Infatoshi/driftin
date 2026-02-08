"""DDPM noise schedule and sampling utilities."""

import torch
import torch.nn.functional as F


class DDPMSchedule:
    """Linear beta schedule for DDPM."""

    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.T = T
        self.betas = torch.linspace(beta_start, beta_end, T, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.alpha_bar_prev = F.pad(self.alpha_bar[:-1], (1, 0), value=1.0)
        # For posterior q(x_{t-1} | x_t, x_0)
        self.posterior_var = self.betas * (1.0 - self.alpha_bar_prev) / (1.0 - self.alpha_bar)

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        self.alpha_bar_prev = self.alpha_bar_prev.to(device)
        self.posterior_var = self.posterior_var.to(device)
        return self

    def q_sample(self, x_0, t, noise=None):
        """Forward process: sample x_t from q(x_t | x_0).

        Args:
            x_0: [B, C, H, W] clean images
            t: [B] timesteps
            noise: optional pre-sampled noise

        Returns:
            x_t: [B, C, H, W] noisy images
            noise: [B, C, H, W] the noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        ab = self.alpha_bar[t].view(-1, 1, 1, 1)
        x_t = torch.sqrt(ab) * x_0 + torch.sqrt(1 - ab) * noise
        return x_t, noise

    @torch.no_grad()
    def p_sample(self, model, x_t, t_int, clip=True):
        """Reverse process: sample x_{t-1} from p(x_{t-1} | x_t).

        Args:
            model: noise prediction network
            x_t: [B, C, H, W] current noisy image
            t_int: integer timestep (same for all samples in batch)
            clip: whether to clip output to [-1, 1]

        Returns:
            x_{t-1}: [B, C, H, W]
        """
        B = x_t.shape[0]
        t_batch = torch.full((B,), t_int, device=x_t.device, dtype=torch.long)
        pred_noise = model(x_t, t_batch)

        beta = self.betas[t_int]
        alpha = self.alphas[t_int]
        ab = self.alpha_bar[t_int]

        # Mean of p(x_{t-1} | x_t)
        mean = (1 / torch.sqrt(alpha)) * (x_t - (beta / torch.sqrt(1 - ab)) * pred_noise)

        if t_int > 0:
            noise = torch.randn_like(x_t)
            sigma = torch.sqrt(beta)
            x_prev = mean + sigma * noise
        else:
            x_prev = mean

        if clip:
            x_prev = x_prev.clamp(-1, 1)
        return x_prev

    @torch.no_grad()
    def ddpm_sample(self, model, n, shape, device, clip=True):
        """Full DDPM sampling: 1000 steps."""
        x = torch.randn(n, *shape, device=device)
        for t in reversed(range(self.T)):
            x = self.p_sample(model, x, t, clip=clip)
        return x

    @torch.no_grad()
    def ddim_sample(self, model, n, shape, device, steps=50, eta=0.0, clip=True):
        """DDIM sampling with configurable steps.

        Args:
            steps: number of denoising steps
            eta: 0.0 for deterministic DDIM, 1.0 for DDPM-like stochasticity
        """
        # Uniform subsequence of timesteps
        step_size = self.T // steps
        timesteps = list(range(0, self.T, step_size))[:steps]
        timesteps = list(reversed(timesteps))

        x = torch.randn(n, *shape, device=device)

        for i, t in enumerate(timesteps):
            t_batch = torch.full((n,), t, device=device, dtype=torch.long)
            pred_noise = model(x, t_batch)

            ab_t = self.alpha_bar[t]

            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
                ab_prev = self.alpha_bar[t_prev]
            else:
                ab_prev = torch.tensor(1.0, device=device)

            # Predicted x_0
            x0_pred = (x - torch.sqrt(1 - ab_t) * pred_noise) / torch.sqrt(ab_t)
            if clip:
                x0_pred = x0_pred.clamp(-1, 1)

            # Direction pointing to x_t
            sigma = eta * torch.sqrt((1 - ab_prev) / (1 - ab_t) * (1 - ab_t / ab_prev))
            dir_xt = torch.sqrt((1 - ab_prev - sigma ** 2).clamp_min(0)) * pred_noise

            x = torch.sqrt(ab_prev) * x0_pred + dir_xt
            if sigma > 0 and i < len(timesteps) - 1:
                x = x + sigma * torch.randn_like(x)

        if clip:
            x = x.clamp(-1, 1)
        return x
