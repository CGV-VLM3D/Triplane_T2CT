from __future__ import annotations

import torch
import torch.nn.functional as F


def latent_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean absolute error per sample. Returns [B]."""
    B = pred.shape[0]
    return (pred - target).abs().reshape(B, -1).mean(dim=1)


def latent_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error per sample. Returns [B]."""
    B = pred.shape[0]
    return ((pred - target) ** 2).reshape(B, -1).mean(dim=1)


def latent_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float | torch.Tensor,
) -> torch.Tensor:
    """PSNR per sample. Returns [B].

    data_range: scalar or [B] tensor. MSE is clamped to 1e-10 so identical
    inputs yield large-finite PSNR rather than +inf.
    """
    mse = latent_mse(pred, target).clamp(min=1e-10)
    if isinstance(data_range, torch.Tensor):
        dr = data_range.to(mse)
    else:
        dr = mse.new_tensor(data_range)
    return 10.0 * torch.log10(dr**2 / mse)


def latent_cosine_similarity(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cosine similarity per sample after flatten. Returns [B]."""
    B = pred.shape[0]
    p = pred.reshape(B, -1)
    t = target.reshape(B, -1)
    return F.cosine_similarity(p, t, dim=1)


def compute_latent_data_range(latents: torch.Tensor) -> torch.Tensor:
    """99th - 1st percentile of the whole latent tensor. Returns scalar."""
    flat = latents.reshape(-1).float()
    q_lo = torch.quantile(flat, 0.01)
    q_hi = torch.quantile(flat, 0.99)
    return q_hi - q_lo
