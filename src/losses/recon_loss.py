from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconLoss(nn.Module):
    """Weighted L1 + L2 + KL reconstruction loss. Zero-weight terms are skipped."""

    def __init__(
        self,
        l1_weight: float = 1.0,
        l2_weight: float = 0.0,
        kl_weight: float = 1e-6,
    ) -> None:
        super().__init__()
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.kl_weight = kl_weight

    def forward(
        self,
        mu_hat: torch.Tensor,
        mu: torch.Tensor,
        kl_loss: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        zero = mu.new_zeros(())
        l1 = F.l1_loss(mu_hat, mu) if self.l1_weight > 0.0 else zero
        l2 = F.mse_loss(mu_hat, mu) if self.l2_weight > 0.0 else zero
        kl = kl_loss if (kl_loss is not None and self.kl_weight > 0.0) else zero
        total = self.l1_weight * l1 + self.l2_weight * l2 + self.kl_weight * kl
        return {"total": total, "l1": l1, "l2": l2, "kl": kl}
