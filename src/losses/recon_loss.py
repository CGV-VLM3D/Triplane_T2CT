from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconLoss(nn.Module):
    """
    Weighted reconstruction loss.  Trial 1: L1 only.
    Trial 2+: set lpips_weight > 0 or gan_weight > 0 to enable those terms.

    When a weight is 0.0 the corresponding term is skipped entirely (no
    forward pass through a potentially expensive network).
    """

    def __init__(
        self,
        l1_weight: float = 1.0,
        lpips_weight: float = 0.0,
        gan_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.l1_weight = l1_weight
        self.lpips_weight = lpips_weight
        self.gan_weight = gan_weight

        # LPIPS and GAN modules are intentionally absent in trial 1.
        # Trial 2+: instantiate and register them here; gate with weight > 0.

    def forward(
        self, mu_hat: torch.Tensor, mu: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        zero = mu.new_zeros(())

        l1 = F.l1_loss(mu_hat, mu) if self.l1_weight > 0.0 else zero
        # lpips/gan: placeholder zeros until trial 2+ adds real networks.
        lpips = zero
        gan = zero

        total = self.l1_weight * l1 + self.lpips_weight * lpips + self.gan_weight * gan
        return {"total": total, "l1": l1, "lpips": lpips, "gan": gan}
