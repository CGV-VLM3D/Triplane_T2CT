from __future__ import annotations

import torch
import torch.nn as nn


def _norm(c: int) -> nn.Module:
    return nn.GroupNorm(num_groups=min(8, c), num_channels=c)


class TriplaneDecoder(nn.Module):
    """
    Three 2D triplane feature maps -> reconstructed 3D MAISI latent.

    Args:
        in_channels:  8   (triplane channel width)
        hidden:       32  (3D conv hidden channels)
        out_channels: 4   (MAISI mu channels)
        latent_shape: (120, 120, 64)  (H, W, D)

    Expand strategy: broadcast each 2D plane to 3D without allocating a
    new tensor via expand() + contiguous().  contiguous() is required before
    Conv3d because expand() produces a non-contiguous view.
    """

    def __init__(
        self,
        in_channels: int = 8,
        hidden: int = 32,
        out_channels: int = 4,
        latent_shape: tuple[int, int, int] = (120, 120, 64),
    ) -> None:
        super().__init__()
        H, W, D = latent_shape
        self.H, self.W, self.D = H, W, D

        # 4 Conv3d blocks: 8->32->32->32->4.
        # Last layer has no norm/activation (clean latent output).
        self.conv_blocks = nn.Sequential(
            nn.Conv3d(in_channels, hidden, kernel_size=3, padding=1),
            _norm(hidden),
            nn.SiLU(),
            nn.Conv3d(hidden, hidden, kernel_size=3, padding=1),
            _norm(hidden),
            nn.SiLU(),
            nn.Conv3d(hidden, hidden, kernel_size=3, padding=1),
            _norm(hidden),
            nn.SiLU(),
            nn.Conv3d(hidden, out_channels, kernel_size=3, padding=1),
        )

    def forward(
        self,
        z_xy: torch.Tensor,
        z_yz: torch.Tensor,
        z_xz: torch.Tensor,
    ) -> torch.Tensor:
        # z_xy: [B, C, H, W]  -> expand D
        # z_yz: [B, C, W, D]  -> expand H
        # z_xz: [B, C, H, D]  -> expand W
        B, C, H, W = z_xy.shape
        D = z_yz.shape[3]

        # Expand XY plane: [B, C, H, W] -> [B, C, H, W, D]
        xy3d = z_xy.unsqueeze(4).expand(B, C, H, W, D).contiguous()

        # Expand YZ plane: [B, C, W, D] -> [B, C, H, W, D]
        # unsqueeze at dim=2 (H position), then expand
        yz3d = z_yz.unsqueeze(2).expand(B, C, H, W, D).contiguous()

        # Expand XZ plane: [B, C, H, D] -> [B, C, H, W, D]
        # unsqueeze at dim=3 (W position), then expand
        xz3d = z_xz.unsqueeze(3).expand(B, C, H, W, D).contiguous()

        fused = xy3d + yz3d + xz3d  # [B, C, H, W, D]
        return self.conv_blocks(fused)  # [B, out_channels, H, W, D]
