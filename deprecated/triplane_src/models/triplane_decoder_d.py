"""TriplaneDecoderD: three 2D triplane latent planes -> reconstructed 3D volume.

Architecture:
    1. TriConv(k=1) re-expands latent_channels → plane_channels.
    2. N TriConvResBlocks mix information within and across planes.
    3. Each plane is broadcast (expand) to [B, C, H, W, D] along its missing axis.
    4. Three broadcasts are concatenated → [B, 3*C, H, W, D].
    5. Conv3d(3C→C, k=3) + GroupNorm + SiLU fuses the planes in 3D.
    6. Conv3d(C→out_channels, k=1) maps to the output channel count.

Plane conventions (matches tri_conv.py and TriplaneEncoder):
    "xy"  [B, C, H, W]   missing axis = D
    "yz"  [B, C, H, D]   missing axis = W
    "xz"  [B, C, W, D]   missing axis = H

where latent_shape = (H, W, D).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .tri_conv import TriConv, TriConvResBlock


def _norm(c: int) -> nn.Module:
    return nn.GroupNorm(num_groups=min(8, c), num_channels=c)


class TriplaneDecoderD(nn.Module):
    """
    Args:
        latent_shape:       (H, W, D) target 3D spatial shape.
        plane_channels:     working channel width for plane processing.
        latent_channels:    input channel count (from encoder mu/logvar).
        n_tri_resblocks:    number of TriConvResBlocks.
        out_channels:       output 3D channels (4 for MAISI latent).
    """

    def __init__(
        self,
        latent_shape: tuple[int, int, int] = (60, 60, 32),
        plane_channels: int = 16,
        latent_channels: int = 16,
        n_tri_resblocks: int = 3,
        out_channels: int = 4,
    ) -> None:
        super().__init__()
        H, W, D = latent_shape
        self.H, self.W, self.D = H, W, D

        # Channel re-expansion: latent_channels → plane_channels.
        self.expand = TriConv(latent_channels, plane_channels, kernel_size=1)

        # TriConvResBlock stack.
        blocks: list[nn.Module] = []
        for _ in range(n_tri_resblocks):
            blocks.append(TriConvResBlock(plane_channels, plane_channels))
        self.tri_blocks = nn.ModuleList(blocks)

        # 3D fusion: concat three broadcasts → Conv3d + norm + act.
        self.fuse_conv = nn.Conv3d(3 * plane_channels, plane_channels, kernel_size=3, padding=1)
        self.fuse_norm = _norm(plane_channels)
        self.fuse_act = nn.SiLU()

        # Final projection to output channels.
        self.out_conv = nn.Conv3d(plane_channels, out_channels, kernel_size=1)

    def forward(self, planes: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            planes: {"xy": [B, latent_channels, H, W],
                     "yz": [B, latent_channels, H, D],
                     "xz": [B, latent_channels, W, D]}

        Returns:
            [B, out_channels, H, W, D]
        """
        H, W, D = self.H, self.W, self.D

        # Re-expand channels.
        planes = self.expand(planes)

        # TriConvResBlock stack.
        for blk in self.tri_blocks:
            planes = blk(planes)

        xy = planes["xy"]  # [B, C, H, W]
        yz = planes["yz"]  # [B, C, H, D]
        xz = planes["xz"]  # [B, C, W, D]
        B, C = xy.shape[:2]

        # Broadcast each plane to [B, C, H, W, D].
        xy3d = xy.unsqueeze(-1).expand(B, C, H, W, D)      # expand D
        yz3d = yz.unsqueeze(3).expand(B, C, H, W, D)       # expand W (insert at dim 3)
        xz3d = xz.unsqueeze(2).expand(B, C, H, W, D)       # expand H (insert at dim 2)

        # Concatenate and fuse.
        fused = torch.cat([xy3d, yz3d, xz3d], dim=1)       # [B, 3C, H, W, D]
        fused = self.fuse_act(self.fuse_norm(self.fuse_conv(fused)))  # [B, C, H, W, D]

        return self.out_conv(fused)                          # [B, out_channels, H, W, D]
