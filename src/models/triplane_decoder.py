from __future__ import annotations

import torch
import torch.nn as nn


def _norm(c: int) -> nn.Module:
    return nn.GroupNorm(num_groups=min(8, c), num_channels=c)


class ResBlock3D(nn.Module):
    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.norm1 = _norm(in_c)
        self.conv1 = nn.Conv3d(in_c, out_c, kernel_size=3, padding=1)
        self.norm2 = _norm(out_c)
        self.conv2 = nn.Conv3d(out_c, out_c, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.skip = (
            nn.Conv3d(in_c, out_c, kernel_size=1) if in_c != out_c else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


class TriplaneDecoder(nn.Module):
    """
    Three 2D triplane feature maps -> reconstructed 3D MAISI latent via unpatchify.

    Pipeline:
        {XY: [B, in_c, Hp, Wp], YZ: [B, in_c, Wp, Dp], XZ: [B, in_c, Hp, Dp]}
          -> Expand each plane to [B, in_c, Hp, Wp, Dp] and sum
        [B, in_c, Hp, Wp, Dp]
          -> n_res_blocks ResBlock3D (spatial resolution unchanged)
        [B, hidden, Hp, Wp, Dp]
          -> Unpatchify: ConvTranspose3d(hidden, out_channels, patch_size, stride=patch_size)
        [B, out_channels, H, W, D]

    Args:
        in_channels:  8    (triplane channel width from encoder)
        hidden:       32   (ResBlock channel count; can be larger than in_channels)
        out_channels: 4    (MAISI latent channels)
        latent_shape: (H, W, D)
        patch_size:   4    (must match encoder patch_size)
        n_res_blocks: 2    (ResBlock3D layers before unpatchify)
    """

    def __init__(
        self,
        in_channels: int = 8,
        hidden: int = 32,
        out_channels: int = 4,
        latent_shape: tuple[int, int, int] = (120, 120, 64),
        patch_size: int = 4,
        n_res_blocks: int = 2,
    ) -> None:
        super().__init__()
        H, W, D = latent_shape
        p = patch_size
        self.Hp, self.Wp, self.Dp = H // p, W // p, D // p

        # Project fused triplane features to hidden dim, then run ResBlocks.
        res_layers: list[nn.Module] = [ResBlock3D(in_channels, hidden)]
        for _ in range(n_res_blocks - 1):
            res_layers.append(ResBlock3D(hidden, hidden))
        self.res_blocks = nn.Sequential(*res_layers)

        # Unpatchify: ConvTranspose3d mirrors the patchify Conv3d exactly.
        self.unpatchify = nn.ConvTranspose3d(
            hidden, out_channels, kernel_size=p, stride=p
        )

    def forward(
        self,
        z_xy: torch.Tensor,
        z_yz: torch.Tensor,
        z_xz: torch.Tensor,
    ) -> torch.Tensor:
        # z_xy: [B, C, Hp, Wp]  — collapses Dp axis
        # z_yz: [B, C, Wp, Dp]  — collapses Hp axis
        # z_xz: [B, C, Hp, Dp]  — collapses Wp axis
        B, C, Hp, Wp = z_xy.shape
        Dp = z_yz.shape[3]

        # Expand XY: [B, C, Hp, Wp] -> [B, C, Hp, Wp, Dp]
        xy3d = z_xy.unsqueeze(4).expand(B, C, Hp, Wp, Dp).contiguous()

        # Expand YZ: [B, C, Wp, Dp] -> [B, C, Hp, Wp, Dp]
        yz3d = z_yz.unsqueeze(2).expand(B, C, Hp, Wp, Dp).contiguous()

        # Expand XZ: [B, C, Hp, Dp] -> [B, C, Hp, Wp, Dp]
        xz3d = z_xz.unsqueeze(3).expand(B, C, Hp, Wp, Dp).contiguous()

        fused = xy3d + yz3d + xz3d  # [B, C, Hp, Wp, Dp]
        x = self.res_blocks(fused)  # [B, hidden, Hp, Wp, Dp]
        return self.unpatchify(x)  # [B, out_channels, H, W, D]
