"""TriplaneDecoderCompress3D: low-res triplane latent → reconstructed 3D volume.

Mirrors the Compress3D encoder (paper Fig 5):

  Input:  triplane latent {xy, yz, xz} at low-res (H/4, W/4, D/4)
    → TriConv(latent_channels → c3, k=1)                   # expand channels
    → per-plane ResBlock×n3 (c3 → c3)                       # paper: ×5 at 128
    → 2× UpSample (bilinear)
    → per-plane ResBlock×n2 (c3 → c2)                       # paper: ×3 at 64
    → 2× UpSample
    → per-plane ResBlock×n1 (c2 → c1)                       # paper: ×3 at 32
  High-res triplane T {xy: [B, c1, H, W], yz: [B, c1, H, D], xz: [B, c1, W, D]}
    → broadcast each plane to [B, c1, H, W, D]
    → concat → Conv3d(3·c1 → c1, k=3) + norm + act           # 3D fusion
    → Conv3d(c1 → out_channels, k=1)                         # output projection
  Output: [B, out_channels, H, W, D]

Plane convention (matches the encoder):
    "xy"  [B, C, H, W]   missing axis = D
    "yz"  [B, C, H, D]   missing axis = W
    "xz"  [B, C, W, D]   missing axis = H
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(c: int) -> nn.Module:
    return nn.GroupNorm(num_groups=min(8, c), num_channels=c)


class _ResBlock2D(nn.Module):
    """GroupNorm + SiLU + Conv2d(k=3,p=1) twice, with 1x1 skip when shapes differ."""

    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.norm1 = _norm(in_c)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.norm2 = _norm(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.skip = (
            nn.Conv2d(in_c, out_c, kernel_size=1) if in_c != out_c else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


def _stage_blocks(in_c: int, out_c: int, n: int) -> nn.Sequential:
    """First block in_c→out_c, then (n-1) blocks out_c→out_c."""
    layers: list[nn.Module] = [_ResBlock2D(in_c, out_c)]
    for _ in range(n - 1):
        layers.append(_ResBlock2D(out_c, out_c))
    return nn.Sequential(*layers)


class _UpStage(nn.Module):
    """ResBlock stack at one resolution, optionally followed by 2x bilinear upsample."""

    def __init__(self, in_c: int, out_c: int, n_blocks: int, upsample: bool) -> None:
        super().__init__()
        self.blocks = _stage_blocks(in_c, out_c, n_blocks)
        self.upsample = upsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return x


class _ThreeStageDecoder(nn.Module):
    """Per-plane stack mirroring the encoder.

    Encoder progression: c1 → c2 → c3 (with downsamples after the first
    ``n_upsamples`` stages). Decoder progression: c3 → c2 → c1, with
    upsamples after the corresponding stages so the spatial dims invert.
    ``n_upsamples`` must match the encoder's ``n_triplane_downsamples``.
    """

    def __init__(
        self,
        channels: tuple[int, int, int],
        n_blocks: tuple[int, int, int],
        n_upsamples: int,
    ) -> None:
        super().__init__()
        c1, c2, c3 = channels
        n1, n2, n3 = n_blocks
        # Symmetric to encoder: down_after[i] = (i < n) for i in 0,1,2.
        # Mirrors give the same mask for the decoder.
        self.stage1 = _UpStage(c3, c3, n3, upsample=(n_upsamples >= 1))
        self.stage2 = _UpStage(c3, c2, n2, upsample=(n_upsamples >= 2))
        self.stage3 = _UpStage(c2, c1, n1, upsample=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x


class TriplaneDecoderCompress3D(nn.Module):
    """Compress3D decoder. Symmetric to ``TriplaneEncoderCompress3D``.

    Args:
        latent_shape:               (H, W, D) target 3D output size.
        plane_channels_progression: (c1, c2, c3) — must match the encoder.
        n_blocks_per_stage:         (n1, n2, n3) blocks per resolution level.
                                    Paper Fig 5 uses (3, 3, 5) (mirror of
                                    encoder's (2, 2, 4) with an extra block
                                    at the bottleneck). Default keeps it
                                    configurable.
        latent_channels:            channel dim of the input triplane latent.
        out_channels:               output 3D channels (4 for MAISI latent).
    """

    def __init__(
        self,
        latent_shape: tuple[int, int, int] = (60, 60, 32),
        plane_channels_progression: tuple[int, int, int] = (32, 64, 128),
        n_blocks_per_stage: tuple[int, int, int] = (3, 3, 5),
        latent_channels: int = 32,
        out_channels: int = 4,
        n_upsamples: int = 2,
    ) -> None:
        super().__init__()
        if n_upsamples not in (0, 1, 2):
            raise ValueError(f"n_upsamples must be 0, 1, or 2; got {n_upsamples}")
        H, W, D = (int(v) for v in latent_shape)
        self.H, self.W, self.D = H, W, D
        c1, c2, c3 = plane_channels_progression
        self.c1 = int(c1)
        self.c3 = int(c3)

        # Per-plane: 1x1 Conv2d to expand latent_channels → c3 (paper's TriConv k=1).
        # Independent 1x1 convs per plane is equivalent to TriConv(k=1) without
        # cross-plane mean injection, which makes the upsample stages mirror the
        # encoder's _ThreeStageEncoder (independent 2D stacks).
        self.expand_xy = nn.Conv2d(latent_channels, c3, kernel_size=1)
        self.expand_yz = nn.Conv2d(latent_channels, c3, kernel_size=1)
        self.expand_xz = nn.Conv2d(latent_channels, c3, kernel_size=1)

        # Per-plane upsample stacks (n_upsamples mirrors the encoder).
        self.dec_xy = _ThreeStageDecoder(
            plane_channels_progression, n_blocks_per_stage, n_upsamples
        )
        self.dec_yz = _ThreeStageDecoder(
            plane_channels_progression, n_blocks_per_stage, n_upsamples
        )
        self.dec_xz = _ThreeStageDecoder(
            plane_channels_progression, n_blocks_per_stage, n_upsamples
        )

        # 3D fusion: broadcast each plane to [B, c1, H, W, D], concat, Conv3d.
        self.fuse_conv = nn.Conv3d(3 * c1, c1, kernel_size=3, padding=1)
        self.fuse_norm = _norm(c1)
        self.fuse_act = nn.SiLU()
        self.out_conv = nn.Conv3d(c1, out_channels, kernel_size=1)

    def forward(self, planes: dict[str, torch.Tensor]) -> torch.Tensor:
        """Args:
            planes: {"xy": [B, lc, rH, rW], "yz": [B, lc, rH, rD], "xz": [B, lc, rW, rD]}
                    where (rH, rW, rD) = (H/4, W/4, D/4).
        Returns:
            [B, out_channels, H, W, D]
        """
        H, W, D = self.H, self.W, self.D

        # Channel expand per plane.
        xy = self.expand_xy(planes["xy"])
        yz = self.expand_yz(planes["yz"])
        xz = self.expand_xz(planes["xz"])

        # Per-plane upsample stacks (4× spatial restore).
        xy = self.dec_xy(xy)  # [B, c1, H, W]
        yz = self.dec_yz(yz)  # [B, c1, H, D]
        xz = self.dec_xz(xz)  # [B, c1, W, D]

        B, C = xy.shape[:2]

        # Broadcast each plane to 3D and concatenate channels.
        xy3d = xy.unsqueeze(-1).expand(B, C, H, W, D)  # expand D
        yz3d = yz.unsqueeze(3).expand(B, C, H, W, D)  # expand W
        xz3d = xz.unsqueeze(2).expand(B, C, H, W, D)  # expand H

        fused = torch.cat([xy3d, yz3d, xz3d], dim=1)
        fused = self.fuse_act(self.fuse_norm(self.fuse_conv(fused)))
        return self.out_conv(fused)
