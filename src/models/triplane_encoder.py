"""TriplaneEncoder: 3D MAISI latent -> three 2D triplane feature maps.

Architecture (Compress3D-style):
    1. 3-axis strided Conv3d projects the 3D volume to three 2D planes by
       collapsing one axis entirely (kernel = full size on that axis).
    2. Per-plane 2D ResBlock stacks mix spatial information within each plane.
    3. Optional CrossAttention3D block re-introduces volume context.
    4. Two TriConv(k=1) heads produce per-plane mu and logvar for the VAE.

Plane naming (matches tri_conv.py):
    "xy"  [B, C, H, W]   collapsed axis = D (last of latent_shape)
    "yz"  [B, C, H, D]   collapsed axis = W (middle of latent_shape)
    "xz"  [B, C, W, D]   collapsed axis = H (first of latent_shape)

where latent_shape = (H, W, D).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .cross_attention_3d import CrossAttention3D
from .tri_conv import TriConv, TriConvResBlock


def _norm(c: int) -> nn.Module:
    return nn.GroupNorm(num_groups=min(8, c), num_channels=c)


class _ResBlock2D(nn.Module):
    """Standard 2D ResBlock: GroupNorm + SiLU + Conv2d(k=3,p=1), with skip."""

    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.norm1 = _norm(in_c)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.norm2 = _norm(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.skip = nn.Conv2d(in_c, out_c, kernel_size=1) if in_c != out_c else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


def _make_res_stack(in_c: int, out_c: int, n: int) -> nn.Sequential:
    layers: list[nn.Module] = [_ResBlock2D(in_c, out_c)]
    for _ in range(n - 1):
        layers.append(_ResBlock2D(out_c, out_c))
    return nn.Sequential(*layers)


class TriplaneEncoder(nn.Module):
    """
    3D MAISI latent -> per-plane (mu, logvar) triplane features.

    Args:
        in_channels:        input channel count (4 for MAISI latent).
        latent_shape:       (H, W, D) spatial dims of the 3D input.
        plane_channels:     intermediate channel width for plane processing.
        latent_channels:    output channels for mu/logvar heads.
        n_resblocks:        (n1, n2, n3) ResBlock counts for the three 2D stacks.
        downsampling:       if True, align volume K/V via strided Conv3d before attn.
        use_cross_attn:     whether to apply CrossAttention3D.
        cross_attn_heads:   heads for CrossAttention3D.
        cross_attn_d_kv:    per-head K/V dim for CrossAttention3D.
        cross_attn_window:  window size (only 1 supported).
    """

    def __init__(
        self,
        in_channels: int = 4,
        latent_shape: tuple[int, int, int] = (60, 60, 32),
        plane_channels: int = 16,
        latent_channels: int = 16,
        n_resblocks: tuple[int, int, int] = (2, 2, 4),
        downsampling: bool = False,
        use_cross_attn: bool = True,
        cross_attn_heads: int = 4,
        cross_attn_d_kv: int = 16,
        cross_attn_window: int = 1,
    ) -> None:
        super().__init__()
        self.latent_shape = tuple(latent_shape)
        H, W, D = latent_shape
        self.H, self.W, self.D = H, W, D
        self.use_cross_attn = use_cross_attn
        self.downsampling = downsampling

        # --- Axis-collapsing Conv3d projections (paper Eq 3-5) ---
        # XY: collapse D axis → [B, plane_channels, H, W]
        self.proj_xy = nn.Conv3d(in_channels, plane_channels, kernel_size=(1, 1, D), stride=(1, 1, D))
        # YZ: collapse W axis → [B, plane_channels, H, D]
        self.proj_yz = nn.Conv3d(in_channels, plane_channels, kernel_size=(1, W, 1), stride=(1, W, 1))
        # XZ: collapse H axis → [B, plane_channels, W, D]  (then squeeze dim 2)
        self.proj_xz = nn.Conv3d(in_channels, plane_channels, kernel_size=(H, 1, 1), stride=(H, 1, 1))

        # --- Per-plane 2D ResBlock stacks (independent weights per plane) ---
        n1, n2, n3 = n_resblocks
        total = n1 + n2 + n3

        def _make_plane_stack() -> nn.Sequential:
            layers: list[nn.Module] = []
            # Stage 1
            layers.extend(list(_make_res_stack(plane_channels, plane_channels, n1)))
            # Stage 2
            layers.extend(list(_make_res_stack(plane_channels, plane_channels, n2)))
            # Stage 3
            layers.extend(list(_make_res_stack(plane_channels, plane_channels, n3)))
            return nn.Sequential(*layers)

        self.res_xy = _make_plane_stack()
        self.res_yz = _make_plane_stack()
        self.res_xz = _make_plane_stack()

        # --- Optional CrossAttention3D ---
        if use_cross_attn:
            # When downsampling, we could stride the volume; currently only no-downsampling path.
            # downsampling=True path: same structure, volume passed directly (no re-encode needed
            # because CrossAttention3D does its own kv_proj from the raw volume).
            self.cross_attn: nn.Module | None = CrossAttention3D(
                plane_channels=plane_channels,
                volume_in_channels=in_channels,
                d_kv=cross_attn_d_kv,
                heads=cross_attn_heads,
                window_size=cross_attn_window,
                latent_shape=latent_shape,
            )
        else:
            self.cross_attn = None

        # --- mu / logvar heads via TriConv(k=1) ---
        self.mu_head = TriConv(plane_channels, latent_channels, kernel_size=1)
        self.logvar_head = TriConv(plane_channels, latent_channels, kernel_size=1)

    def forward(self, z: torch.Tensor) -> dict:
        """
        Args:
            z: [B, in_channels, H, W, D]

        Returns:
            {
                "mu":     {"xy": [B, latent_channels, H, W],
                           "yz": [B, latent_channels, H, D],
                           "xz": [B, latent_channels, W, D]},
                "logvar": same structure,
            }
        """
        # --- Axis-collapse projections ---
        # proj_xy: [B, C, H, W, D] → [B, plane_channels, H, W, 1] → squeeze
        xy = self.proj_xy(z).squeeze(-1)      # [B, plane_channels, H, W]
        yz = self.proj_yz(z).squeeze(3)       # [B, plane_channels, H, D]  (squeeze W dim)
        xz = self.proj_xz(z).squeeze(2)       # [B, plane_channels, W, D]  (squeeze H dim)

        # --- Per-plane ResBlock stacks ---
        xy = self.res_xy(xy)
        yz = self.res_yz(yz)
        xz = self.res_xz(xz)

        planes = {"xy": xy, "yz": yz, "xz": xz}

        # --- Optional CrossAttention3D ---
        if self.cross_attn is not None:
            residual = self.cross_attn(planes, z)
            planes = {k: planes[k] + residual[k] for k in planes}

        # --- mu / logvar heads ---
        mu = self.mu_head(planes)
        logvar = self.logvar_head(planes)

        return {"mu": mu, "logvar": logvar}
