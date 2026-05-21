"""CrossAttention3D: volume-guided cross-attention for triplane features.

Each 2D plane queries the raw 3D volume along the plane's collapsed axis
(m=1 window = column-only).  A residual dict {"xy":.., "yz":.., "xz":..}
is returned so the caller can do T = T + CrossAttention3D(T, vol).

Axis conventions (matches tri_conv.py):
    "xy"  [B, C, H, W]  — collapsed axis is D (last)
    "yz"  [B, C, H, D]  — collapsed axis is W (middle)
    "xz"  [B, C, W, D]  — collapsed axis is H (first)

where latent_shape = (H, W, D).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

from .tri_conv import TriConv


class CrossAttention3D(nn.Module):
    """Volume-to-triplane cross-attention with 3D positional embeddings.

    For each query cell (i,j) on a plane, attends to a column of K/V tokens
    extracted from the 3D volume along the collapsed axis (window_size=1,
    i.e. column-only).  Residual output has the same shape as the input planes.

    Args:
        plane_channels:     C of the input triplane features.
        volume_in_channels: channels of the 3D volume being queried (4 for MAISI).
        d_kv:               per-head key/value dim.
        heads:              number of attention heads.
        window_size:        m in the spec; only m=1 (column-only) is implemented.
        latent_shape:       (H, W, D) — spatial dims of the 3D volume.
    """

    def __init__(
        self,
        plane_channels: int,
        volume_in_channels: int,
        d_kv: int = 16,
        heads: int = 4,
        window_size: int = 1,
        latent_shape: tuple[int, int, int] = (60, 60, 32),
    ) -> None:
        super().__init__()
        if window_size != 1:
            raise NotImplementedError("Only window_size=1 (column-only) is implemented.")

        self.plane_channels = plane_channels
        self.volume_in_channels = volume_in_channels
        self.d_kv = d_kv
        self.heads = heads
        H, W, D = latent_shape
        self.H, self.W, self.D = H, W, D

        inner = d_kv * heads  # total Q/K/V dim

        # Q projections per plane via TriConv(k=1) — cross-plane aware.
        self.q_proj = TriConv(plane_channels, inner, kernel_size=1)

        # Combined K/V projection from volume: 1×1×1 Conv3d → 2*inner channels.
        self.kv_proj = nn.Conv3d(volume_in_channels, 2 * inner, kernel_size=1)

        # 3D additive positional embeddings on the K/V feature map (before split).
        # Broadcast-sum: pe_x[H] + pe_y[W] + pe_z[D] covers the full 3D grid.
        self.pe_x = nn.Parameter(torch.zeros(1, 2 * inner, H, 1, 1))
        self.pe_y = nn.Parameter(torch.zeros(1, 2 * inner, 1, W, 1))
        self.pe_z = nn.Parameter(torch.zeros(1, 2 * inner, 1, 1, D))
        trunc_normal_(self.pe_x, std=0.02)
        trunc_normal_(self.pe_y, std=0.02)
        trunc_normal_(self.pe_z, std=0.02)

        # Output projection: inner → plane_channels, via TriConv(k=1).
        self.out_proj = TriConv(inner, plane_channels, kernel_size=1)

        self._scale = d_kv ** -0.5

    # ------------------------------------------------------------------
    # Column extraction helpers
    # ------------------------------------------------------------------

    def _attn_1d(
        self,
        q: torch.Tensor,    # [B*cells, 1, inner]
        kv: torch.Tensor,   # [B*cells, agg_len, 2*inner]
    ) -> torch.Tensor:
        """Standard multi-head attention for 1-token Q over agg_len K/V tokens.

        Returns [B*cells, inner].
        """
        B_cells = q.shape[0]
        h, d = self.heads, self.d_kv
        inner = h * d

        # [B_cells, 1, h, d] / [B_cells, agg_len, h, d]
        q_h = q.view(B_cells, 1, h, d)
        k, v = kv.split(inner, dim=-1)
        k_h = k.view(B_cells, -1, h, d)
        v_h = v.view(B_cells, -1, h, d)

        # [B_cells, h, 1, agg_len]
        attn = torch.einsum("bqhd,bkhd->bhqk", q_h, k_h) * self._scale
        attn = F.softmax(attn, dim=-1)
        # [B_cells, h, 1, d] → [B_cells, inner]
        out = torch.einsum("bhqk,bkhd->bqhd", attn, v_h)
        return out.reshape(B_cells, inner)

    def _attend_xy(
        self,
        q_xy: torch.Tensor,   # [B, inner, H, W]
        kv_vol: torch.Tensor,  # [B, 2*inner, H, W, D]
    ) -> torch.Tensor:
        """XY plane: each (i,j) queries column [:,:,i,j,:] of length D."""
        B, C, H, W = q_xy.shape
        D = self.D
        inner = self.heads * self.d_kv

        # Q: [B, H, W, inner] → [B*H*W, 1, inner]
        q = q_xy.permute(0, 2, 3, 1).reshape(B * H * W, 1, inner)

        # KV: [B, 2*inner, H, W, D] → [B, H, W, D, 2*inner] → [B*H*W, D, 2*inner]
        kv = kv_vol.permute(0, 2, 3, 4, 1).reshape(B * H * W, D, 2 * inner)

        out = self._attn_1d(q, kv)  # [B*H*W, inner]
        return out.reshape(B, H, W, inner).permute(0, 3, 1, 2)  # [B, inner, H, W]

    def _attend_yz(
        self,
        q_yz: torch.Tensor,   # [B, inner, H, D]
        kv_vol: torch.Tensor,  # [B, 2*inner, H, W, D]
    ) -> torch.Tensor:
        """YZ plane: each (i,j) where i=H_idx, j=D_idx queries column [:,:,:,i,j] of length W."""
        B, C, H, D = q_yz.shape
        W = self.W
        inner = self.heads * self.d_kv

        # Q: [B, H, D, inner] → [B*H*D, 1, inner]
        q = q_yz.permute(0, 2, 3, 1).reshape(B * H * D, 1, inner)

        # KV: need column along W for each (H_idx, D_idx).
        # kv_vol: [B, 2*inner, H, W, D] → permute to [B, H, D, W, 2*inner]
        kv = kv_vol.permute(0, 2, 4, 3, 1).reshape(B * H * D, W, 2 * inner)

        out = self._attn_1d(q, kv)  # [B*H*D, inner]
        return out.reshape(B, H, D, inner).permute(0, 3, 1, 2)  # [B, inner, H, D]

    def _attend_xz(
        self,
        q_xz: torch.Tensor,   # [B, inner, W, D]
        kv_vol: torch.Tensor,  # [B, 2*inner, H, W, D]
    ) -> torch.Tensor:
        """XZ plane: each (i,j) where i=W_idx, j=D_idx queries column [:,:,:,i,j] = wait —
        XZ plane is [B,C,W,D], collapsed axis is H (first spatial). So query (i,j) with
        i=W_idx, j=D_idx attends to kv_vol[:,:,:,i,j] — tokens along H (length H)."""
        B, C, W, D = q_xz.shape
        H = self.H
        inner = self.heads * self.d_kv

        # Q: [B, W, D, inner] → [B*W*D, 1, inner]
        q = q_xz.permute(0, 2, 3, 1).reshape(B * W * D, 1, inner)

        # KV: kv_vol [B, 2*inner, H, W, D] → permute to [B, W, D, H, 2*inner]
        kv = kv_vol.permute(0, 3, 4, 2, 1).reshape(B * W * D, H, 2 * inner)

        out = self._attn_1d(q, kv)  # [B*W*D, inner]
        return out.reshape(B, W, D, inner).permute(0, 3, 1, 2)  # [B, inner, W, D]

    def forward(
        self,
        planes: dict[str, torch.Tensor],
        volume: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            planes: {"xy": [B,C,H,W], "yz": [B,C,H,D], "xz": [B,C,W,D]}
            volume: [B, volume_in_channels, H, W, D]

        Returns residual dict with same shapes as planes.
        """
        # Project volume to K/V feature map + add 3D positional embeddings.
        kv_feat = self.kv_proj(volume)  # [B, 2*inner, H, W, D]
        kv_feat = kv_feat + self.pe_x + self.pe_y + self.pe_z

        # Project planes to Q.
        q_planes = self.q_proj(planes)  # each [B, inner, plane_H, plane_W]

        # Per-plane column attention.
        a_xy = self._attend_xy(q_planes["xy"], kv_feat)
        a_yz = self._attend_yz(q_planes["yz"], kv_feat)
        a_xz = self._attend_xz(q_planes["xz"], kv_feat)

        attn_planes = {"xy": a_xy, "yz": a_yz, "xz": a_xz}

        # Project back to plane_channels.
        return self.out_proj(attn_planes)
