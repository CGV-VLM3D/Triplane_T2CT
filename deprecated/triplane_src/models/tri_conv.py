"""TriConv: 3D-aware 2D convolution ported from Rodin (RodinHD/pretrained_diffusion/nn.py).

RodinHD Conv3DAware uses AdaptiveAvgPool2d((1,None)) / AdaptiveAvgPool2d((None,1))
to collapse one axis to a 1-D vector, then expand back. Because RodinHD uses square
planes (all H×W), both operations are interchangeable in shape. Our planes are
asymmetric (xy:[H,W], yz:[H,D], xz:[W,D]), so we apply axis-aware 1-D mean: collapse
the non-shared axis, keep the shared axis, then expand/permute to match the target.

Plane naming convention (matching the encoder/decoder key names):
    "xy"  -> [B, C, H, W]   = [B, C, 120, 120]
    "yz"  -> [B, C, H, D]   = [B, C, 120, 64]
    "xz"  -> [B, C, W, D]   = [B, C, 120, 64]

where H=120, W=120, D=64 (MAISI latent spatial dims).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _norm(c: int) -> nn.Module:
    return nn.GroupNorm(num_groups=min(8, c), num_channels=c)


class TriConv(nn.Module):
    """3D-aware 2D convolution over a triplane dict.

    For each output plane the input is formed by concatenating the target
    plane with mean projections from the other two planes resized to match
    (using adaptive_avg_pool2d when spatial dims differ).  This lets each
    plane's conv kernel see context from the other two axes.

    Args:
        in_channels:  input channel count per plane (C)
        out_channels: output channel count per plane
        kernel_size:  2D conv kernel size (1 for Q/head projections, 3 for ResBlock)
        stride:       2D conv stride (default 1)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
    ) -> None:
        super().__init__()
        pad = kernel_size // 2
        # Each plane receives its own tensor (self + 2 projections → 3*C in).
        self.conv_xy = nn.Conv2d(
            in_channels * 3, out_channels, kernel_size, stride=stride, padding=pad
        )
        self.conv_yz = nn.Conv2d(
            in_channels * 3, out_channels, kernel_size, stride=stride, padding=pad
        )
        self.conv_xz = nn.Conv2d(
            in_channels * 3, out_channels, kernel_size, stride=stride, padding=pad
        )

    def forward(self, planes: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        xy = planes["xy"]  # [B, C, H, W]
        yz = planes["yz"]  # [B, C, H, D]
        xz = planes["xz"]  # [B, C, W, D]

        _, _, H, W = xy.shape
        _, _, _, D = yz.shape

        # XY plane: shared axis with yz=H(dim2), collapse D(dim3);
        #           shared axis with xz=W(dim3), collapse D(dim3) then permute.
        yz_for_xy = yz.mean(dim=3, keepdim=True).expand(-1, -1, H, W)
        xz_for_xy = (
            xz.mean(dim=3, keepdim=True).permute(0, 1, 3, 2).expand(-1, -1, H, W)
        )
        xy_out = self.conv_xy(torch.cat([xy, yz_for_xy, xz_for_xy], dim=1))

        # YZ plane: shared axis with xy=H(dim2), collapse W(dim3);
        #           shared axis with xz=D(dim3), collapse W(dim2).
        xy_for_yz = xy.mean(dim=3, keepdim=True).expand(-1, -1, H, D)
        xz_for_yz = xz.mean(dim=2, keepdim=True).expand(-1, -1, H, D)
        yz_out = self.conv_yz(torch.cat([yz, xy_for_yz, xz_for_yz], dim=1))

        # XZ plane: shared axis with xy=W(dim3→dim2), collapse H(dim2) then permute;
        #           shared axis with yz=D(dim3), collapse H(dim2).
        xy_for_xz = (
            xy.mean(dim=2, keepdim=True).permute(0, 1, 3, 2).expand(-1, -1, W, D)
        )
        yz_for_xz = yz.mean(dim=2, keepdim=True).expand(-1, -1, W, D)
        xz_out = self.conv_xz(torch.cat([xz, xy_for_xz, yz_for_xz], dim=1))

        return {"xy": xy_out, "yz": yz_out, "xz": xz_out}


class TriConvResBlock(nn.Module):
    """2D ResBlock using TriConv for the interior convolutions."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        # Norms and activations are per-plane (standard 2D GroupNorm).
        self.norm1_xy = _norm(in_channels)
        self.norm1_yz = _norm(in_channels)
        self.norm1_xz = _norm(in_channels)
        self.norm2_xy = _norm(out_channels)
        self.norm2_yz = _norm(out_channels)
        self.norm2_xz = _norm(out_channels)
        self.act = nn.SiLU()

        self.tri1 = TriConv(in_channels, out_channels, kernel_size=3)
        self.tri2 = TriConv(out_channels, out_channels, kernel_size=3)

        if in_channels != out_channels:
            # 1×1 TriConv for the skip (k=1 so no cross-plane mixing, just projection).
            self.skip = TriConv(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = None

    def forward(self, planes: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # Norm + act per plane, then TriConv.
        normed1 = {
            "xy": self.act(self.norm1_xy(planes["xy"])),
            "yz": self.act(self.norm1_yz(planes["yz"])),
            "xz": self.act(self.norm1_xz(planes["xz"])),
        }
        h = self.tri1(normed1)

        normed2 = {
            "xy": self.act(self.norm2_xy(h["xy"])),
            "yz": self.act(self.norm2_yz(h["yz"])),
            "xz": self.act(self.norm2_xz(h["xz"])),
        }
        h = self.tri2(normed2)

        if self.skip is not None:
            skip_out = self.skip(planes)
        else:
            skip_out = planes

        return {
            "xy": h["xy"] + skip_out["xy"],
            "yz": h["yz"] + skip_out["yz"],
            "xz": h["xz"] + skip_out["xz"],
        }
