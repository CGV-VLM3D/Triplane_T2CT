from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.layers import Act


__all__ = ["TriVQAEConv"]


def _norm(c: int) -> nn.Module:
    return nn.GroupNorm(num_groups=min(8, c), num_channels=c)


class ResBlock3D(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.norm1 = _norm(in_c)
        self.conv1 = nn.Conv3d(in_c, out_c, kernel_size=3, padding=1)
        self.norm2 = _norm(out_c)
        self.conv2 = nn.Conv3d(out_c, out_c, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.skip = nn.Conv3d(in_c, out_c, kernel_size=1) if in_c != out_c else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


class AxisOnlyDown(nn.Module):
    def __init__(self, in_c: int, out_c: int, agg_axis: int):
        super().__init__()
        stride = [1, 1, 1]
        stride[agg_axis] = 2
        self.conv = nn.Conv3d(in_c, out_c, kernel_size=3,
                              stride=tuple(stride), padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AxisOnlyUp(nn.Module):
    def __init__(self, in_c: int, out_c: int, agg_axis: int):
        super().__init__()
        scale = [1.0, 1.0, 1.0]
        scale[agg_axis] = 2.0
        self.scale_factor = tuple(scale)
        self.conv = nn.Conv3d(in_c, out_c, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=self.scale_factor,
                          mode="trilinear", align_corners=False)
        return self.conv(x)


def _expand_channels(channels: Sequence[int], n_stages: int) -> list[int]:
    chs = list(channels)
    while len(chs) < n_stages:
        chs.append(chs[-1])
    return chs[:n_stages]


class AxisEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: Sequence[int],
        blocks_per_stage: int,
        plane_channels: int,
        agg_axis: int,
        agg_size: int,
        non_agg_shape: tuple[int, int],
    ):
        super().__init__()
        if agg_axis not in (0, 1, 2):
            raise ValueError(f"agg_axis must be 0/1/2, got {agg_axis}")
        self.agg_axis = agg_axis
        self.agg_axis_tensor = agg_axis + 2

        # Number of axis-only down stages until the aggregate axis is small.
        n_down = max(1, int(math.ceil(math.log2(max(2, agg_size)))))

        chs = _expand_channels(hidden_channels, n_down + 1)

        layers: list[nn.Module] = []
        prev_ch = in_channels
        # n_down stages, each: ResBlock x N + AxisOnlyDown.
        for i in range(n_down):
            ch = chs[i]
            layers.append(ResBlock3D(prev_ch, ch))
            for _ in range(blocks_per_stage - 1):
                layers.append(ResBlock3D(ch, ch))
            layers.append(AxisOnlyDown(ch, ch, agg_axis))
            prev_ch = ch
        # Final ResBlocks at the smallest agg-axis size (still full non-agg).
        ch = chs[-1]
        layers.append(ResBlock3D(prev_ch, ch))
        for _ in range(blocks_per_stage - 1):
            layers.append(ResBlock3D(ch, ch))
        layers.append(nn.Conv3d(ch, plane_channels, kernel_size=1))
        self.body = nn.Sequential(*layers)

        # Final AdaptiveAvgPool: leave the non-agg dims at full resolution,
        # collapse the agg dim to 1 (handles non-power-of-2 leftovers).
        target_3d = list(non_agg_shape)
        target_3d.insert(agg_axis, 1)
        self.target_3d_shape = tuple(target_3d)
        self.pool = nn.AdaptiveAvgPool3d(self.target_3d_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.body(x)
        x = self.pool(x)
        return x.squeeze(self.agg_axis_tensor)


class AxisDecoder(nn.Module):
    def __init__(
        self,
        plane_channels: int,
        hidden_channels: Sequence[int],
        blocks_per_stage: int,
        out_channels: int,
        agg_axis: int,
        agg_size: int,
        target_3d_shape: Sequence[int],
    ):
        super().__init__()
        if agg_axis not in (0, 1, 2):
            raise ValueError(f"agg_axis must be 0/1/2, got {agg_axis}")
        self.agg_axis = agg_axis
        self.agg_axis_tensor = agg_axis + 2
        self.target_3d_shape = tuple(target_3d_shape)

        n_up = max(1, int(math.ceil(math.log2(max(2, agg_size)))))
        chs = _expand_channels(hidden_channels, n_up + 1)

        layers: list[nn.Module] = []
        prev_ch = plane_channels
        for i in range(n_up):
            ch = chs[i]
            layers.append(AxisOnlyUp(prev_ch, ch, agg_axis))
            layers.append(ResBlock3D(ch, ch))
            for _ in range(blocks_per_stage - 1):
                layers.append(ResBlock3D(ch, ch))
            prev_ch = ch
        ch = chs[-1]
        layers.append(ResBlock3D(prev_ch, ch))
        for _ in range(blocks_per_stage - 1):
            layers.append(ResBlock3D(ch, ch))
        layers.append(nn.Conv3d(ch, out_channels, kernel_size=1))
        self.body = nn.Sequential(*layers)

    def forward(self, plane_2d: torch.Tensor) -> torch.Tensor:
        x = plane_2d.unsqueeze(self.agg_axis_tensor)
        x = self.body(x)
        if tuple(x.shape[2:]) != self.target_3d_shape:
            x = F.interpolate(x, size=self.target_3d_shape,
                              mode="trilinear", align_corners=False)
        return x


class TriVQAEConv(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int | None = None,
        latent_shape: Sequence[int] = (120, 120, 64),
        encoder_channels: Sequence[int] = (16, 32, 64, 64, 64, 64),
        blocks_per_stage: int = 2,
        plane_channels: int = 64,
        decoder_channels: Sequence[int] | None = None,
        final_hidden_channels: int = 64,
        final_depth: int = 2,
        output_act: str | None = None,
        use_checkpointing: bool = False,
        **_unused_kwargs,
    ) -> None:
        super().__init__()

        latent_shape = tuple(int(v) for v in latent_shape)
        if len(latent_shape) != 3:
            raise ValueError(f"latent_shape must be (D, H, W); got {latent_shape}")
        D, H, W = latent_shape

        if decoder_channels is None:
            decoder_channels = tuple(reversed(list(encoder_channels)))

        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.latent_shape = latent_shape
        self.plane_channels = plane_channels
        self.output_act = output_act
        self.use_checkpointing = use_checkpointing

        # Plane shapes: full non-agg resolution.
        # DH plane (agg W): non_agg = (D, H)
        # DW plane (agg H): non_agg = (D, W)
        # HW plane (agg D): non_agg = (H, W)
        self.enc_dh = AxisEncoder(in_channels, encoder_channels, blocks_per_stage,
                                   plane_channels, agg_axis=2, agg_size=W,
                                   non_agg_shape=(D, H))
        self.enc_dw = AxisEncoder(in_channels, encoder_channels, blocks_per_stage,
                                   plane_channels, agg_axis=1, agg_size=H,
                                   non_agg_shape=(D, W))
        self.enc_hw = AxisEncoder(in_channels, encoder_channels, blocks_per_stage,
                                   plane_channels, agg_axis=0, agg_size=D,
                                   non_agg_shape=(H, W))

        self.dec_dh = AxisDecoder(plane_channels, decoder_channels, blocks_per_stage,
                                   plane_channels, agg_axis=2, agg_size=W,
                                   target_3d_shape=latent_shape)
        self.dec_dw = AxisDecoder(plane_channels, decoder_channels, blocks_per_stage,
                                   plane_channels, agg_axis=1, agg_size=H,
                                   target_3d_shape=latent_shape)
        self.dec_hw = AxisDecoder(plane_channels, decoder_channels, blocks_per_stage,
                                   plane_channels, agg_axis=0, agg_size=D,
                                   target_3d_shape=latent_shape)

        if final_depth < 1:
            raise ValueError("final_depth must be >= 1")
        layers: list[nn.Module] = []
        c = plane_channels
        for _ in range(final_depth - 1):
            layers.append(nn.Conv3d(c, final_hidden_channels, kernel_size=3, padding=1))
            layers.append(_norm(final_hidden_channels))
            layers.append(nn.SiLU())
            c = final_hidden_channels
        layers.append(nn.Conv3d(c, self.out_channels, kernel_size=3, padding=1))
        self.final = nn.Sequential(*layers)

    def _maybe_ckpt(self, module, x):
        if self.use_checkpointing and self.training and x.requires_grad:
            return torch.utils.checkpoint.checkpoint(module, x, use_reentrant=False)
        return module(x)

    def encode(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "dh": self._maybe_ckpt(self.enc_dh, z),
            "dw": self._maybe_ckpt(self.enc_dw, z),
            "hw": self._maybe_ckpt(self.enc_hw, z),
        }

    def decode_planes(self, planes: dict[str, torch.Tensor]) -> torch.Tensor:
        x_dh = self._maybe_ckpt(self.dec_dh, planes["dh"])
        x_dw = self._maybe_ckpt(self.dec_dw, planes["dw"])
        x_hw = self._maybe_ckpt(self.dec_hw, planes["hw"])
        return x_dh + x_dw + x_hw

    def forward(self, z: torch.Tensor, return_dict: bool = False):
        planes = self.encode(z)
        z_fused = self.decode_planes(planes)
        out = self.final(z_fused)

        if self.output_act is not None:
            out = Act[self.output_act](out)

        zero_loss = z.new_zeros(())

        if return_dict:
            return {
                "reconstruction": out,
                "q_loss": zero_loss,
                "z_fused": z_fused,
                "planes": planes,
            }
        return out, zero_loss

