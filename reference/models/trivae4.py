"""
TriPlaneExpandAE — uses trivae2's triplane encoder + an EG3D-style
explicit 3D-volume decoder that broadcasts each 2D plane back to the full
latent volume, sums (or concats) them coordinate-wise, then refines the
combined feature volume with 3D residual convolutions and reduces the
channel count down to out_channels.

Pipeline:
    z [B, in_c, D, H, W]
      -> 3 AxisEncoders + CrossPlaneMixer (from trivae2)
      -> planes:
            dh: [B, C_p, D, H]
            dw: [B, C_p, D, W]
            hw: [B, C_p, H, W]
      -> Expand each to [B, C_p, D, H, W] by broadcasting along its missing
         axis, then sum (or concat -> [B, 3*C_p, D, H, W]).
      -> Stack of ResBlock3D with optionally tapering channel widths.
      -> 1x1x1 Conv3d -> out_channels.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from monai.networks.layers import Act

from models.trivae2 import AxisEncoder, CrossPlaneMixer, ResBlock3D


__all__ = ["TriPlaneExpandAE", "PlaneExpandSum", "RefineDecoder3D"]


# ---------------------------------------------------------------------------
# Plane → full-volume broadcast + fusion.
# ---------------------------------------------------------------------------
class PlaneExpandSum(nn.Module):
    """
    Expand the three plane features back to the full 3D latent shape by
    broadcasting along the originally-aggregated axis, then sum (or concat)
    coordinate-wise.

    fusion='sum'    -> [B, C_plane, D, H, W]
    fusion='concat' -> [B, 3*C_plane, D, H, W]
    """

    def __init__(self, latent_shape: Sequence[int], fusion: str = "sum"):
        super().__init__()
        if fusion not in ("sum", "concat"):
            raise ValueError(f"fusion must be 'sum' or 'concat', got {fusion}")
        D, H, W = (int(v) for v in latent_shape)
        self.latent_shape = (D, H, W)
        self.fusion = fusion

    @property
    def fused_channels_multiplier(self) -> int:
        return 1 if self.fusion == "sum" else 3

    def forward(self, planes: dict[str, torch.Tensor]) -> torch.Tensor:
        D, H, W = self.latent_shape
        p_dh = planes["dh"]  # [B, C, D, H]  -> add W
        p_dw = planes["dw"]  # [B, C, D, W]  -> add H
        p_hw = planes["hw"]  # [B, C, H, W]  -> add D

        v_dh = p_dh.unsqueeze(-1).expand(-1, -1, -1, -1, W)
        v_dw = p_dw.unsqueeze(3).expand(-1, -1, -1, H, -1)
        v_hw = p_hw.unsqueeze(2).expand(-1, -1, D, -1, -1)

        if self.fusion == "sum":
            return v_dh + v_dw + v_hw
        return torch.cat([v_dh, v_dw, v_hw], dim=1)


# ---------------------------------------------------------------------------
# 3D refinement decoder (no spatial down/up; channel taper only).
# ---------------------------------------------------------------------------
class RefineDecoder3D(nn.Module):
    """
    Stack of ResBlock3D over a fixed spatial grid (D, H, W) with channel
    widths given by `block_channels`. The first block bridges from the input
    feature width (from PlaneExpandSum) to block_channels[0]. The final 1x1x1
    conv reduces to out_channels.

    Example:
        input_channels = 32, block_channels = [32, 16, 8], out_channels = 4
            ResBlock(32 -> 32)
            ResBlock(32 -> 16)
            ResBlock(16 -> 8)
            Conv1x1(8 -> 4)
    """

    def __init__(
        self,
        input_channels: int,
        block_channels: Sequence[int],
        out_channels: int,
    ):
        super().__init__()
        block_channels = [int(c) for c in block_channels]
        if not block_channels:
            raise ValueError("block_channels must have at least 1 entry")

        blocks: list[nn.Module] = []
        prev = input_channels
        for ch in block_channels:
            blocks.append(ResBlock3D(prev, ch))
            prev = ch
        self.blocks = nn.Sequential(*blocks)

        self.proj = nn.Conv3d(prev, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.blocks(x))


# ---------------------------------------------------------------------------
# Full autoencoder: triplane encoder + expand-and-refine decoder.
# ---------------------------------------------------------------------------
class TriPlaneExpandAE(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int | None = None,
        latent_shape: Sequence[int] = (120, 120, 64),
        # encoder (mirrors trivae2: stage count = len(encoder_channels))
        encoder_channels: Sequence[int] = (16, 16, 32, 32),
        blocks_per_stage: int = 2,
        plane_channels: int = 32,
        # decoder
        fusion: str = "sum",                            # "sum" | "concat"
        decoder_block_channels: Sequence[int] = (32, 16, 8),
        output_act: str | None = None,
        **_unused,
    ):
        super().__init__()
        D, H, W = (int(v) for v in latent_shape)
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.latent_shape = (D, H, W)
        self.output_act = output_act

        n_stages_down = max(1, len(encoder_channels) - 1)

        self.enc_dh = AxisEncoder(in_channels, encoder_channels, blocks_per_stage,
                                  plane_channels, agg_axis=2,
                                  non_agg_shape=(D, H),
                                  n_down_stages=n_stages_down)
        self.enc_dw = AxisEncoder(in_channels, encoder_channels, blocks_per_stage,
                                  plane_channels, agg_axis=1,
                                  non_agg_shape=(D, W),
                                  n_down_stages=n_stages_down)
        self.enc_hw = AxisEncoder(in_channels, encoder_channels, blocks_per_stage,
                                  plane_channels, agg_axis=0,
                                  non_agg_shape=(H, W),
                                  n_down_stages=n_stages_down)

        self.enc_mixers = nn.ModuleList(
            CrossPlaneMixer(self.enc_dh.stage_out_channels[i])
            for i in range(self.enc_dh.n_stages)
        )

        self.expand = PlaneExpandSum(latent_shape=self.latent_shape, fusion=fusion)
        decoder_in = plane_channels * self.expand.fused_channels_multiplier

        self.decoder = RefineDecoder3D(
            input_channels=decoder_in,
            block_channels=decoder_block_channels,
            out_channels=self.out_channels,
        )

    def encode(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        x_dh, x_dw, x_hw = z, z, z
        for i in range(self.enc_dh.n_stages):
            x_dh = self.enc_dh.run_stage(i, x_dh)
            x_dw = self.enc_dw.run_stage(i, x_dw)
            x_hw = self.enc_hw.run_stage(i, x_hw)
            x_dh, x_dw, x_hw = self.enc_mixers[i](x_dh, x_dw, x_hw)
        return {
            "dh": self.enc_dh.finalize(x_dh),
            "dw": self.enc_dw.finalize(x_dw),
            "hw": self.enc_hw.finalize(x_hw),
        }

    def decode(self, planes: dict[str, torch.Tensor]) -> torch.Tensor:
        v = self.expand(planes)
        out = self.decoder(v)
        if self.output_act is not None:
            out = Act[self.output_act](out)
        return out

    def forward(self, z: torch.Tensor, return_dict: bool = False):
        planes = self.encode(z)
        out = self.decode(planes)
        zero_loss = z.new_zeros(())
        if return_dict:
            return {
                "reconstruction": out,
                "q_loss": zero_loss,
                "planes": planes,
            }
        return out, zero_loss
