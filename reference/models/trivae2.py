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


class CrossPlaneMixer(nn.Module):
    """
    Exchanges information across the three plane tensors via
    concat + 1x1x1 Conv3d.

    Each plane Q (dh/dw/hw) has two of the three latent axes at full latent
    resolution (its non-agg axes) and one axis at a current (small) size
    (its agg axis). To inject plane P's information into plane Q:

      1. Mean over P's agg axis -> 2D summary in P's two non-agg axes
         (one of which is Q's agg axis, the other is the axis shared with
         Q's non-agg).
      2. Pool the axis-that-equals-Q's-agg from full latent resolution
         down to Q's current agg size (along that axis only).
      3. Insert a size-1 axis at P's agg position (Q's missing axis).
      4. Expand that size-1 axis to its full latent size.
      5. Concat Q with the two folded summaries along channels, then
         Conv1x1x1 to mix (3C -> C).
    """

    def __init__(self, channels: int):
        super().__init__()
        self.mix_dh = nn.Conv3d(channels * 3, channels, kernel_size=1)
        self.mix_dw = nn.Conv3d(channels * 3, channels, kernel_size=1)
        self.mix_hw = nn.Conv3d(channels * 3, channels, kernel_size=1)

    @staticmethod
    def _summary(x: torch.Tensor, agg_axis: int) -> torch.Tensor:
        return x.mean(dim=agg_axis + 2)

    @staticmethod
    def _fold(
        summary: torch.Tensor,
        src_agg: int,
        tgt_agg: int,
        tgt_spatial: tuple[int, int, int],
    ) -> torch.Tensor:
        src_non_agg = sorted({0, 1, 2} - {src_agg})
        tgt_idx_in_summary = src_non_agg.index(tgt_agg)

        new_sz = list(summary.shape[2:])
        new_sz[tgt_idx_in_summary] = tgt_spatial[tgt_agg]
        pooled = F.adaptive_avg_pool2d(summary, output_size=tuple(new_sz))

        tensor_axis = 2 + src_agg
        unsq = pooled.unsqueeze(tensor_axis)
        expand_shape = list(unsq.shape)
        expand_shape[tensor_axis] = tgt_spatial[src_agg]
        return unsq.expand(*expand_shape).contiguous()

    def forward(
        self,
        dh: torch.Tensor,
        dw: torch.Tensor,
        hw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        AGG_DH, AGG_DW, AGG_HW = 2, 1, 0

        sum_dh = self._summary(dh, AGG_DH)
        sum_dw = self._summary(dw, AGG_DW)
        sum_hw = self._summary(hw, AGG_HW)

        dh_sp = tuple(dh.shape[2:])
        dw_sp = tuple(dw.shape[2:])
        hw_sp = tuple(hw.shape[2:])

        dw_in_dh = self._fold(sum_dw, AGG_DW, AGG_DH, dh_sp)
        hw_in_dh = self._fold(sum_hw, AGG_HW, AGG_DH, dh_sp)

        dh_in_dw = self._fold(sum_dh, AGG_DH, AGG_DW, dw_sp)
        hw_in_dw = self._fold(sum_hw, AGG_HW, AGG_DW, dw_sp)

        dh_in_hw = self._fold(sum_dh, AGG_DH, AGG_HW, hw_sp)
        dw_in_hw = self._fold(sum_dw, AGG_DW, AGG_HW, hw_sp)

        dh_new = self.mix_dh(torch.cat([dh, dw_in_dh, hw_in_dh], dim=1))
        dw_new = self.mix_dw(torch.cat([dw, dh_in_dw, hw_in_dw], dim=1))
        hw_new = self.mix_hw(torch.cat([hw, dh_in_hw, dw_in_hw], dim=1))
        return dh_new, dw_new, hw_new


class AxisEncoder(nn.Module):
    """
    Per-axis encoder split into stages so an outer module can interleave
    cross-plane mixing between stages.

    n_down_stages stages of [ResBlock x B + AxisOnlyDown], plus one final
    stage of [ResBlock x B] (no Down). After all stages, `finalize` applies
    a 1x1 projection to plane_channels, adaptive-pools the agg axis to 1,
    and squeezes.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: Sequence[int],
        blocks_per_stage: int,
        plane_channels: int,
        agg_axis: int,
        non_agg_shape: tuple[int, int],
        n_down_stages: int,
    ):
        super().__init__()
        if agg_axis not in (0, 1, 2):
            raise ValueError(f"agg_axis must be 0/1/2, got {agg_axis}")
        self.agg_axis = agg_axis
        self.agg_axis_tensor = agg_axis + 2

        chs = _expand_channels(hidden_channels, n_down_stages + 1)
        self.stages = nn.ModuleList()
        self.stage_out_channels: list[int] = []

        prev_ch = in_channels
        for i in range(n_down_stages):
            ch = chs[i]
            blocks = [ResBlock3D(prev_ch, ch)]
            for _ in range(blocks_per_stage - 1):
                blocks.append(ResBlock3D(ch, ch))
            self.stages.append(nn.ModuleDict({
                "blocks": nn.Sequential(*blocks),
                "down": AxisOnlyDown(ch, ch, agg_axis),
            }))
            self.stage_out_channels.append(ch)
            prev_ch = ch

        ch = chs[-1]
        final_blocks = [ResBlock3D(prev_ch, ch)]
        for _ in range(blocks_per_stage - 1):
            final_blocks.append(ResBlock3D(ch, ch))
        self.stages.append(nn.ModuleDict({
            "blocks": nn.Sequential(*final_blocks),
            "down": nn.Identity(),
        }))
        self.stage_out_channels.append(ch)

        self.proj = nn.Conv3d(ch, plane_channels, kernel_size=1)

        target_3d = list(non_agg_shape)
        target_3d.insert(agg_axis, 1)
        self.target_3d_shape = tuple(target_3d)
        self.pool = nn.AdaptiveAvgPool3d(self.target_3d_shape)

    @property
    def n_stages(self) -> int:
        return len(self.stages)

    def run_stage(self, i: int, x: torch.Tensor) -> torch.Tensor:
        stage = self.stages[i]
        x = stage["blocks"](x)
        x = stage["down"](x)
        return x

    def finalize(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.pool(x)
        return x.squeeze(self.agg_axis_tensor)


class AxisDecoder(nn.Module):
    """
    Per-axis decoder split into stages, symmetric to AxisEncoder.

    n_up_stages stages of [AxisOnlyUp + ResBlock x B], plus one final stage
    of [ResBlock x B] (no Up). `finalize` applies the 1x1 projection and a
    trilinear resize to target_3d_shape if needed.
    """

    def __init__(
        self,
        plane_channels: int,
        hidden_channels: Sequence[int],
        blocks_per_stage: int,
        out_channels: int,
        agg_axis: int,
        target_3d_shape: Sequence[int],
        n_up_stages: int,
    ):
        super().__init__()
        if agg_axis not in (0, 1, 2):
            raise ValueError(f"agg_axis must be 0/1/2, got {agg_axis}")
        self.agg_axis = agg_axis
        self.agg_axis_tensor = agg_axis + 2
        self.target_3d_shape = tuple(target_3d_shape)

        chs = _expand_channels(hidden_channels, n_up_stages + 1)
        self.stages = nn.ModuleList()
        self.stage_out_channels: list[int] = []

        prev_ch = plane_channels
        for i in range(n_up_stages):
            ch = chs[i]
            up = AxisOnlyUp(prev_ch, ch, agg_axis)
            blocks = [ResBlock3D(ch, ch)]
            for _ in range(blocks_per_stage - 1):
                blocks.append(ResBlock3D(ch, ch))
            self.stages.append(nn.ModuleDict({
                "up": up,
                "blocks": nn.Sequential(*blocks),
            }))
            self.stage_out_channels.append(ch)
            prev_ch = ch

        ch = chs[-1]
        final_blocks = [ResBlock3D(prev_ch, ch)]
        for _ in range(blocks_per_stage - 1):
            final_blocks.append(ResBlock3D(ch, ch))
        self.stages.append(nn.ModuleDict({
            "up": nn.Identity(),
            "blocks": nn.Sequential(*final_blocks),
        }))
        self.stage_out_channels.append(ch)

        self.proj = nn.Conv3d(ch, out_channels, kernel_size=1)

    @property
    def n_stages(self) -> int:
        return len(self.stages)

    def run_stage(self, i: int, x: torch.Tensor) -> torch.Tensor:
        stage = self.stages[i]
        x = stage["up"](x)
        x = stage["blocks"](x)
        return x

    def finalize(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
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

        # Uniform stage count across the three planes so cross-plane mixers
        # have a well-defined per-stage channel count and the three encoders
        # / decoders run in lockstep.
        max_dim = max(D, H, W)
        n_stages_down = max(1, int(math.ceil(math.log2(max(2, max_dim)))))
        n_stages_up = n_stages_down

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

        self.dec_dh = AxisDecoder(plane_channels, decoder_channels, blocks_per_stage,
                                   plane_channels, agg_axis=2,
                                   target_3d_shape=latent_shape,
                                   n_up_stages=n_stages_up)
        self.dec_dw = AxisDecoder(plane_channels, decoder_channels, blocks_per_stage,
                                   plane_channels, agg_axis=1,
                                   target_3d_shape=latent_shape,
                                   n_up_stages=n_stages_up)
        self.dec_hw = AxisDecoder(plane_channels, decoder_channels, blocks_per_stage,
                                   plane_channels, agg_axis=0,
                                   target_3d_shape=latent_shape,
                                   n_up_stages=n_stages_up)

        self.dec_mixers = nn.ModuleList(
            CrossPlaneMixer(self.dec_dh.stage_out_channels[i])
            for i in range(self.dec_dh.n_stages)
        )

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

    def decode_planes(self, planes: dict[str, torch.Tensor]) -> torch.Tensor:
        x_dh = planes["dh"].unsqueeze(self.dec_dh.agg_axis_tensor)
        x_dw = planes["dw"].unsqueeze(self.dec_dw.agg_axis_tensor)
        x_hw = planes["hw"].unsqueeze(self.dec_hw.agg_axis_tensor)

        for i in range(self.dec_dh.n_stages):
            x_dh = self.dec_dh.run_stage(i, x_dh)
            x_dw = self.dec_dw.run_stage(i, x_dw)
            x_hw = self.dec_hw.run_stage(i, x_hw)
            x_dh, x_dw, x_hw = self.dec_mixers[i](x_dh, x_dw, x_hw)

        x_dh = self.dec_dh.finalize(x_dh)
        x_dw = self.dec_dw.finalize(x_dw)
        x_hw = self.dec_hw.finalize(x_hw)
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
