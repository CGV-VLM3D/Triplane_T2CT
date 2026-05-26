"""
TriPlaneAE — uses trivae2's triplane encoder + an implicit-function decoder
that samples per-coordinate features from the three refined plane
representations and feeds them through a coordinate-conditioned MLP.

Pipeline:
    z [B, in_channels, D, H, W]
      -> 3 AxisEncoders + CrossPlaneMixer (from trivae2)
      -> planes {"dh", "dw", "hw"} of [B, C_plane, ., .]
      -> 2D residual refinement per plane (C_plane -> C_refined)
      -> for each 3D query coord p=(d,h,w):
            f_dh = grid_sample(plane_dh, (h, d))
            f_dw = grid_sample(plane_dw, (w, d))
            f_hw = grid_sample(plane_hw, (w, h))
            f    = f_dh + f_dw + f_hw           (or concat)
            pe   = sinusoidal_pos_encoding(p)
            out  = MLP([f, pe])
      -> reshape to [B, out_channels, D, H, W]
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.layers import Act

from models.trivae2 import AxisEncoder, CrossPlaneMixer


__all__ = ["TriPlaneAE", "TriPlaneImplicitDecoder", "sinusoidal_pos_encoding"]


# ---------------------------------------------------------------------------
# 2D residual blocks for plane refinement.
# ---------------------------------------------------------------------------
class ResBlock2D(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_c), in_c)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, out_c), out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.skip = (nn.Conv2d(in_c, out_c, kernel_size=1)
                     if in_c != out_c else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


class GroupedResBlock2D(nn.Module):
    """
    3 planes stacked along channel dim, processed with groups=3 so each plane
    is refined independently while sharing the module layout.

    REQUIRES all three planes to have the SAME spatial shape (cubic latent).
    """

    def __init__(self, in_c_per_plane: int, out_c_per_plane: int, n_planes: int = 3):
        super().__init__()
        in_c = in_c_per_plane * n_planes
        out_c = out_c_per_plane * n_planes
        self.norm1 = nn.GroupNorm(n_planes, in_c)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, groups=n_planes)
        self.norm2 = nn.GroupNorm(n_planes, out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, groups=n_planes)
        self.act = nn.SiLU()
        if in_c_per_plane != out_c_per_plane:
            self.skip = nn.Conv2d(in_c, out_c, kernel_size=1, groups=n_planes)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


# ---------------------------------------------------------------------------
# Positional encoding.
# ---------------------------------------------------------------------------
def sinusoidal_pos_encoding(
    coords: torch.Tensor,
    num_freqs: int,
    include_input: bool = False,
) -> torch.Tensor:
    """
    NeRF-style sinusoidal positional encoding.

    coords: [..., D] tensor in [-1, 1].
    Returns: [..., D * 2 * num_freqs (+ D if include_input)] tensor.
    """
    if num_freqs <= 0:
        return coords if include_input else coords.new_zeros((*coords.shape[:-1], 0))

    freqs = (2.0 ** torch.arange(num_freqs, device=coords.device, dtype=coords.dtype)) * math.pi
    # scaled: [..., D, num_freqs]
    scaled = coords.unsqueeze(-1) * freqs
    # [..., D, num_freqs, 2]
    enc = torch.stack([scaled.sin(), scaled.cos()], dim=-1)
    # [..., D * 2 * num_freqs]
    enc = enc.flatten(-3)
    if include_input:
        enc = torch.cat([coords, enc], dim=-1)
    return enc


# ---------------------------------------------------------------------------
# Coordinate-conditioned MLP with optional NeRF-style skip-concat.
# ---------------------------------------------------------------------------
class SkipConcatMLP(nn.Module):
    """
    Plain MLP with optional NeRF-style skip connections that re-inject the
    full input vector at chosen layer boundaries.

    Args:
        in_dim:        input feature dim.
        hidden_dim:    hidden width.
        out_dim:       final output dim.
        depth:         total number of Linear layers (>=1). depth==1 collapses
                       to a single Linear(in_dim, out_dim).
        skip_layers:   1-indexed layer positions before which the original
                       input is concatenated to the hidden state. Indices
                       must be in [1, depth-1]; 0 (input) and `depth` (final)
                       are not valid skip points.
        activation:    'gelu' (default) or 'relu' / 'silu' for the non-final
                       Linears.

    skip_layers=[] reproduces the old simple MLP.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        depth: int,
        skip_layers: Sequence[int] = (),
        activation: str = "gelu",
    ):
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        skip_set = set(int(i) for i in skip_layers)
        for s in skip_set:
            if not (1 <= s <= depth - 1):
                raise ValueError(
                    f"skip layer index {s} out of range; valid range is "
                    f"[1, depth-1] = [1, {depth - 1}] for depth={depth}."
                )
        self.skip_set = skip_set
        self.depth = depth

        act_map = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}
        if activation not in act_map:
            raise ValueError(f"activation must be one of {list(act_map)}; got {activation}")
        self.act = act_map[activation]()

        self.layers = nn.ModuleList()
        d_in = in_dim
        for i in range(depth - 1):
            self.layers.append(nn.Linear(d_in, hidden_dim))
            d_in = hidden_dim + in_dim if (i + 1) in skip_set else hidden_dim
        # Final layer (no activation after).
        self.final = nn.Linear(d_in if depth > 1 else in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x
        h = x
        for i, layer in enumerate(self.layers):
            h = self.act(layer(h))
            if (i + 1) in self.skip_set:
                h = torch.cat([h, x_in], dim=-1)
        return self.final(h)


# ---------------------------------------------------------------------------
# Implicit decoder.
# ---------------------------------------------------------------------------
class TriPlaneImplicitDecoder(nn.Module):
    """
    Refine each triplane feature with a 2D ResBlock, then for any 3D query
    sample bilinearly from the three planes, optionally add sinusoidal
    positional encoding of the coordinate, and run an MLP to predict the
    per-voxel output.
    """

    def __init__(
        self,
        plane_in_channels: int,
        plane_refined_channels: int = 64,
        mlp_hidden: int = 128,
        mlp_depth: int = 3,
        mlp_skip_layers: Sequence[int] = (),
        mlp_activation: str = "gelu",
        out_channels: int = 4,
        fusion: str = "sum",          # "sum" | "concat"
        refine: str = "per_plane",    # "per_plane" | "grouped"
        num_pos_freqs: int = 4,
        include_input_pos: bool = False,
        output_act: str | None = None,
    ):
        super().__init__()
        if fusion not in ("sum", "concat"):
            raise ValueError(f"fusion must be 'sum' or 'concat', got {fusion}")
        if refine not in ("per_plane", "grouped"):
            raise ValueError(f"refine must be 'per_plane' or 'grouped', got {refine}")

        self.fusion = fusion
        self.refine = refine
        self.num_pos_freqs = num_pos_freqs
        self.include_input_pos = include_input_pos
        self.output_act = output_act

        if refine == "per_plane":
            self.refiner_dh = ResBlock2D(plane_in_channels, plane_refined_channels)
            self.refiner_dw = ResBlock2D(plane_in_channels, plane_refined_channels)
            self.refiner_hw = ResBlock2D(plane_in_channels, plane_refined_channels)
        else:
            self.refiner = GroupedResBlock2D(plane_in_channels, plane_refined_channels, n_planes=3)

        per_query_feat = (plane_refined_channels if fusion == "sum"
                          else 3 * plane_refined_channels)
        pos_enc_dim = 3 * 2 * num_pos_freqs + (3 if include_input_pos else 0)
        mlp_in = per_query_feat + pos_enc_dim

        self.mlp = SkipConcatMLP(
            in_dim=mlp_in,
            hidden_dim=mlp_hidden,
            out_dim=out_channels,
            depth=mlp_depth,
            skip_layers=mlp_skip_layers,
            activation=mlp_activation,
        )

    def _refine_planes(self, p_dh, p_dw, p_hw):
        if self.refine == "per_plane":
            return self.refiner_dh(p_dh), self.refiner_dw(p_dw), self.refiner_hw(p_hw)
        # Grouped path: requires equal spatial shape across planes.
        if not (p_dh.shape[-2:] == p_dw.shape[-2:] == p_hw.shape[-2:]):
            raise ValueError(
                "refine='grouped' requires all planes to share spatial shape, "
                f"got {p_dh.shape[-2:]}, {p_dw.shape[-2:]}, {p_hw.shape[-2:]}. "
                "Use refine='per_plane' for non-cubic latents."
            )
        stacked = torch.cat([p_dh, p_dw, p_hw], dim=1)
        refined = self.refiner(stacked)
        c = refined.shape[1] // 3
        return refined[:, :c], refined[:, c:2 * c], refined[:, 2 * c:]

    @staticmethod
    def _sample_plane(plane: torch.Tensor, coords_xy: torch.Tensor) -> torch.Tensor:
        """
        plane: [B, C, Hp, Wp]
        coords_xy: [B, N, 2] in [-1, 1], (x corresponds to last spatial Wp,
                                          y corresponds to second-last spatial Hp).
        returns: [B, N, C]
        """
        grid = coords_xy.unsqueeze(2)  # [B, N, 1, 2]
        sampled = F.grid_sample(
            plane, grid, mode="bilinear",
            align_corners=True, padding_mode="border",
        )
        # [B, C, N, 1] -> [B, N, C]
        return sampled.squeeze(-1).transpose(1, 2)

    def forward(
        self,
        planes: dict[str, torch.Tensor],
        coords_3d: torch.Tensor,
    ) -> torch.Tensor:
        """
        planes:
            dh: [B, C, D, H]
            dw: [B, C, D, W]
            hw: [B, C, H, W]
        coords_3d: [B, N, 3] in [-1, 1], order (d, h, w).
        returns: [B, N, out_channels]
        """
        p_dh, p_dw, p_hw = self._refine_planes(planes["dh"], planes["dw"], planes["hw"])

        d = coords_3d[..., 0]
        h = coords_3d[..., 1]
        w = coords_3d[..., 2]

        # grid_sample axes: x=last spatial, y=second-last spatial.
        # DH [.., D, H] -> x=h, y=d
        # DW [.., D, W] -> x=w, y=d
        # HW [.., H, W] -> x=w, y=h
        c_dh = torch.stack([h, d], dim=-1)
        c_dw = torch.stack([w, d], dim=-1)
        c_hw = torch.stack([w, h], dim=-1)

        f_dh = self._sample_plane(p_dh, c_dh)
        f_dw = self._sample_plane(p_dw, c_dw)
        f_hw = self._sample_plane(p_hw, c_hw)

        if self.fusion == "sum":
            f = f_dh + f_dw + f_hw
        else:
            f = torch.cat([f_dh, f_dw, f_hw], dim=-1)

        pe = sinusoidal_pos_encoding(coords_3d, self.num_pos_freqs, self.include_input_pos)
        x = torch.cat([f, pe], dim=-1)
        out = self.mlp(x)
        if self.output_act is not None:
            out = Act[self.output_act](out)
        return out


# ---------------------------------------------------------------------------
# Full autoencoder: triplane encoder + implicit decoder.
# ---------------------------------------------------------------------------
class TriPlaneAE(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int | None = None,
        latent_shape: Sequence[int] = (120, 120, 64),
        encoder_channels: Sequence[int] = (16, 32, 32, 64, 64, 64),
        blocks_per_stage: int = 2,
        plane_channels: int = 16,
        # decoder
        plane_refined_channels: int = 64,
        mlp_hidden: int = 128,
        mlp_depth: int = 3,
        mlp_skip_layers: Sequence[int] = (),
        mlp_activation: str = "gelu",
        fusion: str = "sum",
        refine: str = "per_plane",
        num_pos_freqs: int = 4,
        include_input_pos: bool = False,
        output_act: str | None = None,
        **_unused,
    ):
        super().__init__()
        D, H, W = (int(v) for v in latent_shape)
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.latent_shape = (D, H, W)

        # Stage count = len(encoder_channels). The agg axis is downsampled
        # `len-1` times; the remainder is collapsed by AdaptivePool in
        # AxisEncoder.finalize.
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

        self.decoder = TriPlaneImplicitDecoder(
            plane_in_channels=plane_channels,
            plane_refined_channels=plane_refined_channels,
            mlp_hidden=mlp_hidden,
            mlp_depth=mlp_depth,
            mlp_skip_layers=mlp_skip_layers,
            mlp_activation=mlp_activation,
            out_channels=self.out_channels,
            fusion=fusion,
            refine=refine,
            num_pos_freqs=num_pos_freqs,
            include_input_pos=include_input_pos,
            output_act=output_act,
        )

        # Pre-computed full-volume query grid (does NOT depend on B).
        self.register_buffer(
            "_full_grid_coords",
            self._make_full_grid(D, H, W),
            persistent=False,
        )

    @staticmethod
    def _make_full_grid(D: int, H: int, W: int) -> torch.Tensor:
        d = torch.linspace(-1, 1, D)
        h = torch.linspace(-1, 1, H)
        w = torch.linspace(-1, 1, W)
        gd, gh, gw = torch.meshgrid(d, h, w, indexing="ij")
        # [D, H, W, 3] -> [D*H*W, 3]
        return torch.stack([gd, gh, gw], dim=-1).reshape(-1, 3)

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

    def decode(
        self,
        planes: dict[str, torch.Tensor],
        coords: torch.Tensor | None = None,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        """
        coords=None  → reconstruct the full latent volume: returns [B, C, D, H, W].
        coords given → returns [B, N, out_channels].

        chunk_size: optional, splits the query batch to limit peak memory for
                    large grids during inference. Must be None during training
                    so gradients flow through one fused call.
        """
        B = planes["dh"].shape[0]
        if coords is None:
            coords = self._full_grid_coords.unsqueeze(0).expand(B, -1, -1)
            full_grid = True
        else:
            full_grid = False

        if chunk_size is None or coords.shape[1] <= chunk_size:
            out = self.decoder(planes, coords)
        else:
            outs = []
            for s in range(0, coords.shape[1], chunk_size):
                outs.append(self.decoder(planes, coords[:, s:s + chunk_size]))
            out = torch.cat(outs, dim=1)

        if full_grid:
            D, H, W = self.latent_shape
            # [B, D*H*W, C] -> [B, C, D, H, W]
            out = out.reshape(B, D, H, W, self.out_channels).permute(0, 4, 1, 2, 3).contiguous()
        return out

    def forward(
        self,
        z: torch.Tensor,
        return_dict: bool = False,
        coords: torch.Tensor | None = None,
        chunk_size: int | None = None,
    ):
        planes = self.encode(z)
        out = self.decode(planes, coords=coords, chunk_size=chunk_size)
        zero_loss = z.new_zeros(())
        if return_dict:
            return {
                "reconstruction": out,
                "q_loss": zero_loss,
                "planes": planes,
            }
        return out, zero_loss
