"""TriplaneEncoderCompress3D: paper-faithful Compress3D encoder for MAISI latents.

Differences from the simplified ``TriplaneEncoder`` (one-shot axis-collapse +
column-only cross-attn on raw volume):

  1. **Feature lift** (Eq./Fig 3): 1x1x1 Conv3d expands the 4-channel MAISI
     latent to ``feature_lift_channels`` (default 32) before any projection.
  2. **High-res triplane via axis-collapse** (Eq 3-5): single Conv3d with
     full-axis kernel still produces the 2D triplane, but now operating on
     the lifted feature volume.
  3. **Per-plane 2D ResBlock + 2x downsample stages** (Fig 3 "x2 → DownSample
     1/2 → x2 → DownSample 1/2 → x4"): channel progression
     ``plane_channels_progression`` defaults to ``[32, 64, 128]`` (paper).
  4. **3D-aware cross-attention** (Eq 6-11):
       V^n_d = 3DConv(V^n, k=o, s=o)         # learned volume downsample
       Q = TriConv(low-res-triplane, k=1)    # query from compressed plane
       K, V = 3DConv(V^n_d, k=1)             # key/value from deep 3D feature
       Each query (i,j) attends to a cube of size m_a x m_b x r''_c in V^n_d,
       where the m's are computed per-axis from r''/r'. Default ``o=2`` gives
       m=2 in our 60^3 / 120^3 settings — true 3D context, not column-only.
  5. **mu / logvar heads** (TriConv k=1): output the low-res VAE latent.

Plane convention (same as existing ``TriplaneEncoder`` / ``tri_conv.py``):
    "xy"  [B, C, H, W]   collapsed axis = D (last)
    "yz"  [B, C, H, D]   collapsed axis = W (middle)
    "xz"  [B, C, W, D]   collapsed axis = H (first)

with ``latent_shape = (H, W, D)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

from .tri_conv import TriConv


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


class _PlaneStage(nn.Module):
    """`n_blocks` ResBlock2D + optional 2x AvgPool downsample.

    First block changes in_c → out_c; subsequent blocks are out_c → out_c.
    """

    def __init__(self, in_c: int, out_c: int, n_blocks: int, downsample: bool) -> None:
        super().__init__()
        blocks: list[nn.Module] = [_ResBlock2D(in_c, out_c)]
        for _ in range(n_blocks - 1):
            blocks.append(_ResBlock2D(out_c, out_c))
        self.blocks = nn.Sequential(*blocks)
        self.down: nn.Module = (
            nn.AvgPool2d(kernel_size=2) if downsample else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.blocks(x))


class _ThreeStageEncoder(nn.Module):
    """Per-plane stack: stage1(in→c1,n1) → stage2(c1→c2,n2) → stage3(c2→c3,n3,no down).

    ``n_downsamples`` ∈ {0, 1, 2} controls how many 2× spatial downsamples are
    inserted (after stage 1, then stage 2). The last stage never downsamples.
    """

    def __init__(
        self,
        in_c: int,
        channels: tuple[int, int, int],
        n_blocks: tuple[int, int, int],
        n_downsamples: int,
    ) -> None:
        super().__init__()
        c1, c2, c3 = channels
        n1, n2, n3 = n_blocks
        self.stage1 = _PlaneStage(in_c, c1, n1, downsample=(n_downsamples >= 1))
        self.stage2 = _PlaneStage(c1, c2, n2, downsample=(n_downsamples >= 2))
        self.stage3 = _PlaneStage(c2, c3, n3, downsample=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x


class _CubeCrossAttention3D(nn.Module):
    """3D-aware cross-attention with cube-region keys/values (paper Eq 6-11).

    Differences from the existing ``CrossAttention3D``:
    - Q comes from a **low-res** triplane (post downsample), K/V from a
      **downsampled** volume V^n_d, not raw input z.
    - Each triplane cell attends to a cube of m_a x m_b x r''_c tokens
      (m_a, m_b can be > 1 when ``volume_downsample < triplane_downsample``).
    - 3D additive positional embedding lives on V^n_d (downsampled grid).
    """

    def __init__(
        self,
        plane_channels: int,
        volume_kv_channels: int,
        d_kv: int,
        heads: int,
        triplane_shape_xy: tuple[int, int],
        triplane_shape_yz: tuple[int, int],
        triplane_shape_xz: tuple[int, int],
        volume_shape: tuple[int, int, int],
    ) -> None:
        super().__init__()
        self.heads = heads
        self.d_kv = d_kv
        inner = heads * d_kv
        self.inner = inner
        self._scale = d_kv**-0.5

        H_v, W_v, D_v = volume_shape
        self.H_v, self.W_v, self.D_v = H_v, W_v, D_v

        # Compute per-axis cube sizes m for each plane.
        # XY plane queries (H, W) of V^n_d, attends along full D_v.
        rH_xy, rW_xy = triplane_shape_xy
        self.m_xy_H = H_v // rH_xy
        self.m_xy_W = W_v // rW_xy
        assert self.m_xy_H * rH_xy == H_v, (
            f"H_v {H_v} not divisible by triplane H {rH_xy}"
        )
        assert self.m_xy_W * rW_xy == W_v, (
            f"W_v {W_v} not divisible by triplane W {rW_xy}"
        )
        self.rH_xy, self.rW_xy = rH_xy, rW_xy

        # YZ plane queries (H, D), attends along full W_v.
        rH_yz, rD_yz = triplane_shape_yz
        self.m_yz_H = H_v // rH_yz
        self.m_yz_D = D_v // rD_yz
        assert self.m_yz_H * rH_yz == H_v
        assert self.m_yz_D * rD_yz == D_v
        self.rH_yz, self.rD_yz = rH_yz, rD_yz

        # XZ plane queries (W, D), attends along full H_v.
        rW_xz, rD_xz = triplane_shape_xz
        self.m_xz_W = W_v // rW_xz
        self.m_xz_D = D_v // rD_xz
        assert self.m_xz_W * rW_xz == W_v
        assert self.m_xz_D * rD_xz == D_v
        self.rW_xz, self.rD_xz = rW_xz, rD_xz

        # Q from low-res triplane (cross-plane-aware via TriConv k=1).
        self.q_proj = TriConv(plane_channels, inner, kernel_size=1)
        # K/V from V^n_d.
        self.kv_proj = nn.Conv3d(volume_kv_channels, 2 * inner, kernel_size=1)
        # Output projection back to plane_channels.
        self.out_proj = TriConv(inner, plane_channels, kernel_size=1)

        # Separable additive 3D positional embedding on V^n_d (downsampled grid).
        self.pe_x = nn.Parameter(torch.zeros(1, 2 * inner, H_v, 1, 1))
        self.pe_y = nn.Parameter(torch.zeros(1, 2 * inner, 1, W_v, 1))
        self.pe_z = nn.Parameter(torch.zeros(1, 2 * inner, 1, 1, D_v))
        trunc_normal_(self.pe_x, std=0.02)
        trunc_normal_(self.pe_y, std=0.02)
        trunc_normal_(self.pe_z, std=0.02)

    def _attn_1d(
        self,
        q: torch.Tensor,  # [B*cells, 1, inner]
        kv: torch.Tensor,  # [B*cells, agg_len, 2*inner]
    ) -> torch.Tensor:
        B_cells = q.shape[0]
        h, d, inner = self.heads, self.d_kv, self.inner
        q_h = q.view(B_cells, 1, h, d)
        k, v = kv.split(inner, dim=-1)
        k_h = k.view(B_cells, -1, h, d)
        v_h = v.view(B_cells, -1, h, d)
        attn = torch.einsum("bqhd,bkhd->bhqk", q_h, k_h) * self._scale
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum("bhqk,bkhd->bqhd", attn, v_h)
        return out.reshape(B_cells, inner)

    def _attend_xy(self, q_xy: torch.Tensor, kv_vol: torch.Tensor) -> torch.Tensor:
        """Q_xy [B, inner, rH, rW]; KV from V^n_d [B, 2*inner, H_v, W_v, D_v].
        Cube per query (i,j): kv_vol[:, :, i*mH:(i+1)*mH, j*mW:(j+1)*mW, :]
        Cube shape: (mH, mW, D_v). Token count: mH*mW*D_v."""
        B = q_xy.shape[0]
        rH, rW = self.rH_xy, self.rW_xy
        mH, mW = self.m_xy_H, self.m_xy_W
        D_v = self.D_v
        inner = self.inner

        # KV reshape: [B, 2*inner, rH, mH, rW, mW, D_v]
        #   -> permute to [B, rH, rW, mH, mW, D_v, 2*inner]
        #   -> reshape to [B*rH*rW, mH*mW*D_v, 2*inner]
        kv = kv_vol.view(B, 2 * inner, rH, mH, rW, mW, D_v)
        kv = kv.permute(0, 2, 4, 3, 5, 6, 1).contiguous()
        kv = kv.view(B * rH * rW, mH * mW * D_v, 2 * inner)

        # Q reshape: [B, inner, rH, rW] -> [B*rH*rW, 1, inner]
        q = q_xy.permute(0, 2, 3, 1).reshape(B * rH * rW, 1, inner)

        out = self._attn_1d(q, kv)  # [B*rH*rW, inner]
        return out.view(B, rH, rW, inner).permute(0, 3, 1, 2).contiguous()

    def _attend_yz(self, q_yz: torch.Tensor, kv_vol: torch.Tensor) -> torch.Tensor:
        """Q_yz [B, inner, rH, rD]; KV from V^n_d [B, 2*inner, H_v, W_v, D_v].
        Cube per query (i,j): kv_vol[:, :, i*mH:(i+1)*mH, :, j*mD:(j+1)*mD]
        Cube shape: (mH, W_v, mD). Token count: mH*W_v*mD."""
        B = q_yz.shape[0]
        rH, rD = self.rH_yz, self.rD_yz
        mH, mD = self.m_yz_H, self.m_yz_D
        W_v = self.W_v
        inner = self.inner

        # KV: [B, 2*inner, rH, mH, W_v, rD, mD]
        #   -> permute to [B, rH, rD, mH, W_v, mD, 2*inner]
        #   -> reshape to [B*rH*rD, mH*W_v*mD, 2*inner]
        kv = kv_vol.view(B, 2 * inner, rH, mH, W_v, rD, mD)
        kv = kv.permute(0, 2, 5, 3, 4, 6, 1).contiguous()
        kv = kv.view(B * rH * rD, mH * W_v * mD, 2 * inner)

        q = q_yz.permute(0, 2, 3, 1).reshape(B * rH * rD, 1, inner)
        out = self._attn_1d(q, kv)
        return out.view(B, rH, rD, inner).permute(0, 3, 1, 2).contiguous()

    def _attend_xz(self, q_xz: torch.Tensor, kv_vol: torch.Tensor) -> torch.Tensor:
        """Q_xz [B, inner, rW, rD]; KV from V^n_d [B, 2*inner, H_v, W_v, D_v].
        Cube per query (i,j): kv_vol[:, :, :, i*mW:(i+1)*mW, j*mD:(j+1)*mD]
        Cube shape: (H_v, mW, mD). Token count: H_v*mW*mD."""
        B = q_xz.shape[0]
        rW, rD = self.rW_xz, self.rD_xz
        mW, mD = self.m_xz_W, self.m_xz_D
        H_v = self.H_v
        inner = self.inner

        # KV: [B, 2*inner, H_v, rW, mW, rD, mD]
        #   -> permute to [B, rW, rD, H_v, mW, mD, 2*inner]
        #   -> reshape to [B*rW*rD, H_v*mW*mD, 2*inner]
        kv = kv_vol.view(B, 2 * inner, H_v, rW, mW, rD, mD)
        kv = kv.permute(0, 3, 5, 2, 4, 6, 1).contiguous()
        kv = kv.view(B * rW * rD, H_v * mW * mD, 2 * inner)

        q = q_xz.permute(0, 2, 3, 1).reshape(B * rW * rD, 1, inner)
        out = self._attn_1d(q, kv)
        return out.view(B, rW, rD, inner).permute(0, 3, 1, 2).contiguous()

    def forward(
        self,
        planes_lr: dict[str, torch.Tensor],
        volume_d: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # Add positional embed to V^n_d (broadcast across H/W/D).
        kv_feat = self.kv_proj(volume_d) + self.pe_x + self.pe_y + self.pe_z
        q_planes = self.q_proj(planes_lr)
        a_xy = self._attend_xy(q_planes["xy"], kv_feat)
        a_yz = self._attend_yz(q_planes["yz"], kv_feat)
        a_xz = self._attend_xz(q_planes["xz"], kv_feat)
        return self.out_proj({"xy": a_xy, "yz": a_yz, "xz": a_xz})


class TriplaneEncoderCompress3D(nn.Module):
    """Paper-faithful Compress3D encoder for MAISI 3D latents.

    Pipeline:
        z [B, in_channels, H, W, D]
          → Conv3d(in→feature_lift_channels, k=1)             # feature lift
        V^n [B, c_lift, H, W, D]
          → axis-collapse Conv3d (one per plane, full-axis kernel)
        T^(0) {xy: [B, c1, H, W], yz: [B, c1, H, D], xz: [B, c1, W, D]}
          → per-plane: ResBlock×n1 + Down (c1→c2)
          → per-plane: ResBlock×n2 + Down (c2→c3)
          → per-plane: ResBlock×n3 (no down)
        T^l (low-res triplane)
          → cube cross-attention against V^n_d
        T^e = T^l + A
          → TriConv(c3 → latent_channels) heads for mu / logvar

    Args:
        in_channels:               input channels (4 for MAISI latent).
        latent_shape:              (H, W, D) of input volume.
        feature_lift_channels:     output of the 1x1x1 input Conv3d.
        plane_channels_progression: (c1, c2, c3) channel widths across the
                                   three encoder stages.
        n_blocks_per_stage:        (n1, n2, n3) ResBlock counts.
        n_triplane_downsamples:    fixed at 2 (paper). Kept as a sanity arg.
        volume_downsample:         o in paper Eq 6. Default 2.
        volume_kv_channels:        c'' in paper. Defaults to ``feature_lift_channels``.
        cross_attn_heads:          MHA heads.
        cross_attn_d_kv:           per-head key/value dim.
        latent_channels:           output channels for mu/logvar.
    """

    def __init__(
        self,
        in_channels: int = 4,
        latent_shape: tuple[int, int, int] = (60, 60, 32),
        feature_lift_channels: int = 32,
        plane_channels_progression: tuple[int, int, int] = (32, 64, 128),
        n_blocks_per_stage: tuple[int, int, int] = (2, 2, 4),
        n_triplane_downsamples: int = 2,
        volume_downsample: int = 2,
        volume_kv_channels: int | None = None,
        cross_attn_heads: int = 4,
        cross_attn_d_kv: int = 32,
        latent_channels: int = 32,
    ) -> None:
        super().__init__()
        if n_triplane_downsamples not in (0, 1, 2):
            raise ValueError(
                f"n_triplane_downsamples must be 0, 1, or 2; got {n_triplane_downsamples}"
            )
        H, W, D = (int(v) for v in latent_shape)
        self.H, self.W, self.D = H, W, D
        self.latent_shape = (H, W, D)
        c_lift = int(feature_lift_channels)
        self.c_lift = c_lift
        c1, c2, c3 = plane_channels_progression
        self.c3 = int(c3)
        self.n_triplane_downsamples = int(n_triplane_downsamples)

        down = 2**n_triplane_downsamples
        if H % down != 0 or W % down != 0 or D % down != 0:
            raise ValueError(
                f"latent_shape {latent_shape} not divisible by 2^{n_triplane_downsamples}={down}"
            )
        rH, rW, rD = H // down, W // down, D // down
        self.rH, self.rW, self.rD = rH, rW, rD

        o = int(volume_downsample)
        # Cube m ≥ 1 requires r'' >= r', i.e., o <= 2^n_triplane_downsamples.
        if o > down:
            raise ValueError(
                f"volume_downsample={o} > 2^n_triplane_downsamples={down}; "
                "this would make cube m < 1. Reduce volume_downsample or "
                "increase n_triplane_downsamples."
            )
        if H % o != 0 or W % o != 0 or D % o != 0:
            raise ValueError(
                f"latent_shape {latent_shape} not divisible by volume_downsample {o}"
            )
        Hv, Wv, Dv = H // o, W // o, D // o
        self.Hv, self.Wv, self.Dv = Hv, Wv, Dv

        c_kv = int(volume_kv_channels) if volume_kv_channels is not None else c_lift

        # --- Feature lift: raw latent → c_lift channels ---
        self.feature_lift = nn.Conv3d(in_channels, c_lift, kernel_size=1)

        # --- Axis-collapse projections (Eq 3-5), input is c_lift not in_channels ---
        self.proj_xy = nn.Conv3d(c_lift, c1, kernel_size=(1, 1, D), stride=(1, 1, D))
        self.proj_yz = nn.Conv3d(c_lift, c1, kernel_size=(1, W, 1), stride=(1, W, 1))
        self.proj_xz = nn.Conv3d(c_lift, c1, kernel_size=(H, 1, 1), stride=(H, 1, 1))

        # --- Per-plane independent 3-stage 2D ResBlock (+ optional Down) stacks ---
        self.enc_xy = _ThreeStageEncoder(
            c1, plane_channels_progression, n_blocks_per_stage, n_triplane_downsamples
        )
        self.enc_yz = _ThreeStageEncoder(
            c1, plane_channels_progression, n_blocks_per_stage, n_triplane_downsamples
        )
        self.enc_xz = _ThreeStageEncoder(
            c1, plane_channels_progression, n_blocks_per_stage, n_triplane_downsamples
        )

        # --- Volume downsample branch: V^n → V^n_d (o=1 → just a 1x1 channel proj) ---
        self.volume_down = nn.Conv3d(c_lift, c_kv, kernel_size=o, stride=o)

        # --- 3D-aware cube cross-attention ---
        self.cross_attn = _CubeCrossAttention3D(
            plane_channels=c3,
            volume_kv_channels=c_kv,
            d_kv=cross_attn_d_kv,
            heads=cross_attn_heads,
            triplane_shape_xy=(rH, rW),
            triplane_shape_yz=(rH, rD),
            triplane_shape_xz=(rW, rD),
            volume_shape=(Hv, Wv, Dv),
        )

        # --- mu / logvar heads (TriConv k=1, cross-plane aware) ---
        self.mu_head = TriConv(c3, latent_channels, kernel_size=1)
        self.logvar_head = TriConv(c3, latent_channels, kernel_size=1)

    def forward(self, z: torch.Tensor) -> dict:
        """Args:
            z: [B, in_channels, H, W, D]
        Returns:
            {"mu":     {"xy": [B, lc, rH, rW], "yz": [B, lc, rH, rD], "xz": [B, lc, rW, rD]},
             "logvar": same structure}
        """
        # 1. Feature lift
        v = self.feature_lift(z)  # [B, c_lift, H, W, D]

        # 2. Axis-collapse to high-res triplane
        xy = self.proj_xy(v).squeeze(-1)  # [B, c1, H, W]
        yz = self.proj_yz(v).squeeze(3)  # [B, c1, H, D]
        xz = self.proj_xz(v).squeeze(2)  # [B, c1, W, D]

        # 3. Per-plane 2D ResBlock + Down (×2) stages
        xy = self.enc_xy(xy)  # [B, c3, rH, rW]
        yz = self.enc_yz(yz)  # [B, c3, rH, rD]
        xz = self.enc_xz(xz)  # [B, c3, rW, rD]

        # 4. Volume downsample branch
        v_d = self.volume_down(v)  # [B, c_kv, Hv, Wv, Dv]

        # 5. 3D-aware cube cross-attention, residual add
        planes = {"xy": xy, "yz": yz, "xz": xz}
        residual = self.cross_attn(planes, v_d)
        planes = {k: planes[k] + residual[k] for k in planes}

        # 6. mu / logvar heads
        mu = self.mu_head(planes)
        logvar = self.logvar_head(planes)
        return {"mu": mu, "logvar": logvar}
