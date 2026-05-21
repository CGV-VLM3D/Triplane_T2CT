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


class TriplaneEncoder(nn.Module):
    """
    3D MAISI latent -> three 2D triplane feature maps via patchify + F_psi.

    Pipeline:
        [B, 4, H, W, D]
          -> Patchify: Conv3d(4, C, patch_size, stride=patch_size)
        [B, C, H', W', D']   where H'=H//p, W'=W//p, D'=D//p
          -> (optional) n_res_blocks ResBlock3D in the patch grid (0 by default)
          -> F_psi per plane: self-attention over the collapsed axis with a
             prepended z_init query token reads index 0 as the aggregated rep.
        {XY: [B, out_c, H', W'], YZ: [B, out_c, W', D'], XZ: [B, out_c, H', D']}

    F_psi token shapes after patchify (B=1, p=4):
        XY collapses D': [H'*W', D'+1, emb_dim] = [900, 17, 256]
        YZ collapses H': [W'*D', H'+1, emb_dim] = [480, 31, 256]
        XZ collapses W': [H'*D', W'+1, emb_dim] = [480, 31, 256]

    Args:
        in_channels:  4   (MAISI latent channels)
        emb_dim:      256 (transformer + patch embed channels)
        n_layers:     4
        n_heads:      8
        out_channels: 8   (triplane channel width, fed to decoder)
        latent_shape: (H, W, D) — must each be divisible by patch_size
        patch_size:   4
        n_res_blocks: 0   (ResBlock3D layers in patch grid; 0 = skip entirely)
    """

    def __init__(
        self,
        in_channels: int = 4,
        emb_dim: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        out_channels: int = 8,
        latent_shape: tuple[int, int, int] = (120, 120, 64),
        patch_size: int = 4,
        n_res_blocks: int = 0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.out_channels = out_channels
        self.patch_size = patch_size

        H, W, D = latent_shape
        self.H, self.W, self.D = H, W, D
        p = patch_size
        self.Hp, self.Wp, self.Dp = H // p, W // p, D // p

        # Patchify: non-overlapping 3D conv, exactly equivalent to rearranging
        # each p×p×p cube into a C-dim token.
        self.patchify = nn.Conv3d(in_channels, emb_dim, kernel_size=p, stride=p)

        # Optional ResBlock3D stack in the patch grid (depth controlled by caller).
        if n_res_blocks > 0:
            self.res_blocks: nn.Module = nn.Sequential(
                *[ResBlock3D(emb_dim, emb_dim) for _ in range(n_res_blocks)]
            )
        else:
            self.res_blocks = nn.Identity()

        # Shared transformer for all three planes.
        attn_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=emb_dim * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.attn = nn.TransformerEncoder(attn_layer, num_layers=n_layers)

        # Per-plane input projections: C -> emb_dim.
        # Identity when C == emb_dim (the default), lightweight Linear otherwise.
        def _proj_in() -> nn.Module:
            return nn.Linear(emb_dim, emb_dim)  # identity-like, but trainable

        self.proj_in_xy = _proj_in()
        self.proj_in_yz = _proj_in()
        self.proj_in_xz = _proj_in()

        # Output projections: emb_dim -> out_channels
        self.proj_out_xy = nn.Linear(emb_dim, out_channels)
        self.proj_out_yz = nn.Linear(emb_dim, out_channels)
        self.proj_out_xz = nn.Linear(emb_dim, out_channels)

        # Learnable z_init query tokens (D3T pattern).
        self.z_init_xy = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.z_init_yz = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.z_init_xz = nn.Parameter(torch.zeros(1, 1, emb_dim))

        # Learnable positional embeddings: z_init slot + agg_len patch positions.
        # XY collapses Dp:  length = Dp+1
        # YZ collapses Hp:  length = Hp+1
        # XZ collapses Wp:  length = Wp+1
        self.pos_embed_xy = nn.Parameter(torch.zeros(1, self.Dp + 1, emb_dim))
        self.pos_embed_yz = nn.Parameter(torch.zeros(1, self.Hp + 1, emb_dim))
        self.pos_embed_xz = nn.Parameter(torch.zeros(1, self.Wp + 1, emb_dim))
        nn.init.trunc_normal_(self.pos_embed_xy, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_yz, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_xz, std=0.02)

    def _f_psi(
        self,
        seq: torch.Tensor,
        z_init: nn.Parameter,
        pos_embed: nn.Parameter,
        proj_in: nn.Linear,
        proj_out: nn.Linear,
    ) -> torch.Tensor:
        """F_psi aggregation — no chunking needed at patchified scale.

        Args:
            seq:  [N, agg_len, emb_dim]
            Returns [N, out_channels]
        """
        N = seq.shape[0]
        tokens = proj_in(seq)  # [N, agg_len, emb_dim]
        tokens = torch.cat(
            [z_init.expand(N, -1, -1), tokens], dim=1
        )  # [N, agg_len+1, emb_dim]
        tokens = tokens + pos_embed  # broadcast over N
        out = self.attn(tokens)
        return proj_out(out[:, 0])  # [N, out_channels]

    def forward(self, mu: torch.Tensor) -> dict[str, torch.Tensor]:
        # mu: [B, C, H, W, D]
        B = mu.shape[0]
        Hp, Wp, Dp = self.Hp, self.Wp, self.Dp

        # Patchify -> [B, emb_dim, Hp, Wp, Dp]
        x = self.patchify(mu)
        x = self.res_blocks(x)

        # --- XY plane: aggregate over Dp, output [B, out_c, Hp, Wp] ---
        # permute: [B, Hp, Wp, Dp, emb_dim], reshape: [B*Hp*Wp, Dp, emb_dim]
        xy_seq = x.permute(0, 2, 3, 4, 1).reshape(B * Hp * Wp, Dp, self.emb_dim)
        xy = self._f_psi(
            xy_seq, self.z_init_xy, self.pos_embed_xy, self.proj_in_xy, self.proj_out_xy
        )
        xy = xy.reshape(B, Hp, Wp, self.out_channels).permute(
            0, 3, 1, 2
        )  # [B, out_c, Hp, Wp]

        # --- YZ plane: aggregate over Hp, output [B, out_c, Wp, Dp] ---
        # permute: [B, Wp, Dp, Hp, emb_dim], reshape: [B*Wp*Dp, Hp, emb_dim]
        yz_seq = x.permute(0, 3, 4, 2, 1).reshape(B * Wp * Dp, Hp, self.emb_dim)
        yz = self._f_psi(
            yz_seq, self.z_init_yz, self.pos_embed_yz, self.proj_in_yz, self.proj_out_yz
        )
        yz = yz.reshape(B, Wp, Dp, self.out_channels).permute(
            0, 3, 1, 2
        )  # [B, out_c, Wp, Dp]

        # --- XZ plane: aggregate over Wp, output [B, out_c, Hp, Dp] ---
        # permute: [B, Hp, Dp, Wp, emb_dim], reshape: [B*Hp*Dp, Wp, emb_dim]
        xz_seq = x.permute(0, 2, 4, 3, 1).reshape(B * Hp * Dp, Wp, self.emb_dim)
        xz = self._f_psi(
            xz_seq, self.z_init_xz, self.pos_embed_xz, self.proj_in_xz, self.proj_out_xz
        )
        xz = xz.reshape(B, Hp, Dp, self.out_channels).permute(
            0, 3, 1, 2
        )  # [B, out_c, Hp, Dp]

        return {"z_xy": xy, "z_yz": yz, "z_xz": xz}
