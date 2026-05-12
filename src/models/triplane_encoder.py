from __future__ import annotations

import torch
import torch.nn as nn


class TriplaneEncoder(nn.Module):
    """
    3D MAISI latent -> three 2D triplane feature maps.

    Args:
        in_channels:  4   (MAISI mu channels)
        emb_dim:      512 (transformer token dim)
        n_layers:     4   (shared-weight self-attention iterations)
        n_heads:      8
        out_channels: 8   (triplane channel width)
        latent_shape: (120, 120, 64)  (H, W, D)

    Depth-collapse strategy: mean-pool over the aggregate axis before the
    transformer, then project to out_channels.  Alternative (learned linear
    projection over D) is slightly more expressive but adds parameters
    proportional to D; mean-pool is parameter-free and worked well in trivae.py.
    """

    def __init__(
        self,
        in_channels: int = 4,
        emb_dim: int = 512,
        n_layers: int = 4,
        n_heads: int = 8,
        out_channels: int = 8,
        latent_shape: tuple[int, int, int] = (120, 120, 64),
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.out_channels = out_channels
        H, W, D = latent_shape
        self.H, self.W, self.D = H, W, D

        # One shared self-attention layer called n_layers times (weight sharing
        # as specified; same module object re-used each forward pass).
        self.attn = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=emb_dim * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,  # pre-norm: more stable training
        )

        # Per-plane input projections: collapse depth axis via mean-pool,
        # then flatten spatial dims and project each position to emb_dim.
        # XY plane: mean over D -> [B, C, H, W] -> tokens [B, H*W, emb_dim]
        self.proj_in_xy = nn.Linear(in_channels, emb_dim)
        # YZ plane: mean over H -> [B, C, W, D] -> wait, spec says YZ: [B,C,120,64]
        # Axes: input [B, C, H, W, D]. XY collapses D; YZ collapses H (axis=2 in BCWHY);
        # XZ collapses W.
        # YZ plane tokens: [B, W*D, emb_dim] but output is [B, C, W, D]... hmm.
        # Spec: z_yz [B, 8, 120, 64] => spatial (W=120, D=64), collapsed H.
        # XZ: z_xz [B, 8, 120, 64] => spatial (H=120, D=64), collapsed W.
        # XY: z_xy [B, 8, 120, 120] => spatial (H=120, W=120), collapsed D.
        self.proj_in_yz = nn.Linear(in_channels, emb_dim)
        self.proj_in_xz = nn.Linear(in_channels, emb_dim)

        # Output projections: emb_dim -> out_channels per token
        self.proj_out_xy = nn.Linear(emb_dim, out_channels)
        self.proj_out_yz = nn.Linear(emb_dim, out_channels)
        self.proj_out_xz = nn.Linear(emb_dim, out_channels)

    def _run_attn(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, N, emb_dim]
        # Same layer called n_layers times (shared weights).
        for _ in range(self.n_layers):
            tokens = self.attn(tokens)
        return tokens

    def forward(self, mu: torch.Tensor) -> dict[str, torch.Tensor]:
        # mu: [B, C, H, W, D]
        B, C, H, W, D = mu.shape

        # --- XY plane: collapse D (mean over dim=4) -> [B, C, H, W] ---
        xy = mu.mean(dim=4)  # [B, C, H, W]
        xy = xy.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, H*W, C]
        xy = self.proj_in_xy(xy)  # [B, H*W, emb_dim]
        xy = self._run_attn(xy)
        xy = self.proj_out_xy(xy)  # [B, H*W, out_channels]
        xy = xy.reshape(B, H, W, self.out_channels).permute(
            0, 3, 1, 2
        )  # [B, out_c, H, W]

        # --- YZ plane: collapse H (mean over dim=2) -> [B, C, W, D] ---
        yz = mu.mean(dim=2)  # [B, C, W, D]
        yz = yz.permute(0, 2, 3, 1).reshape(B, W * D, C)  # [B, W*D, C]
        yz = self.proj_in_yz(yz)
        yz = self._run_attn(yz)
        yz = self.proj_out_yz(yz)  # [B, W*D, out_channels]
        yz = yz.reshape(B, W, D, self.out_channels).permute(
            0, 3, 1, 2
        )  # [B, out_c, W, D]

        # --- XZ plane: collapse W (mean over dim=3) -> [B, C, H, D] ---
        xz = mu.mean(dim=3)  # [B, C, H, D]
        xz = xz.permute(0, 2, 3, 1).reshape(B, H * D, C)  # [B, H*D, C]
        xz = self.proj_in_xz(xz)
        xz = self._run_attn(xz)
        xz = self.proj_out_xz(xz)  # [B, H*D, out_channels]
        xz = xz.reshape(B, H, D, self.out_channels).permute(
            0, 3, 1, 2
        )  # [B, out_c, H, D]

        return {"z_xy": xy, "z_yz": yz, "z_xz": xz}
