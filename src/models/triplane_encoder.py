from __future__ import annotations

import torch
import torch.nn as nn


class TriplaneEncoder(nn.Module):
    """
    3D MAISI latent -> three 2D triplane feature maps.

    Args:
        in_channels:  4   (MAISI mu channels)
        emb_dim:      512 (transformer token dim)
        n_layers:     4   (independent self-attention layers stacked)
        n_heads:      8
        out_channels: 8   (triplane channel width)
        latent_shape: (120, 120, 64)  (H, W, D)

    Depth-collapse strategy: D3T F_psi pattern.  For each spatial position
    on the output plane we form a sequence of tokens along the collapsed axis,
    prepend a learnable z_init query token, add a learnable axial positional
    embedding, run a stack of self-attention layers, and read the z_init
    position (index 0) as the aggregated representation.

    The positional embedding is plane-specific because the collapsed axis
    length differs across planes (D=64 for XY, H=120 for YZ, W=120 for XZ).
    Without it, the transformer would be permutation-equivariant along the
    collapsed axis -- spatial ordering of CT slices would be lost.
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

        # Per-plane input projections: map each position along the collapsed
        # axis from in_channels -> emb_dim before feeding to attention.
        # XY plane collapses D; tokens are shaped [B*H*W, D, emb_dim].
        self.proj_in_xy = nn.Linear(in_channels, emb_dim)
        # YZ plane collapses H; tokens are shaped [B*W*D, H, emb_dim].
        self.proj_in_yz = nn.Linear(in_channels, emb_dim)
        # XZ plane collapses W; tokens are shaped [B*H*D, W, emb_dim].
        self.proj_in_xz = nn.Linear(in_channels, emb_dim)

        # Output projections: emb_dim -> out_channels applied to z_init output
        self.proj_out_xy = nn.Linear(emb_dim, out_channels)
        self.proj_out_yz = nn.Linear(emb_dim, out_channels)
        self.proj_out_xz = nn.Linear(emb_dim, out_channels)

        # D3T F_psi: one learnable query token per plane.  Prepended at
        # position 0; after attention, position 0 carries the aggregate.
        self.z_init_xy = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.z_init_yz = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.z_init_xz = nn.Parameter(torch.zeros(1, 1, emb_dim))

        # Learnable 1D positional embeddings per plane covering z_init (idx 0)
        # plus the agg_len positions along the collapsed axis. Initialized
        # with trunc_normal_(std=0.02) per timm ViT convention.
        self.pos_embed_xy = nn.Parameter(torch.zeros(1, D + 1, emb_dim))
        self.pos_embed_yz = nn.Parameter(torch.zeros(1, H + 1, emb_dim))
        self.pos_embed_xz = nn.Parameter(torch.zeros(1, W + 1, emb_dim))
        nn.init.trunc_normal_(self.pos_embed_xy, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_yz, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_xz, std=0.02)

    def _f_psi(
        self,
        seq: torch.Tensor,
        z_init: torch.Tensor,
        pos_embed: torch.Tensor,
        proj_in: nn.Linear,
        proj_out: nn.Linear,
    ) -> torch.Tensor:
        """D3T F_psi aggregation over one axis with learnable positional embedding.

        Args:
            seq:        [N, agg_len, in_channels]  N = batch * spatial positions
            z_init:     [1, 1, emb_dim]            learnable query (broadcast)
            pos_embed:  [1, agg_len+1, emb_dim]    learnable PE (broadcast)
            proj_in, proj_out: per-plane linear projections

        Returns:
            [N, out_channels]
        """
        N = seq.shape[0]
        tokens = proj_in(seq)  # [N, agg_len, emb_dim]
        z = z_init.expand(N, -1, -1)  # [N, 1, emb_dim]
        tokens = torch.cat([z, tokens], dim=1)  # [N, agg_len+1, emb_dim]
        tokens = tokens + pos_embed  # broadcast over N
        tokens = self.attn(tokens)
        return proj_out(tokens[:, 0])  # [N, out_channels]

    def forward(self, mu: torch.Tensor) -> dict[str, torch.Tensor]:
        # mu: [B, C, H, W, D]
        B, C, H, W, D = mu.shape

        # --- XY plane: aggregate over D, output [B, out_c, H, W] ---
        # permute: [B, H, W, D, C], reshape: [B*H*W, D, C]
        xy_seq = mu.permute(0, 2, 3, 4, 1).reshape(B * H * W, D, C)
        xy = self._f_psi(
            xy_seq, self.z_init_xy, self.pos_embed_xy, self.proj_in_xy, self.proj_out_xy
        )
        xy = xy.reshape(B, H, W, self.out_channels).permute(
            0, 3, 1, 2
        )  # [B, out_c, H, W]

        # --- YZ plane: aggregate over H, output [B, out_c, W, D] ---
        # permute: [B, W, D, H, C], reshape: [B*W*D, H, C]
        yz_seq = mu.permute(0, 3, 4, 2, 1).reshape(B * W * D, H, C)
        yz = self._f_psi(
            yz_seq, self.z_init_yz, self.pos_embed_yz, self.proj_in_yz, self.proj_out_yz
        )
        yz = yz.reshape(B, W, D, self.out_channels).permute(
            0, 3, 1, 2
        )  # [B, out_c, W, D]

        # --- XZ plane: aggregate over W, output [B, out_c, H, D] ---
        # permute: [B, H, D, W, C], reshape: [B*H*D, W, C]
        xz_seq = mu.permute(0, 2, 4, 3, 1).reshape(B * H * D, W, C)
        xz = self._f_psi(
            xz_seq, self.z_init_xz, self.pos_embed_xz, self.proj_in_xz, self.proj_out_xz
        )
        xz = xz.reshape(B, H, D, self.out_channels).permute(
            0, 3, 1, 2
        )  # [B, out_c, H, D]

        return {"z_xy": xy, "z_yz": yz, "z_xz": xz}
