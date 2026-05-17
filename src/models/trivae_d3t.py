"""D3T-style triplane autoencoder for MAISI latent reconstruction.

Architecture:
  - 3D Swin Transformer Encoder (E_psi): PatchEmbed + N SwinTransformerBlocks
  - F_psi Projective Transformer: per-plane z_init + cross-axis MSA collapse
  - 3D Swin Transformer Decoder (D_psi): broadcast-sum + M SwinTransformerBlocks + ConvTranspose3d

Reference: D3T paper Sec. 4.2, Fig. 4 (TriVQAE).  VQ is skipped here —
continuous triplane bottleneck only; add VQ later if needed.

Input/Output: MAISI latent [B, 4, 120, 120, 64] -> round-trip same shape.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from monai.networks.nets.swin_unetr import (
    SwinTransformerBlock,
    compute_mask,
    PatchEmbed,
)


__all__ = ["TriVAE_D3T"]


def _build_swin_blocks(
    dim: int,
    num_heads: int,
    window_size: tuple[int, int, int],
    n_layers: int,
) -> nn.ModuleList:
    """Alternate W-MSA (shift=0) and SW-MSA (shift=half-window) blocks."""
    half_ws = tuple(w // 2 for w in window_size)
    blocks = nn.ModuleList()
    for i in range(n_layers):
        shift = (0, 0, 0) if i % 2 == 0 else half_ws
        blocks.append(
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=shift,
            )
        )
    return blocks


class SwinStack(nn.Module):
    """N alternating W-MSA/SW-MSA blocks operating on [B, D, H, W, C] tensors.

    MONAI's SwinTransformerBlock pads internally when the spatial dims are not
    multiples of window_size, so we do not need to pad here.  The attention
    mask is cached at construction time from patched_grid_size and reused.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: tuple[int, int, int],
        n_layers: int,
        patched_grid_size: tuple[int, int, int],
    ) -> None:
        super().__init__()
        self.blocks = _build_swin_blocks(dim, num_heads, window_size, n_layers)
        half_ws = tuple(w // 2 for w in window_size)

        # Pre-compute attention masks (one per block, shifted blocks need one).
        # compute_mask returns None for shift=(0,0,0) — MONAI handles that.
        self._masks: list[torch.Tensor | None] = []
        for i in range(n_layers):
            shift = (0, 0, 0) if i % 2 == 0 else half_ws
            mask = compute_mask(
                list(patched_grid_size), window_size, shift, torch.device("cpu")
            )
            self._masks.append(mask)  # moved to device in forward

        self._device_set = False
        self._mask_tensors: list[torch.Tensor | None] = [None] * n_layers

    def _ensure_masks_on_device(self, device: torch.device) -> None:
        if not self._device_set:
            for i, m in enumerate(self._masks):
                self._mask_tensors[i] = m.to(device) if m is not None else None
            self._device_set = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D, H, W, C]
        self._ensure_masks_on_device(x.device)
        for block, mask in zip(self.blocks, self._mask_tensors):
            x = block(x, mask)
        return x


class FPsiPlane(nn.Module):
    """Projective transformer that collapses one axis to produce a 2D plane.

    For a given patch grid [B, Hp, Wp, Dp, C], collapses `agg_axis` (0=Hp,
    1=Wp, 2=Dp) via prepending a learnable z_init token, running shared MSA,
    then reading position 0.

    Output plane shape after un-batching:
        agg_axis=0 (Hp):  [B, out_ch, Wp, Dp]  (YZ plane)
        agg_axis=1 (Wp):  [B, out_ch, Hp, Dp]  (XZ plane)
        agg_axis=2 (Dp):  [B, out_ch, Hp, Wp]  (XY plane)
    """

    def __init__(
        self,
        emb_dim: int,
        out_channels: int,
        agg_len: int,
        n_layers: int,
        n_heads: int,
    ) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.agg_len = agg_len
        self.out_channels = out_channels

        self.proj_in = nn.Linear(emb_dim, emb_dim)
        self.proj_out = nn.Linear(emb_dim, out_channels)
        self.z_init = nn.Parameter(torch.zeros(1, 1, emb_dim))
        nn.init.trunc_normal_(self.z_init, std=0.02)

        # Learnable positional embedding: z_init slot (pos 0) + agg_len tokens.
        self.pos_embed = nn.Parameter(torch.zeros(1, agg_len + 1, emb_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Independent attention layers (not shared across planes in D3T).
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

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        seq: [N, agg_len, emb_dim]  (N = B * non_agg_h * non_agg_w)
        returns: [N, out_channels]
        """
        N = seq.shape[0]
        tokens = self.proj_in(seq)
        tokens = torch.cat([self.z_init.expand(N, -1, -1), tokens], dim=1)
        tokens = tokens + self.pos_embed
        out = self.attn(tokens)
        return self.proj_out(out[:, 0])


class TriVAE_D3T(nn.Module):
    """D3T-style triplane autoencoder (continuous bottleneck, no VQ).

    Args:
        in_channels:       Input latent channels (4 for MAISI).
        emb_dim:           Patch embedding / Swin block dimension.
        patch_size:        Spatial patch size (isotropic, applied to H/W/D).
        n_swin_enc_layers: Number of Swin blocks in the encoder.
        swin_window_size:  Window size for Swin blocks; must divide patched dims.
        swin_num_heads:    Attention heads in Swin blocks.
        n_fpsi_layers:     Transformer layers in each F_psi projective module.
        fpsi_num_heads:    Attention heads in F_psi modules.
        out_channels:      Triplane channel width (bottleneck width).
        n_swin_dec_layers: Number of Swin blocks in the decoder.
        latent_shape:      (H, W, D) of the input MAISI latent.
    """

    def __init__(
        self,
        in_channels: int = 4,
        emb_dim: int = 256,
        patch_size: int = 4,
        n_swin_enc_layers: int = 4,
        swin_window_size: tuple = (5, 5, 4),
        swin_num_heads: int = 8,
        n_fpsi_layers: int = 2,
        fpsi_num_heads: int = 8,
        out_channels: int = 8,
        n_swin_dec_layers: int = 4,
        latent_shape: tuple = (120, 120, 64),
    ) -> None:
        super().__init__()

        latent_shape = tuple(int(v) for v in latent_shape)
        swin_window_size = tuple(int(v) for v in swin_window_size)
        H, W, D = latent_shape
        p = patch_size
        Hp, Wp, Dp = H // p, W // p, D // p

        assert H % p == 0 and W % p == 0 and D % p == 0, (
            f"latent_shape {latent_shape} must be divisible by patch_size={p}"
        )
        assert Hp % swin_window_size[0] == 0, (
            f"Patched H={Hp} not divisible by swin_window_size[0]={swin_window_size[0]}"
        )
        assert Wp % swin_window_size[1] == 0, (
            f"Patched W={Wp} not divisible by swin_window_size[1]={swin_window_size[1]}"
        )
        assert Dp % swin_window_size[2] == 0, (
            f"Patched D={Dp} not divisible by swin_window_size[2]={swin_window_size[2]}"
        )

        self.in_channels = in_channels
        self.emb_dim = emb_dim
        self.patch_size = p
        self.out_channels = out_channels
        self.Hp, self.Wp, self.Dp = Hp, Wp, Dp
        self.latent_shape = latent_shape

        # ---- 1. Encoder: PatchEmbed + Swin stack ----
        # PatchEmbed returns [B, C, Hp, Wp, Dp]
        self.patch_embed = PatchEmbed(
            patch_size=p,
            in_chans=in_channels,
            embed_dim=emb_dim,
            spatial_dims=3,
        )
        # Swin operates on [B, Hp, Wp, Dp, C] (channel-last)
        self.enc_swin = SwinStack(
            dim=emb_dim,
            num_heads=swin_num_heads,
            window_size=swin_window_size,
            n_layers=n_swin_enc_layers,
            patched_grid_size=(Hp, Wp, Dp),
        )

        # ---- 2. F_psi projective transformers (one per plane) ----
        # XY collapses Dp: sequences of length Dp, N = B*Hp*Wp
        self.fpsi_xy = FPsiPlane(
            emb_dim,
            out_channels,
            agg_len=Dp,
            n_layers=n_fpsi_layers,
            n_heads=fpsi_num_heads,
        )
        # YZ collapses Hp: sequences of length Hp, N = B*Wp*Dp
        self.fpsi_yz = FPsiPlane(
            emb_dim,
            out_channels,
            agg_len=Hp,
            n_layers=n_fpsi_layers,
            n_heads=fpsi_num_heads,
        )
        # XZ collapses Wp: sequences of length Wp, N = B*Hp*Dp
        self.fpsi_xz = FPsiPlane(
            emb_dim,
            out_channels,
            agg_len=Wp,
            n_layers=n_fpsi_layers,
            n_heads=fpsi_num_heads,
        )

        # ---- 3. Decoder ----
        # Project triplane out_channels back to emb_dim for Swin blocks
        self.dec_proj_in = nn.Conv3d(out_channels, emb_dim, kernel_size=1)
        self.dec_swin = SwinStack(
            dim=emb_dim,
            num_heads=swin_num_heads,
            window_size=swin_window_size,
            n_layers=n_swin_dec_layers,
            patched_grid_size=(Hp, Wp, Dp),
        )
        # Unpatchify: mirrors PatchEmbed (channel-first after Swin stack)
        self.unpatchify = nn.ConvTranspose3d(
            emb_dim, in_channels, kernel_size=p, stride=p
        )

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """x: [B, 4, H, W, D] -> {z_xy, z_yz, z_xz}"""
        B = x.shape[0]
        Hp, Wp, Dp = self.Hp, self.Wp, self.Dp

        # PatchEmbed: [B, emb_dim, Hp, Wp, Dp]
        feat = self.patch_embed(x)

        # Swin encoder: rearrange to channel-last [B, Hp, Wp, Dp, C]
        feat = feat.permute(0, 2, 3, 4, 1).contiguous()
        feat = self.enc_swin(feat)  # [B, Hp, Wp, Dp, C]

        # F_psi: produce three 2D planes via axis collapse

        # XY plane — collapse Dp axis
        # Reshape: [B*Hp*Wp, Dp, C]
        xy_seq = feat.reshape(B * Hp * Wp, Dp, self.emb_dim)
        xy = self.fpsi_xy(xy_seq)  # [B*Hp*Wp, out_ch]
        xy = xy.reshape(B, Hp, Wp, self.out_channels).permute(
            0, 3, 1, 2
        )  # [B, out_ch, Hp, Wp]

        # YZ plane — collapse Hp axis
        # permute to [B, Wp, Dp, Hp, C], reshape [B*Wp*Dp, Hp, C]
        yz_seq = feat.permute(0, 2, 3, 1, 4).reshape(B * Wp * Dp, Hp, self.emb_dim)
        yz = self.fpsi_yz(yz_seq)  # [B*Wp*Dp, out_ch]
        yz = yz.reshape(B, Wp, Dp, self.out_channels).permute(
            0, 3, 1, 2
        )  # [B, out_ch, Wp, Dp]

        # XZ plane — collapse Wp axis
        # permute to [B, Hp, Dp, Wp, C], reshape [B*Hp*Dp, Wp, C]
        xz_seq = feat.permute(0, 1, 3, 2, 4).reshape(B * Hp * Dp, Wp, self.emb_dim)
        xz = self.fpsi_xz(xz_seq)  # [B*Hp*Dp, out_ch]
        xz = xz.reshape(B, Hp, Dp, self.out_channels).permute(
            0, 3, 1, 2
        )  # [B, out_ch, Hp, Dp]

        return {"z_xy": xy, "z_yz": yz, "z_xz": xz}

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def decode(self, z: dict[str, torch.Tensor]) -> torch.Tensor:
        """z: {z_xy, z_yz, z_xz} -> [B, 4, H, W, D]"""
        z_xy = z["z_xy"]  # [B, out_ch, Hp, Wp]
        z_yz = z["z_yz"]  # [B, out_ch, Wp, Dp]
        z_xz = z["z_xz"]  # [B, out_ch, Hp, Dp]
        B = z_xy.shape[0]
        Hp, Wp, Dp = self.Hp, self.Wp, self.Dp

        # Broadcast-sum: expand each plane to [B, out_ch, Hp, Wp, Dp] and sum
        xy3d = z_xy.unsqueeze(4).expand(B, self.out_channels, Hp, Wp, Dp).contiguous()
        yz3d = z_yz.unsqueeze(2).expand(B, self.out_channels, Hp, Wp, Dp).contiguous()
        xz3d = z_xz.unsqueeze(3).expand(B, self.out_channels, Hp, Wp, Dp).contiguous()
        fused = xy3d + yz3d + xz3d  # [B, out_ch, Hp, Wp, Dp]

        # Project to emb_dim for Swin decoder
        feat = self.dec_proj_in(fused)  # [B, emb_dim, Hp, Wp, Dp]

        # Swin decoder: channel-last
        feat = feat.permute(0, 2, 3, 4, 1).contiguous()  # [B, Hp, Wp, Dp, emb_dim]
        feat = self.dec_swin(feat)
        feat = feat.permute(0, 4, 1, 2, 3).contiguous()  # [B, emb_dim, Hp, Wp, Dp]

        # Unpatchify -> [B, in_channels, H, W, D]
        return self.unpatchify(feat)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg):
        m = cfg.model
        return cls(
            in_channels=int(getattr(m, "in_channels", 4)),
            emb_dim=int(getattr(m, "emb_dim", 256)),
            patch_size=int(getattr(m, "patch_size", 4)),
            n_swin_enc_layers=int(getattr(m, "n_swin_enc_layers", 4)),
            swin_window_size=tuple(m.swin_window_size),
            swin_num_heads=int(getattr(m, "swin_num_heads", 8)),
            n_fpsi_layers=int(getattr(m, "n_fpsi_layers", 2)),
            fpsi_num_heads=int(getattr(m, "fpsi_num_heads", 8)),
            out_channels=int(getattr(m, "out_channels", 8)),
            n_swin_dec_layers=int(getattr(m, "n_swin_dec_layers", 4)),
            latent_shape=tuple(m.latent_shape),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """x: [B, 4, H, W, D] -> (reconstruction, triplane_dict)

        Returns a (recon, triplane) tuple to match TriplaneAE's calling
        convention so the existing train loop works without modification.
        """
        triplane = self.encode(x)
        recon = self.decode(triplane)
        return recon, triplane
