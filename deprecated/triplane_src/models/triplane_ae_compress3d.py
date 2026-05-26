"""TriplaneAECompress3D: paper-faithful Compress3D VAE for MAISI latents.

Wires ``TriplaneEncoderCompress3D`` + ``TriplaneDecoderCompress3D`` with the
standard reparameterize / KL pieces shared with ``TriplaneAE``. Latent lives
at low-res (4× spatial downsample of the input triplane axes).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .triplane_decoder_compress3d import TriplaneDecoderCompress3D
from .triplane_encoder_compress3d import TriplaneEncoderCompress3D


class TriplaneAECompress3D(nn.Module):
    """Compress3D-style triplane VAE (paper-faithful).

    Args:
        in_channels:                  MAISI latent input channels (4).
        latent_shape:                 (H, W, D) of the input volume.
        feature_lift_channels:        c after the 1x1x1 input Conv3d.
        plane_channels_progression:   (c1, c2, c3) channel widths across the
                                      encoder's three stages (paper: 32,64,128).
        enc_n_blocks_per_stage:       (n1, n2, n3) encoder ResBlock counts.
        dec_n_blocks_per_stage:       (n1, n2, n3) decoder ResBlock counts.
        n_triplane_downsamples:       fixed at 2 (paper).
        volume_downsample:            o in paper Eq 6 (default 2).
        volume_kv_channels:           c'' of V^n_d (default = feature_lift_channels).
        cross_attn_heads:             MHA heads.
        cross_attn_d_kv:              per-head K/V dim.
        latent_channels:              channel dim of the per-plane latent.
    """

    def __init__(
        self,
        in_channels: int = 4,
        latent_shape: tuple[int, int, int] = (60, 60, 32),
        feature_lift_channels: int = 32,
        plane_channels_progression: tuple[int, int, int] = (32, 64, 128),
        enc_n_blocks_per_stage: tuple[int, int, int] = (2, 2, 4),
        dec_n_blocks_per_stage: tuple[int, int, int] = (3, 3, 5),
        n_triplane_downsamples: int = 2,
        volume_downsample: int = 2,
        volume_kv_channels: int | None = None,
        cross_attn_heads: int = 4,
        cross_attn_d_kv: int = 32,
        latent_channels: int = 32,
    ) -> None:
        super().__init__()
        latent_shape = tuple(int(v) for v in latent_shape)

        self.encoder = TriplaneEncoderCompress3D(
            in_channels=in_channels,
            latent_shape=latent_shape,
            feature_lift_channels=feature_lift_channels,
            plane_channels_progression=plane_channels_progression,
            n_blocks_per_stage=enc_n_blocks_per_stage,
            n_triplane_downsamples=n_triplane_downsamples,
            volume_downsample=volume_downsample,
            volume_kv_channels=volume_kv_channels,
            cross_attn_heads=cross_attn_heads,
            cross_attn_d_kv=cross_attn_d_kv,
            latent_channels=latent_channels,
        )
        self.decoder = TriplaneDecoderCompress3D(
            latent_shape=latent_shape,
            plane_channels_progression=plane_channels_progression,
            n_blocks_per_stage=dec_n_blocks_per_stage,
            latent_channels=latent_channels,
            out_channels=in_channels,
            n_upsamples=n_triplane_downsamples,
        )

    @classmethod
    def from_config(cls, cfg) -> "TriplaneAECompress3D":
        enc = cfg.model.encoder
        dec = getattr(cfg.model, "decoder", None)

        def _get(obj, key, default):
            return getattr(obj, key, default) if obj is not None else default

        return cls(
            in_channels=int(getattr(enc, "in_channels", 4)),
            latent_shape=tuple(int(v) for v in enc.latent_shape),
            feature_lift_channels=int(getattr(enc, "feature_lift_channels", 32)),
            plane_channels_progression=tuple(
                int(v)
                for v in getattr(enc, "plane_channels_progression", (32, 64, 128))
            ),
            enc_n_blocks_per_stage=tuple(
                int(v) for v in getattr(enc, "n_blocks_per_stage", (2, 2, 4))
            ),
            dec_n_blocks_per_stage=tuple(
                int(v) for v in _get(dec, "n_blocks_per_stage", (3, 3, 5))
            ),
            n_triplane_downsamples=int(getattr(enc, "n_triplane_downsamples", 2)),
            volume_downsample=int(getattr(enc, "volume_downsample", 2)),
            volume_kv_channels=(
                int(getattr(enc, "volume_kv_channels"))
                if getattr(enc, "volume_kv_channels", None) is not None
                else None
            ),
            cross_attn_heads=int(getattr(enc, "cross_attn_heads", 4)),
            cross_attn_d_kv=int(getattr(enc, "cross_attn_d_kv", 32)),
            latent_channels=int(getattr(enc, "latent_channels", 32)),
        )

    def reparameterize(
        self, mu: dict[str, torch.Tensor], logvar: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        if self.training:
            return {
                k: mu[k]
                + torch.randn_like(mu[k]) * (0.5 * logvar[k].clamp(-30, 20)).exp()
                for k in mu
            }
        return mu

    def kl_loss(
        self, mu: dict[str, torch.Tensor], logvar: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        kl = 0.0
        for k in mu:
            mu_k = mu[k]
            lv_k = logvar[k].clamp(-30, 20)
            kl = kl + 0.5 * (mu_k.pow(2) + lv_k.exp() - lv_k - 1).mean()
        return kl / len(mu)  # type: ignore[return-value]

    def forward(self, z: torch.Tensor):
        """Same interface as TriplaneAE: returns a dict that also tuple-unpacks
        as ``mu_hat, aux = model(z)`` for train.py compatibility."""
        enc_out = self.encoder(z)
        mu_planes = enc_out["mu"]
        logvar_planes = enc_out["logvar"]

        z_sample = self.reparameterize(mu_planes, logvar_planes)
        mu_hat = self.decoder(z_sample)
        kl = self.kl_loss(mu_planes, logvar_planes)

        return _DictWithTupleUnpack(
            {
                "mu_hat": mu_hat,
                "mu_planes": mu_planes,
                "logvar_planes": logvar_planes,
                "kl_loss": kl,
            }
        )


class _DictWithTupleUnpack(dict):
    """Allow ``mu_hat, _aux = model(z)`` legacy callers."""

    def __iter__(self):
        return iter((self["mu_hat"], dict(self)))

    def __len__(self):
        return 2
