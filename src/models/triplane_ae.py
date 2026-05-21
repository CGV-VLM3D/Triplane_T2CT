"""TriplaneAE: full triplane variational autoencoder.

Wires TriplaneEncoder (new Compress3D-style) + TriplaneDecoderD with VAE
reparameterization.  The old transformer-based TriplaneEncoder is replaced;
TriplaneDecoder (Decoder A) is kept in its own file for comparison.

forward() returns a dict so train.py can access mu_hat, kl_loss, and the
per-plane latents for downstream diffusion training.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .triplane_decoder_d import TriplaneDecoderD
from .triplane_encoder import TriplaneEncoder


class TriplaneAE(nn.Module):
    """Compress3D-style triplane VAE.

    Args:
        in_channels:            input/output MAISI latent channels (4).
        latent_shape:           (H, W, D) spatial dims of the 3D latent.
        plane_channels:         intermediate channel width.
        latent_channels:        channel dim of the per-plane latent (mu/logvar).
        n_resblocks:            (n1, n2, n3) 2D ResBlock counts in encoder.
        downsampling:           whether encoder projects the volume before attn.
        use_cross_attn:         whether encoder uses CrossAttention3D.
        cross_attn_heads:       attention heads.
        cross_attn_d_kv:        per-head K/V dim.
        cross_attn_window:      attention window size (only 1 supported).
        decoder_n_tri_resblocks: TriConvResBlock count in decoder.
    """

    def __init__(
        self,
        in_channels: int = 4,
        latent_shape: tuple[int, int, int] = (60, 60, 32),
        plane_channels: int = 16,
        latent_channels: int = 16,
        n_resblocks: tuple[int, int, int] = (2, 2, 4),
        downsampling: bool = False,
        use_cross_attn: bool = True,
        cross_attn_heads: int = 4,
        cross_attn_d_kv: int = 16,
        cross_attn_window: int = 1,
        decoder_n_tri_resblocks: int = 3,
    ) -> None:
        super().__init__()
        latent_shape = tuple(int(v) for v in latent_shape)

        self.encoder = TriplaneEncoder(
            in_channels=in_channels,
            latent_shape=latent_shape,
            plane_channels=plane_channels,
            latent_channels=latent_channels,
            n_resblocks=n_resblocks,
            downsampling=downsampling,
            use_cross_attn=use_cross_attn,
            cross_attn_heads=cross_attn_heads,
            cross_attn_d_kv=cross_attn_d_kv,
            cross_attn_window=cross_attn_window,
        )
        self.decoder = TriplaneDecoderD(
            latent_shape=latent_shape,
            plane_channels=plane_channels,
            latent_channels=latent_channels,
            n_tri_resblocks=decoder_n_tri_resblocks,
            out_channels=in_channels,
        )

    @classmethod
    def from_config(cls, cfg) -> "TriplaneAE":
        enc = cfg.model.encoder
        dec = getattr(cfg.model, "decoder", None)

        def _get(obj, key, default):
            return getattr(obj, key, default) if obj is not None else default

        return cls(
            in_channels=int(getattr(enc, "in_channels", 4)),
            latent_shape=tuple(int(v) for v in enc.latent_shape),
            plane_channels=int(getattr(enc, "plane_channels", 16)),
            latent_channels=int(getattr(enc, "latent_channels", 16)),
            n_resblocks=tuple(int(v) for v in getattr(enc, "n_resblocks", (2, 2, 4))),
            downsampling=bool(getattr(enc, "downsampling", False)),
            use_cross_attn=bool(getattr(enc, "use_cross_attn", True)),
            cross_attn_heads=int(getattr(enc, "cross_attn_heads", 4)),
            cross_attn_d_kv=int(getattr(enc, "cross_attn_d_kv", 16)),
            cross_attn_window=int(getattr(enc, "cross_attn_window", 1)),
            decoder_n_tri_resblocks=int(_get(dec, "n_tri_resblocks", 3)),
        )

    def reparameterize(
        self, mu: dict[str, torch.Tensor], logvar: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Sample z ~ N(mu, exp(logvar)) during training, else return mu.

        logvar is clamped to [-30, 20] before exp() to prevent overflow when
        the KL weight is zero or very small (standard VAE practice).
        """
        if self.training:
            return {
                k: mu[k] + torch.randn_like(mu[k]) * (0.5 * logvar[k].clamp(-30, 20)).exp()
                for k in mu
            }
        return mu

    def kl_loss(
        self, mu: dict[str, torch.Tensor], logvar: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """KL divergence: 0.5 * mean(mu^2 + sigma^2 - log(sigma^2) - 1).

        logvar is clamped to [-30, 20] to match reparameterize and prevent
        inf/NaN when kl_weight=0 and logvar drifts freely during training.
        """
        kl = 0.0
        for k in mu:
            mu_k = mu[k]
            lv_k = logvar[k].clamp(-30, 20)
            kl = kl + 0.5 * (mu_k.pow(2) + lv_k.exp() - lv_k - 1).mean()
        # Average over the 3 planes.
        return kl / len(mu)  # type: ignore[return-value]

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: [B, in_channels, H, W, D]

        Returns dict:
            {
                "mu_hat":        [B, in_channels, H, W, D]  — reconstruction,
                "mu_planes":     {"xy":..., "yz":..., "xz":...} — per-plane mu,
                "logvar_planes": {"xy":..., "yz":..., "xz":...} — per-plane logvar,
                "kl_loss":       scalar tensor,
            }

        Also supports legacy tuple-unpack `mu_hat, aux = model(z)` used by
        train.py's existing TriVQAEConv-compatible call sites:
            mu_hat, _ = model(mu)
        by making the return value a _DictWithTupleUnpack.
        """
        enc_out = self.encoder(z)
        mu_planes = enc_out["mu"]
        logvar_planes = enc_out["logvar"]

        z_sample = self.reparameterize(mu_planes, logvar_planes)
        mu_hat = self.decoder(z_sample)
        kl = self.kl_loss(mu_planes, logvar_planes)

        result = {
            "mu_hat": mu_hat,
            "mu_planes": mu_planes,
            "logvar_planes": logvar_planes,
            "kl_loss": kl,
        }
        # Wrap so callers doing `out, aux = model(x)` still work.
        return _DictWithTupleUnpack(result)


class _DictWithTupleUnpack(dict):
    """dict subclass that also supports two-element tuple unpacking.

    Allows legacy code:  ``mu_hat, _aux = model(mu)``
    as well as new code: ``out["mu_hat"]``.
    """

    def __iter__(self):
        # When used as `a, b = obj`, Python calls iter() then next() twice.
        # We yield mu_hat first, then the full dict as aux, to match the
        # (recon, aux) convention of TriVQAEConv.
        return iter((self["mu_hat"], dict(self)))

    def __len__(self):
        # Report length=2 so tuple-unpacking `a, b = obj` works.
        return 2
