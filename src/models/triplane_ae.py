from __future__ import annotations

import torch
import torch.nn as nn

from .triplane_decoder import TriplaneDecoder
from .triplane_encoder import TriplaneEncoder


class TriplaneAE(nn.Module):
    """Encoder + decoder wrapper for the triplane autoencoder."""

    def __init__(
        self,
        in_channels: int = 4,
        emb_dim: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        out_channels: int = 8,
        decoder_hidden: int = 32,
        decoder_n_res_blocks: int = 2,
        latent_shape: tuple[int, int, int] = (120, 120, 64),
        patch_size: int = 4,
    ) -> None:
        super().__init__()
        self.encoder = TriplaneEncoder(
            in_channels=in_channels,
            emb_dim=emb_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            out_channels=out_channels,
            latent_shape=latent_shape,
            patch_size=patch_size,
        )
        self.decoder = TriplaneDecoder(
            in_channels=out_channels,
            hidden=decoder_hidden,
            out_channels=in_channels,
            latent_shape=latent_shape,
            patch_size=patch_size,
            n_res_blocks=decoder_n_res_blocks,
        )

    @classmethod
    def from_config(cls, cfg):
        enc = cfg.model.encoder
        dec = cfg.model.decoder
        return cls(
            in_channels=int(enc.in_channels),
            emb_dim=int(enc.emb_dim),
            n_layers=int(enc.n_layers),
            n_heads=int(enc.n_heads),
            out_channels=int(enc.out_channels),
            decoder_hidden=int(dec.hidden),
            decoder_n_res_blocks=int(getattr(dec, "n_res_blocks", 2)),
            latent_shape=tuple(enc.latent_shape),
            patch_size=int(getattr(enc, "patch_size", 1)),
        )

    def forward(self, mu: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        triplane = self.encoder(mu)
        mu_hat = self.decoder(triplane["z_xy"], triplane["z_yz"], triplane["z_xz"])
        return mu_hat, triplane
