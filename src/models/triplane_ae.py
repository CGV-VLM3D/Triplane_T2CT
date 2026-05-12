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
        emb_dim: int = 512,
        n_layers: int = 4,
        n_heads: int = 8,
        out_channels: int = 8,
        decoder_hidden: int = 32,
        latent_shape: tuple[int, int, int] = (120, 120, 64),
    ) -> None:
        super().__init__()
        self.encoder = TriplaneEncoder(
            in_channels=in_channels,
            emb_dim=emb_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            out_channels=out_channels,
            latent_shape=latent_shape,
        )
        self.decoder = TriplaneDecoder(
            in_channels=out_channels,
            hidden=decoder_hidden,
            out_channels=in_channels,
            latent_shape=latent_shape,
        )

    def forward(self, mu: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        triplane = self.encoder(mu)
        mu_hat = self.decoder(triplane["z_xy"], triplane["z_yz"], triplane["z_xz"])
        return mu_hat, triplane
