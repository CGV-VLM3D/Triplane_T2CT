"""Shape checks for encoder, decoder, and round-trip AE."""

from __future__ import annotations

import torch
import pytest

from src.models.triplane_encoder import TriplaneEncoder
from src.models.triplane_decoder import TriplaneDecoder
from src.models.triplane_ae import TriplaneAE

# Reduced dims to keep CPU memory reasonable during testing.
# emb_dim=64, n_heads=4 instead of 512/8.
LATENT_SHAPE = (120, 120, 64)
B = 2


@pytest.fixture(scope="module")
def encoder():
    return TriplaneEncoder(
        in_channels=4,
        emb_dim=64,
        n_layers=2,
        n_heads=4,
        out_channels=8,
        latent_shape=LATENT_SHAPE,
    ).eval()


@pytest.fixture(scope="module")
def decoder():
    return TriplaneDecoder(
        in_channels=8,
        hidden=16,
        out_channels=4,
        latent_shape=LATENT_SHAPE,
    ).eval()


@pytest.fixture(scope="module")
def ae():
    return TriplaneAE(
        in_channels=4,
        emb_dim=64,
        n_layers=2,
        n_heads=4,
        out_channels=8,
        decoder_hidden=16,
        latent_shape=LATENT_SHAPE,
    ).eval()


@pytest.fixture(scope="module")
def mu():
    torch.manual_seed(0)
    return torch.randn(B, 4, *LATENT_SHAPE)


def test_encoder_xy_shape(encoder, mu):
    with torch.no_grad():
        planes = encoder(mu)
    assert planes["z_xy"].shape == (B, 8, 120, 120), planes["z_xy"].shape


def test_encoder_yz_shape(encoder, mu):
    with torch.no_grad():
        planes = encoder(mu)
    assert planes["z_yz"].shape == (B, 8, 120, 64), planes["z_yz"].shape


def test_encoder_xz_shape(encoder, mu):
    with torch.no_grad():
        planes = encoder(mu)
    assert planes["z_xz"].shape == (B, 8, 120, 64), planes["z_xz"].shape


def test_decoder_shape(encoder, decoder, mu):
    with torch.no_grad():
        planes = encoder(mu)
        mu_hat = decoder(planes["z_xy"], planes["z_yz"], planes["z_xz"])
    assert mu_hat.shape == (B, 4, 120, 120, 64), mu_hat.shape


def test_ae_roundtrip_shape(ae, mu):
    with torch.no_grad():
        mu_hat, _ = ae(mu)
    assert mu_hat.shape == mu.shape, mu_hat.shape
