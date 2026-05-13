"""Verify that D3T z_init parameters receive nonzero gradients."""

from __future__ import annotations

import torch
import pytest

from src.models.triplane_ae import TriplaneAE

LATENT_SHAPE = (120, 120, 64)


@pytest.fixture(scope="module")
def model():
    return TriplaneAE(
        in_channels=4,
        emb_dim=64,
        n_layers=2,
        n_heads=4,
        out_channels=8,
        decoder_hidden=16,
        latent_shape=LATENT_SHAPE,
    )


def test_z_init_gradients(model):
    torch.manual_seed(0)
    mu = torch.randn(1, 4, *LATENT_SHAPE)
    mu_hat, _ = model(mu)
    mu_hat.sum().backward()

    enc = model.encoder

    assert enc.z_init_xy.grad is not None, "z_init_xy has no gradient"
    assert enc.z_init_xy.grad.abs().sum() > 0, "z_init_xy gradient is all zeros"

    assert enc.z_init_yz.grad is not None, "z_init_yz has no gradient"
    assert enc.z_init_yz.grad.abs().sum() > 0, "z_init_yz gradient is all zeros"

    assert enc.z_init_xz.grad is not None, "z_init_xz has no gradient"
    assert enc.z_init_xz.grad.abs().sum() > 0, "z_init_xz gradient is all zeros"
