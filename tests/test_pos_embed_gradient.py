"""Verify axial positional embeddings receive gradients and break permutation
invariance along the collapsed axis."""

from __future__ import annotations

import torch
import pytest

from src.models.triplane_ae import TriplaneAE
from src.models.triplane_encoder import TriplaneEncoder

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


def test_pos_embed_gradients(model):
    torch.manual_seed(0)
    mu = torch.randn(1, 4, *LATENT_SHAPE)
    mu_hat, _ = model(mu)
    mu_hat.sum().backward()

    enc = model.encoder
    for name in ("pos_embed_xy", "pos_embed_yz", "pos_embed_xz"):
        p = getattr(enc, name)
        assert p.grad is not None, f"{name} has no gradient"
        assert p.grad.abs().sum() > 0, f"{name} gradient is all zeros"


def test_axial_pe_breaks_permutation_invariance():
    """With learnable axial PE, permuting the collapsed axis must change the
    encoder output. Without PE the transformer would be permutation-equivariant.
    XY plane aggregates over D, so permuting D must affect z_xy."""
    enc = TriplaneEncoder(
        in_channels=4,
        emb_dim=64,
        n_layers=2,
        n_heads=4,
        out_channels=8,
        latent_shape=LATENT_SHAPE,
    ).eval()
    torch.manual_seed(0)
    mu = torch.randn(1, 4, *LATENT_SHAPE)
    perm = torch.randperm(LATENT_SHAPE[2])
    mu_perm = mu[..., perm]
    with torch.no_grad():
        out_orig = enc(mu)["z_xy"]
        out_perm = enc(mu_perm)["z_xy"]
    assert not torch.allclose(out_orig, out_perm, atol=1e-4), (
        "z_xy is unchanged after permuting D -- PE may not be active"
    )
