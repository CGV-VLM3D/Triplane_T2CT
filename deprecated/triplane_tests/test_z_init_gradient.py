"""Gradient-flow checks for the Compress3D TriplaneAE encoder.

The old transformer encoder had learnable z_init_xy/yz/xz embeddings whose
gradient flow we verified here.  The new Compress3D encoder has no such
embeddings; this file now checks that ALL encoder parameters receive
non-zero gradients in a standard backward pass.
"""

from __future__ import annotations

import torch
import pytest

from src.models.triplane_ae import TriplaneAE

LATENT_SHAPE = (12, 8, 6)


@pytest.fixture(scope="module")
def model():
    return TriplaneAE(
        in_channels=4,
        latent_shape=LATENT_SHAPE,
        plane_channels=8,
        latent_channels=8,
        n_resblocks=(1, 1, 1),
        use_cross_attn=False,
        decoder_n_tri_resblocks=1,
    )


def test_encoder_params_receive_gradients(model):
    """Every encoder parameter must get a nonzero gradient after one backward pass."""
    torch.manual_seed(0)
    mu = torch.randn(1, 4, *LATENT_SHAPE)
    out = model(mu)
    (out["mu_hat"].sum() + out["kl_loss"]).backward()

    missing = []
    zero_grad = []
    for name, param in model.encoder.named_parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            missing.append(name)
        elif param.grad.abs().sum() == 0:
            zero_grad.append(name)

    assert not missing, f"No gradient at all for encoder params: {missing}"
    assert not zero_grad, f"All-zero gradient for encoder params: {zero_grad}"
