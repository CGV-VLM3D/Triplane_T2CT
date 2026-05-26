"""Forward/backward pass: no NaN/Inf, gradients flow to all trainable params.

Tests the Compress3D TriplaneAE with the ReconLoss used during training.
The AE forward() returns a _DictWithTupleUnpack, so both dict-style access
and legacy ``mu_hat, _ = model(mu)`` are exercised here.
"""

from __future__ import annotations

import torch
import pytest

from src.models.triplane_ae import TriplaneAE
from src.losses.recon_loss import ReconLoss

LATENT_SHAPE = (12, 8, 6)
B = 2


@pytest.fixture(scope="module")
def ae():
    return TriplaneAE(
        in_channels=4,
        latent_shape=LATENT_SHAPE,
        plane_channels=8,
        latent_channels=8,
        n_resblocks=(1, 1, 1),
        use_cross_attn=False,
        decoder_n_tri_resblocks=1,
    )


@pytest.fixture(scope="module")
def loss_fn():
    return ReconLoss(l1_weight=1.0)


@pytest.fixture(scope="module")
def result(ae, loss_fn):
    """Run one forward+backward; return (mu, mu_hat, loss_dict)."""
    torch.manual_seed(42)
    mu = torch.randn(B, 4, *LATENT_SHAPE)
    mu_hat, _ = ae(mu)
    loss_dict = loss_fn(mu_hat, mu)
    loss_dict["total"].backward()
    return mu, mu_hat, loss_dict


def test_output_no_nan(result):
    _, mu_hat, _ = result
    assert not torch.isnan(mu_hat).any(), "NaN in mu_hat"
    assert not torch.isinf(mu_hat).any(), "Inf in mu_hat"


def test_loss_no_nan(result):
    _, _, loss_dict = result
    assert not torch.isnan(loss_dict["total"]), "NaN in total loss"
    assert not torch.isinf(loss_dict["total"]), "Inf in total loss"


def test_all_params_have_grad(ae, result):
    missing = [
        name for name, p in ae.named_parameters() if p.requires_grad and p.grad is None
    ]
    assert not missing, f"No gradient for: {missing}"


def test_no_nan_grads(ae, result):
    bad = [
        name
        for name, p in ae.named_parameters()
        if p.requires_grad and p.grad is not None and torch.isnan(p.grad).any()
    ]
    assert not bad, f"NaN gradient in: {bad}"
