"""Forward/backward pass: no NaN/Inf, gradients flow to all trainable params."""

from __future__ import annotations

import torch
import pytest

from src.models.triplane_ae import TriplaneAE
from src.losses.recon_loss import ReconLoss

LATENT_SHAPE = (120, 120, 64)
B = 2


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
    # backward() was called in the result fixture
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
