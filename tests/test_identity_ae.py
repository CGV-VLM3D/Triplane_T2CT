"""Tests for IdentityAE — the no-op sanity-check autoencoder."""

from __future__ import annotations

import torch
import pytest

from src.models.identity_ae import IdentityAE


@pytest.fixture()
def model() -> IdentityAE:
    return IdentityAE(in_channels=4, emb_dim=512, latent_shape=(120, 120, 64))


def _rand(seed: int = 0) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randn(2, 4, 120, 120, 64, generator=g)


# ---------------------------------------------------------------------------
# Shape preservation
# ---------------------------------------------------------------------------


def test_output_shape(model: IdentityAE) -> None:
    mu = _rand()
    mu_hat, triplane = model(mu)
    assert mu_hat.shape == mu.shape, f"expected {mu.shape}, got {mu_hat.shape}"


# ---------------------------------------------------------------------------
# Exact equality
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_exact_equality(model: IdentityAE, seed: int) -> None:
    mu = _rand(seed)
    mu_hat, _ = model(mu)
    assert torch.equal(mu_hat, mu), "IdentityAE output must equal input exactly"


# ---------------------------------------------------------------------------
# triplane_dict contract
# ---------------------------------------------------------------------------


def test_triplane_dict_is_dict(model: IdentityAE) -> None:
    _, triplane = model(_rand())
    assert isinstance(triplane, dict)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_backward_no_error(model: IdentityAE) -> None:
    mu = _rand().requires_grad_(True)
    mu_hat, _ = model(mu)
    loss = (
        mu_hat.sum() + model._dummy.sum()
    )  # include dummy so optimizer is non-trivial
    loss.backward()
    assert mu.grad is not None, "gradient did not flow back to input"


def test_dummy_param_receives_gradient(model: IdentityAE) -> None:
    mu = _rand().requires_grad_(True)
    mu_hat, _ = model(mu)
    loss = mu_hat.sum() + model._dummy.sum()
    loss.backward()
    assert model._dummy.grad is not None, "_dummy parameter should have a gradient"
