"""Shape checks for patchified encoder, decoder, and round-trip AE."""

from __future__ import annotations

import pytest
import torch

from src.models.triplane_ae import TriplaneAE
from src.models.triplane_decoder import TriplaneDecoder
from src.models.triplane_encoder import TriplaneEncoder

LATENT_SHAPE = (120, 120, 64)
PATCH_SIZE = 4
# Patchified spatial dims: 120//4=30, 120//4=30, 64//4=16
Hp, Wp, Dp = 30, 30, 16
B = 1  # keep CPU memory reasonable


@pytest.fixture(scope="module")
def encoder():
    return TriplaneEncoder(
        in_channels=4,
        emb_dim=64,
        n_layers=2,
        n_heads=4,
        out_channels=8,
        latent_shape=LATENT_SHAPE,
        patch_size=PATCH_SIZE,
    ).eval()


@pytest.fixture(scope="module")
def decoder():
    return TriplaneDecoder(
        in_channels=8,
        hidden=16,
        out_channels=4,
        latent_shape=LATENT_SHAPE,
        patch_size=PATCH_SIZE,
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
        patch_size=PATCH_SIZE,
    ).eval()


@pytest.fixture(scope="module")
def mu():
    torch.manual_seed(0)
    return torch.randn(B, 4, *LATENT_SHAPE)


# ---------------------------------------------------------------------------
# Triplane intermediate shapes (patch_size=4 → 30×30×16 patch grid)
# ---------------------------------------------------------------------------


def test_encoder_xy_shape(encoder, mu):
    with torch.no_grad():
        planes = encoder(mu)
    assert planes["z_xy"].shape == (B, 8, Hp, Wp), planes["z_xy"].shape


def test_encoder_yz_shape(encoder, mu):
    with torch.no_grad():
        planes = encoder(mu)
    assert planes["z_yz"].shape == (B, 8, Wp, Dp), planes["z_yz"].shape


def test_encoder_xz_shape(encoder, mu):
    with torch.no_grad():
        planes = encoder(mu)
    assert planes["z_xz"].shape == (B, 8, Hp, Dp), planes["z_xz"].shape


def test_decoder_shape(encoder, decoder, mu):
    with torch.no_grad():
        planes = encoder(mu)
        mu_hat = decoder(planes["z_xy"], planes["z_yz"], planes["z_xz"])
    assert mu_hat.shape == (B, 4, *LATENT_SHAPE), mu_hat.shape


def test_ae_roundtrip_shape(ae, mu):
    """Round-trip must reproduce exact input spatial shape."""
    with torch.no_grad():
        mu_hat, _ = ae(mu)
    assert mu_hat.shape == mu.shape, mu_hat.shape


# ---------------------------------------------------------------------------
# Patch-size parameterisation: patch_size=2 (120//2=60, 64//2=32)
# ---------------------------------------------------------------------------


def test_roundtrip_shape_patch2():
    ae2 = TriplaneAE(
        in_channels=4,
        emb_dim=32,
        n_layers=1,
        n_heads=4,
        out_channels=8,
        decoder_hidden=16,
        latent_shape=LATENT_SHAPE,
        patch_size=2,
    ).eval()
    torch.manual_seed(1)
    x = torch.randn(1, 4, *LATENT_SHAPE)
    with torch.no_grad():
        out, planes = ae2(x)
    assert out.shape == x.shape, out.shape
    # Verify intermediate plane shapes for patch_size=2
    assert planes["z_xy"].shape == (1, 8, 60, 60), planes["z_xy"].shape
    assert planes["z_yz"].shape == (1, 8, 60, 32), planes["z_yz"].shape
    assert planes["z_xz"].shape == (1, 8, 60, 32), planes["z_xz"].shape


# ---------------------------------------------------------------------------
# Forward/backward: no NaN, gradients flow to all trainable params
# ---------------------------------------------------------------------------


def test_forward_no_nan(ae, mu):
    with torch.no_grad():
        mu_hat, _ = ae(mu)
    assert not torch.isnan(mu_hat).any(), "NaN in mu_hat"
    assert not torch.isinf(mu_hat).any(), "Inf in mu_hat"


def test_backward_no_nan():
    """Train-mode backward pass: no NaN outputs or gradients."""
    torch.manual_seed(7)
    model = TriplaneAE(
        in_channels=4,
        emb_dim=32,
        n_layers=1,
        n_heads=4,
        out_channels=8,
        decoder_hidden=16,
        latent_shape=LATENT_SHAPE,
        patch_size=PATCH_SIZE,
    )
    x = torch.randn(1, 4, *LATENT_SHAPE)
    mu_hat, _ = model(x)
    loss = mu_hat.sum()
    loss.backward()
    assert not torch.isnan(mu_hat).any()
    bad = [
        n
        for n, p in model.named_parameters()
        if p.grad is not None and torch.isnan(p.grad).any()
    ]
    assert not bad, f"NaN gradient in: {bad}"


def test_all_params_have_grad():
    """Every trainable parameter must receive a gradient."""
    torch.manual_seed(8)
    model = TriplaneAE(
        in_channels=4,
        emb_dim=32,
        n_layers=1,
        n_heads=4,
        out_channels=8,
        decoder_hidden=16,
        latent_shape=LATENT_SHAPE,
        patch_size=PATCH_SIZE,
    )
    x = torch.randn(1, 4, *LATENT_SHAPE)
    mu_hat, _ = model(x)
    mu_hat.sum().backward()
    missing = [
        n for n, p in model.named_parameters() if p.requires_grad and p.grad is None
    ]
    assert not missing, f"No gradient for: {missing}"


# ---------------------------------------------------------------------------
# Inference memory: <2 GiB peak on GPU at emb_dim=256, patch_size=4
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_inference_memory_patchified():
    """Peak GPU memory for one forward pass must be below 2 GiB."""
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    model = (
        TriplaneAE(
            in_channels=4,
            emb_dim=256,
            n_layers=4,
            n_heads=8,
            out_channels=8,
            decoder_hidden=32,
            latent_shape=(120, 120, 64),
            patch_size=4,
        )
        .eval()
        .to(device)
    )
    x = torch.randn(1, 4, 120, 120, 64, device=device)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        _ = model(x)
    torch.cuda.synchronize()
    peak_gib = torch.cuda.max_memory_allocated(device) / 1024**3
    assert peak_gib < 2.0, f"Peak memory {peak_gib:.2f} GiB exceeds 2 GiB ceiling"
