"""Shape and gradient checks for the Compress3D TriplaneAE and its components.

Architecture under test
-----------------------
TriplaneAE (triplane_ae.py)
  encoder: TriplaneEncoder (triplane_encoder.py)  -- axis-collapse + 2D ResBlocks + optional cross-attn
  decoder: TriplaneDecoderD (triplane_decoder_d.py) -- TriConv-based upsampler

The old transformer-based encoder/decoder (F_psi) lives in
triplane_encoder_F_psi.py / triplane_decoder_F_psi.py and is kept for
comparison but is NOT in MODEL_REGISTRY.  Tests for it can be added
to test_f_psi.py if needed.
"""

from __future__ import annotations

import pytest
import torch

from src.models.triplane_ae import TriplaneAE
from src.models.triplane_encoder import TriplaneEncoder

# Asymmetric shape makes it easy to catch axis-ordering bugs.
# (H=12, W=8, D=6)
LATENT_SHAPE = (12, 8, 6)
H, W, D = LATENT_SHAPE
B = 1
PC = 8  # plane_channels
LC = 8  # latent_channels


@pytest.fixture(scope="module")
def ae():
    return TriplaneAE(
        in_channels=4,
        latent_shape=LATENT_SHAPE,
        plane_channels=PC,
        latent_channels=LC,
        n_resblocks=(1, 1, 1),
        use_cross_attn=False,
        decoder_n_tri_resblocks=1,
    ).eval()


@pytest.fixture(scope="module")
def encoder():
    return TriplaneEncoder(
        in_channels=4,
        latent_shape=LATENT_SHAPE,
        plane_channels=PC,
        latent_channels=LC,
        n_resblocks=(1, 1, 1),
        use_cross_attn=False,
    ).eval()


@pytest.fixture(scope="module")
def mu():
    torch.manual_seed(0)
    return torch.randn(B, 4, *LATENT_SHAPE)


# ---------------------------------------------------------------------------
# Encoder output shapes
# Projection onto each axis collapses one spatial dim:
#   xy: collapse D  → [B, LC, H, W]
#   yz: collapse W  → [B, LC, H, D]
#   xz: collapse H  → [B, LC, W, D]
# ---------------------------------------------------------------------------


def test_encoder_planes_have_correct_keys(encoder, mu):
    with torch.no_grad():
        out = encoder(mu)
    assert set(out["mu"].keys()) == {"xy", "yz", "xz"}
    assert set(out["logvar"].keys()) == {"xy", "yz", "xz"}


def test_encoder_xy_shape(encoder, mu):
    with torch.no_grad():
        out = encoder(mu)
    assert out["mu"]["xy"].shape == (B, LC, H, W), out["mu"]["xy"].shape


def test_encoder_yz_shape(encoder, mu):
    with torch.no_grad():
        out = encoder(mu)
    assert out["mu"]["yz"].shape == (B, LC, H, D), out["mu"]["yz"].shape


def test_encoder_xz_shape(encoder, mu):
    with torch.no_grad():
        out = encoder(mu)
    assert out["mu"]["xz"].shape == (B, LC, W, D), out["mu"]["xz"].shape


# ---------------------------------------------------------------------------
# TriplaneAE roundtrip
# ---------------------------------------------------------------------------


def test_ae_roundtrip_shape(ae, mu):
    with torch.no_grad():
        out = ae(mu)
    assert out["mu_hat"].shape == mu.shape, out["mu_hat"].shape


def test_ae_forward_keys(ae, mu):
    with torch.no_grad():
        out = ae(mu)
    assert {"mu_hat", "mu_planes", "logvar_planes", "kl_loss"} <= set(out.keys())


def test_ae_legacy_tuple_unpack(ae, mu):
    """train.py uses `mu_hat, _ = model(mu)` — must still work."""
    with torch.no_grad():
        mu_hat, _ = ae(mu)
    assert mu_hat.shape == mu.shape


def test_ae_forward_no_nan(ae, mu):
    with torch.no_grad():
        out = ae(mu)
    assert not torch.isnan(out["mu_hat"]).any(), "NaN in mu_hat"
    assert not torch.isinf(out["mu_hat"]).any(), "Inf in mu_hat"


def test_ae_kl_loss_is_scalar(ae, mu):
    with torch.no_grad():
        out = ae(mu)
    kl = out["kl_loss"]
    assert kl.shape == (), f"kl_loss is not a scalar: {kl.shape}"
    assert torch.isfinite(kl), f"kl_loss is not finite: {kl}"


# ---------------------------------------------------------------------------
# Backward / gradient checks (train-mode)
# ---------------------------------------------------------------------------


def test_backward_no_nan():
    torch.manual_seed(7)
    model = TriplaneAE(
        in_channels=4,
        latent_shape=LATENT_SHAPE,
        plane_channels=PC,
        latent_channels=LC,
        n_resblocks=(1, 1, 1),
        use_cross_attn=False,
        decoder_n_tri_resblocks=1,
    )
    x = torch.randn(B, 4, *LATENT_SHAPE)
    out = model(x)
    loss = out["mu_hat"].sum() + out["kl_loss"]
    loss.backward()
    assert not torch.isnan(out["mu_hat"]).any()
    bad = [
        n
        for n, p in model.named_parameters()
        if p.grad is not None and torch.isnan(p.grad).any()
    ]
    assert not bad, f"NaN gradient in: {bad}"


def test_all_params_have_grad():
    torch.manual_seed(8)
    model = TriplaneAE(
        in_channels=4,
        latent_shape=LATENT_SHAPE,
        plane_channels=PC,
        latent_channels=LC,
        n_resblocks=(1, 1, 1),
        use_cross_attn=False,
        decoder_n_tri_resblocks=1,
    )
    x = torch.randn(B, 4, *LATENT_SHAPE)
    out = model(x)
    (out["mu_hat"].sum() + out["kl_loss"]).backward()
    missing = [
        n for n, p in model.named_parameters() if p.requires_grad and p.grad is None
    ]
    assert not missing, f"No gradient for: {missing}"


# ---------------------------------------------------------------------------
# Cross-attention variant: shapes stay the same
# ---------------------------------------------------------------------------


def test_ae_with_cross_attn_roundtrip_shape():
    model = TriplaneAE(
        in_channels=4,
        latent_shape=LATENT_SHAPE,
        plane_channels=PC,
        latent_channels=LC,
        n_resblocks=(1, 1, 1),
        use_cross_attn=True,
        cross_attn_heads=4,
        cross_attn_d_kv=8,
        decoder_n_tri_resblocks=1,
    ).eval()
    torch.manual_seed(2)
    x = torch.randn(B, 4, *LATENT_SHAPE)
    with torch.no_grad():
        out = model(x)
    assert out["mu_hat"].shape == x.shape


# ---------------------------------------------------------------------------
# GPU memory guard (skip when no GPU)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_inference_memory():
    """Peak GPU memory for one forward pass at standard 1mm size must be < 4 GiB."""
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    model = (
        TriplaneAE(
            in_channels=4,
            latent_shape=(120, 120, 64),
            plane_channels=16,
            latent_channels=16,
            n_resblocks=(2, 2, 4),
            use_cross_attn=True,
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
    assert peak_gib < 4.0, f"Peak memory {peak_gib:.2f} GiB exceeds 4 GiB"
