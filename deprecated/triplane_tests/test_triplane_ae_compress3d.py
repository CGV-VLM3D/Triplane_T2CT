"""Shape, gradient and memory checks for TriplaneAECompress3D.

This is the paper-faithful Compress3D variant:
    encoder: TriplaneEncoderCompress3D (feature lift + per-plane 2D ResBlock
                                        stages with 4× downsample + 3D-aware
                                        cube cross-attention)
    decoder: TriplaneDecoderCompress3D (per-plane 2D ResBlock stages with
                                        4× upsample + 3D broadcast-fuse)
"""

from __future__ import annotations

import pytest
import torch

from src.models.triplane_ae_compress3d import TriplaneAECompress3D
from src.models.triplane_encoder_compress3d import TriplaneEncoderCompress3D

# Asymmetric, divisible-by-4-and-2 latent shape. (H=16, W=12, D=8)
LATENT_SHAPE = (16, 12, 8)
H, W, D = LATENT_SHAPE
# Triplane low-res after 2 downsamples (rH=4, rW=3, rD=2).
rH, rW, rD = H // 4, W // 4, D // 4
B = 1
PLANE_PROG = (8, 16, 24)  # tiny channel progression for fast tests
LC = 6  # latent_channels


def _make_ae(**kwargs):
    """TriplaneAECompress3D with tiny defaults for tests."""
    defaults = dict(
        in_channels=4,
        latent_shape=LATENT_SHAPE,
        feature_lift_channels=8,
        plane_channels_progression=PLANE_PROG,
        enc_n_blocks_per_stage=(1, 1, 1),
        dec_n_blocks_per_stage=(1, 1, 1),
        n_triplane_downsamples=2,
        volume_downsample=2,
        volume_kv_channels=8,
        cross_attn_heads=2,
        cross_attn_d_kv=4,
        latent_channels=LC,
    )
    defaults.update(kwargs)
    return TriplaneAECompress3D(**defaults)


@pytest.fixture(scope="module")
def ae():
    return _make_ae().eval()


@pytest.fixture(scope="module")
def encoder():
    return TriplaneEncoderCompress3D(
        in_channels=4,
        latent_shape=LATENT_SHAPE,
        feature_lift_channels=8,
        plane_channels_progression=PLANE_PROG,
        n_blocks_per_stage=(1, 1, 1),
        n_triplane_downsamples=2,
        volume_downsample=2,
        volume_kv_channels=8,
        cross_attn_heads=2,
        cross_attn_d_kv=4,
        latent_channels=LC,
    ).eval()


@pytest.fixture(scope="module")
def mu():
    torch.manual_seed(0)
    return torch.randn(B, 4, *LATENT_SHAPE)


# ---------------------------------------------------------------------------
# Encoder output shapes (low-res triplane)
# ---------------------------------------------------------------------------


def test_encoder_output_keys(encoder, mu):
    with torch.no_grad():
        out = encoder(mu)
    assert set(out["mu"].keys()) == {"xy", "yz", "xz"}
    assert set(out["logvar"].keys()) == {"xy", "yz", "xz"}


def test_encoder_xy_shape_is_low_res(encoder, mu):
    with torch.no_grad():
        out = encoder(mu)
    assert out["mu"]["xy"].shape == (B, LC, rH, rW), out["mu"]["xy"].shape


def test_encoder_yz_shape_is_low_res(encoder, mu):
    with torch.no_grad():
        out = encoder(mu)
    assert out["mu"]["yz"].shape == (B, LC, rH, rD), out["mu"]["yz"].shape


def test_encoder_xz_shape_is_low_res(encoder, mu):
    with torch.no_grad():
        out = encoder(mu)
    assert out["mu"]["xz"].shape == (B, LC, rW, rD), out["mu"]["xz"].shape


def test_encoder_logvar_matches_mu(encoder, mu):
    with torch.no_grad():
        out = encoder(mu)
    for k in ("xy", "yz", "xz"):
        assert out["mu"][k].shape == out["logvar"][k].shape


# ---------------------------------------------------------------------------
# AE roundtrip
# ---------------------------------------------------------------------------


def test_ae_roundtrip_shape(ae, mu):
    with torch.no_grad():
        out = ae(mu)
    assert out["mu_hat"].shape == mu.shape, out["mu_hat"].shape


def test_ae_forward_no_nan(ae, mu):
    with torch.no_grad():
        out = ae(mu)
    assert not torch.isnan(out["mu_hat"]).any()
    assert not torch.isinf(out["mu_hat"]).any()


def test_ae_kl_loss_is_finite_scalar(ae, mu):
    with torch.no_grad():
        out = ae(mu)
    kl = out["kl_loss"]
    assert kl.shape == ()
    assert torch.isfinite(kl)


def test_ae_legacy_tuple_unpack(ae, mu):
    """train.py uses ``mu_hat, _ = model(mu)`` — must still work."""
    with torch.no_grad():
        mu_hat, aux = ae(mu)
    assert mu_hat.shape == mu.shape
    assert "kl_loss" in aux


# ---------------------------------------------------------------------------
# Backward / grad
# ---------------------------------------------------------------------------


def test_backward_no_nan_grad():
    torch.manual_seed(7)
    model = _make_ae()
    x = torch.randn(B, 4, *LATENT_SHAPE)
    out = model(x)
    (out["mu_hat"].sum() + out["kl_loss"]).backward()
    bad = [
        n
        for n, p in model.named_parameters()
        if p.grad is not None and torch.isnan(p.grad).any()
    ]
    assert not bad, f"NaN gradient in: {bad}"


def test_all_params_have_grad():
    torch.manual_seed(8)
    model = _make_ae()
    x = torch.randn(B, 4, *LATENT_SHAPE)
    out = model(x)
    (out["mu_hat"].sum() + out["kl_loss"]).backward()
    missing = [
        n for n, p in model.named_parameters() if p.requires_grad and p.grad is None
    ]
    assert not missing, f"No gradient for: {missing}"


# ---------------------------------------------------------------------------
# Cube cross-attention sanity
# ---------------------------------------------------------------------------


def test_cube_m_equals_2_for_default_settings():
    """With n_triplane_downsamples=2 and volume_downsample=2, m=2 on every axis
    (true 3D cube attention, not column-only). Verifies the m_a derivation."""
    enc = TriplaneEncoderCompress3D(
        in_channels=4,
        latent_shape=LATENT_SHAPE,
        feature_lift_channels=8,
        plane_channels_progression=PLANE_PROG,
        n_blocks_per_stage=(1, 1, 1),
        n_triplane_downsamples=2,
        volume_downsample=2,
        volume_kv_channels=8,
        cross_attn_heads=2,
        cross_attn_d_kv=4,
        latent_channels=LC,
    )
    ca = enc.cross_attn
    # H=16, W=12, D=8 → V^n_d at (8, 6, 4); triplane low-res (4, 3, 2).
    assert ca.m_xy_H == 2 and ca.m_xy_W == 2
    assert ca.m_yz_H == 2 and ca.m_yz_D == 2
    assert ca.m_xz_W == 2 and ca.m_xz_D == 2


def test_invalid_latent_shape_raises():
    with pytest.raises(ValueError):
        # 15 not divisible by 4 (triplane down) — should fail clearly.
        TriplaneEncoderCompress3D(
            in_channels=4,
            latent_shape=(15, 12, 8),
            feature_lift_channels=8,
            plane_channels_progression=PLANE_PROG,
            n_blocks_per_stage=(1, 1, 1),
        )


def test_n_downsamples_zero_keeps_latent_at_full_res():
    """Config C — hires: no triplane down, no volume down → latent same spatial
    size as input, deep K/V via feature lift, column-only attention (m=1)."""
    model = _make_ae(
        n_triplane_downsamples=0,
        volume_downsample=1,
    )
    x = torch.randn(B, 4, *LATENT_SHAPE)
    model.eval()
    with torch.no_grad():
        out = model(x)
    # Latent stays at full triplane resolution (no spatial downsample).
    assert out["mu_planes"]["xy"].shape == (B, LC, H, W)
    assert out["mu_planes"]["yz"].shape == (B, LC, H, D)
    assert out["mu_planes"]["xz"].shape == (B, LC, W, D)
    assert out["mu_hat"].shape == x.shape

    # Cube degenerates to column (m=1).
    ca = model.encoder.cross_attn
    assert ca.m_xy_H == 1 and ca.m_xy_W == 1
    assert ca.m_yz_H == 1 and ca.m_yz_D == 1
    assert ca.m_xz_W == 1 and ca.m_xz_D == 1


def test_n_downsamples_one_2x_compression():
    """Intermediate config: 1 triplane down (2× spatial), volume full-res →
    cube m=2 (real 3D context) at moderate compression."""
    model = _make_ae(
        n_triplane_downsamples=1,
        volume_downsample=1,
    )
    x = torch.randn(B, 4, *LATENT_SHAPE)
    model.eval()
    with torch.no_grad():
        out = model(x)
    assert out["mu_planes"]["xy"].shape == (B, LC, H // 2, W // 2)
    assert out["mu_hat"].shape == x.shape
    ca = model.encoder.cross_attn
    assert ca.m_xy_H == 2 and ca.m_xy_W == 2


def test_volume_downsample_exceeds_triplane_down_raises():
    """o > 2^n triggers cube m < 1 — should fail clearly at construction."""
    with pytest.raises(ValueError, match="cube m < 1"):
        TriplaneEncoderCompress3D(
            in_channels=4,
            latent_shape=LATENT_SHAPE,
            feature_lift_channels=8,
            plane_channels_progression=PLANE_PROG,
            n_blocks_per_stage=(1, 1, 1),
            n_triplane_downsamples=0,
            volume_downsample=2,
        )


def test_backward_no_nan_n_downsamples_zero():
    """Grad health check for the hires config (column attn + deep K/V)."""
    torch.manual_seed(11)
    model = _make_ae(n_triplane_downsamples=0, volume_downsample=1)
    x = torch.randn(B, 4, *LATENT_SHAPE)
    out = model(x)
    (out["mu_hat"].sum() + out["kl_loss"]).backward()
    bad = [
        n
        for n, p in model.named_parameters()
        if p.grad is not None and torch.isnan(p.grad).any()
    ]
    assert not bad, f"NaN gradient in: {bad}"
    missing = [
        n for n, p in model.named_parameters() if p.requires_grad and p.grad is None
    ]
    assert not missing, f"No gradient for: {missing}"


# ---------------------------------------------------------------------------
# GPU memory guard at production 1mm size
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_inference_memory_1mm():
    """One forward at full 1mm latent (120,120,64) with paper-default widths
    must stay under 12 GiB on a 4090/A6000. The model is heavier than the
    simplified TriplaneAE because of the deep triplane stages."""
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    model = (
        TriplaneAECompress3D(
            in_channels=4,
            latent_shape=(120, 120, 64),
            feature_lift_channels=32,
            plane_channels_progression=(32, 64, 128),
            enc_n_blocks_per_stage=(2, 2, 4),
            dec_n_blocks_per_stage=(3, 3, 5),
            n_triplane_downsamples=2,
            volume_downsample=2,
            cross_attn_heads=4,
            cross_attn_d_kv=32,
            latent_channels=32,
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
    assert peak_gib < 12.0, f"Peak memory {peak_gib:.2f} GiB exceeds 12 GiB"
