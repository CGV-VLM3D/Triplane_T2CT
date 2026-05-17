"""Tests for TriVAE_D3T: shapes, backward, gradient flow, memory."""

from __future__ import annotations

import pytest
import torch

from src.models.trivae_d3t import TriVAE_D3T

LATENT_SHAPE = (120, 120, 64)
# patch_size=4 -> patched dims (30, 30, 16); window (5,5,4) divides cleanly
PATCH_SIZE = 4
# patched dims for patch_size=4
Hp, Wp, Dp = 30, 30, 16
B = 1


@pytest.fixture(scope="module")
def small_model():
    """Tiny model for fast CPU tests."""
    return TriVAE_D3T(
        in_channels=4,
        emb_dim=64,
        patch_size=PATCH_SIZE,
        n_swin_enc_layers=2,
        swin_window_size=(5, 5, 4),
        swin_num_heads=4,
        n_fpsi_layers=1,
        fpsi_num_heads=4,
        out_channels=8,
        n_swin_dec_layers=2,
        latent_shape=LATENT_SHAPE,
    ).eval()


@pytest.fixture(scope="module")
def x():
    torch.manual_seed(0)
    return torch.randn(B, 4, *LATENT_SHAPE)


# ---------------------------------------------------------------------------
# 1. Round-trip shape
# ---------------------------------------------------------------------------


def test_roundtrip_shape(small_model, x):
    with torch.no_grad():
        recon, triplane = small_model(x)
    assert recon.shape == (B, 4, *LATENT_SHAPE), recon.shape


def test_encode_plane_shapes(small_model, x):
    with torch.no_grad():
        triplane = small_model.encode(x)
    # XY: collapses Dp -> [B, out_ch, Hp, Wp]
    assert triplane["z_xy"].shape == (B, 8, Hp, Wp), triplane["z_xy"].shape
    # YZ: collapses Hp -> [B, out_ch, Wp, Dp]
    assert triplane["z_yz"].shape == (B, 8, Wp, Dp), triplane["z_yz"].shape
    # XZ: collapses Wp -> [B, out_ch, Hp, Dp]
    assert triplane["z_xz"].shape == (B, 8, Hp, Dp), triplane["z_xz"].shape


def test_decode_shape(small_model, x):
    with torch.no_grad():
        triplane = small_model.encode(x)
        recon = small_model.decode(triplane)
    assert recon.shape == (B, 4, *LATENT_SHAPE), recon.shape


# ---------------------------------------------------------------------------
# 2. Forward/backward: no NaN, gradients flow to all trainable params
# ---------------------------------------------------------------------------


def test_forward_no_nan(small_model, x):
    with torch.no_grad():
        recon, _ = small_model(x)
    assert not torch.isnan(recon).any(), "NaN in recon"
    assert not torch.isinf(recon).any(), "Inf in recon"


def test_backward_no_nan():
    torch.manual_seed(7)
    model = TriVAE_D3T(
        in_channels=4,
        emb_dim=32,
        patch_size=PATCH_SIZE,
        n_swin_enc_layers=2,
        swin_window_size=(5, 5, 4),
        swin_num_heads=4,
        n_fpsi_layers=1,
        fpsi_num_heads=4,
        out_channels=8,
        n_swin_dec_layers=2,
        latent_shape=LATENT_SHAPE,
    )
    x = torch.randn(1, 4, *LATENT_SHAPE)
    recon, _ = model(x)
    loss = (recon - x).abs().mean()
    loss.backward()

    assert not torch.isnan(recon).any(), "NaN in recon"
    bad_grads = [
        n
        for n, p in model.named_parameters()
        if p.grad is not None and torch.isnan(p.grad).any()
    ]
    assert not bad_grads, f"NaN gradient in: {bad_grads}"


def test_all_params_get_grad():
    """Every trainable parameter must receive a gradient on a backward pass."""
    torch.manual_seed(8)
    model = TriVAE_D3T(
        in_channels=4,
        emb_dim=32,
        patch_size=PATCH_SIZE,
        n_swin_enc_layers=2,
        swin_window_size=(5, 5, 4),
        swin_num_heads=4,
        n_fpsi_layers=1,
        fpsi_num_heads=4,
        out_channels=8,
        n_swin_dec_layers=2,
        latent_shape=LATENT_SHAPE,
    )
    x = torch.randn(1, 4, *LATENT_SHAPE)
    recon, _ = model(x)
    recon.sum().backward()

    missing = [
        n for n, p in model.named_parameters() if p.requires_grad and p.grad is None
    ]
    assert not missing, f"No gradient for: {missing}"


# ---------------------------------------------------------------------------
# 3. Patch sanity: patch_size=4 (default) and patch_size=2
# ---------------------------------------------------------------------------


def test_patch_size_4_shapes():
    """patch_size=4 on (120,120,64): patched=(30,30,16), window=(5,5,4) divides."""
    model = TriVAE_D3T(
        in_channels=4,
        emb_dim=32,
        patch_size=4,
        n_swin_enc_layers=2,
        swin_window_size=(5, 5, 4),
        swin_num_heads=4,
        n_fpsi_layers=1,
        fpsi_num_heads=4,
        out_channels=8,
        n_swin_dec_layers=2,
        latent_shape=(120, 120, 64),
    ).eval()
    torch.manual_seed(1)
    x = torch.randn(1, 4, 120, 120, 64)
    with torch.no_grad():
        recon, triplane = model(x)
    assert recon.shape == (1, 4, 120, 120, 64)
    assert triplane["z_xy"].shape == (1, 8, 30, 30)
    assert triplane["z_yz"].shape == (1, 8, 30, 16)
    assert triplane["z_xz"].shape == (1, 8, 30, 16)


def test_patch_size_2_shapes():
    """patch_size=2 on (120,120,64): patched=(60,60,32).
    Window (5,5,4) divides (60,60,32) cleanly (60%5=0, 32%4=0).
    """
    model = TriVAE_D3T(
        in_channels=4,
        emb_dim=32,
        patch_size=2,
        n_swin_enc_layers=2,
        swin_window_size=(5, 5, 4),
        swin_num_heads=4,
        n_fpsi_layers=1,
        fpsi_num_heads=4,
        out_channels=8,
        n_swin_dec_layers=2,
        latent_shape=(120, 120, 64),
    ).eval()
    torch.manual_seed(2)
    x = torch.randn(1, 4, 120, 120, 64)
    with torch.no_grad():
        recon, triplane = model(x)
    assert recon.shape == (1, 4, 120, 120, 64)
    assert triplane["z_xy"].shape == (1, 8, 60, 60)
    assert triplane["z_yz"].shape == (1, 8, 60, 32)
    assert triplane["z_xz"].shape == (1, 8, 60, 32)


# ---------------------------------------------------------------------------
# 4. Smoke memory (CUDA-only): peak memory < 5 GiB at default config, batch=1
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_inference_memory_d3t():
    """Peak GPU memory for one forward pass must be below 5 GiB."""
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    model = (
        TriVAE_D3T(
            in_channels=4,
            emb_dim=256,
            patch_size=4,
            n_swin_enc_layers=4,
            swin_window_size=(5, 5, 4),
            swin_num_heads=8,
            n_fpsi_layers=2,
            fpsi_num_heads=8,
            out_channels=8,
            n_swin_dec_layers=4,
            latent_shape=(120, 120, 64),
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
    assert peak_gib < 5.0, f"Peak memory {peak_gib:.2f} GiB exceeds 5 GiB ceiling"
