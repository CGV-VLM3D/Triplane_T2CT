"""Tests for src/metrics: shape, identity, known-value, batch-consistency."""

from __future__ import annotations

import math

import pytest
import torch

from src.metrics.image_metrics import (
    image_lpips_2d_avg,
    image_psnr_3d,
    image_ssim_2d_avg,
    image_ssim_3d,
)
from src.metrics.latent_metrics import (
    compute_latent_data_range,
    latent_cosine_similarity,
    latent_l1,
    latent_mse,
    latent_psnr,
)

# Small tensors so tests run quickly on CPU.
B = 2
LATENT = (B, 4, 16, 16, 8)
IMAGE = (B, 1, 32, 32, 16)  # spatial dims >= win_size=7
WIN = 7


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def lat_rand():
    torch.manual_seed(0)
    return torch.randn(*LATENT)


@pytest.fixture(scope="module")
def img_rand():
    torch.manual_seed(1)
    return torch.randn(*IMAGE) * 500.0  # HU-like range


# ---------------------------------------------------------------------------
# latent_l1
# ---------------------------------------------------------------------------


def test_latent_l1_shape(lat_rand):
    out = latent_l1(lat_rand, lat_rand + 0.1)
    assert out.shape == (B,), out.shape


def test_latent_l1_identity(lat_rand):
    out = latent_l1(lat_rand, lat_rand)
    assert torch.allclose(out, torch.zeros(B)), out


def test_latent_l1_batch_consistency(lat_rand):
    a, b = lat_rand[:1], lat_rand[1:]
    full = latent_l1(lat_rand, lat_rand + 0.3)
    s0 = latent_l1(a, a + 0.3)
    s1 = latent_l1(b, b + 0.3)
    assert torch.allclose(full[0], s0[0]) and torch.allclose(full[1], s1[0])


# ---------------------------------------------------------------------------
# latent_mse
# ---------------------------------------------------------------------------


def test_latent_mse_shape(lat_rand):
    out = latent_mse(lat_rand, lat_rand + 0.1)
    assert out.shape == (B,), out.shape


def test_latent_mse_identity(lat_rand):
    out = latent_mse(lat_rand, lat_rand)
    assert torch.allclose(out, torch.zeros(B)), out


def test_latent_mse_batch_consistency(lat_rand):
    a, b = lat_rand[:1], lat_rand[1:]
    full = latent_mse(lat_rand, lat_rand + 0.5)
    s0 = latent_mse(a, a + 0.5)
    s1 = latent_mse(b, b + 0.5)
    assert torch.allclose(full[0], s0[0]) and torch.allclose(full[1], s1[0])


# ---------------------------------------------------------------------------
# latent_psnr
# ---------------------------------------------------------------------------


def test_latent_psnr_shape(lat_rand):
    out = latent_psnr(lat_rand, lat_rand + 0.1, data_range=2.0)
    assert out.shape == (B,), out.shape


def test_latent_psnr_identity_large(lat_rand):
    # eps-clamp ensures finite but very large value (>= 80 dB)
    out = latent_psnr(lat_rand, lat_rand, data_range=2.0)
    assert (out >= 80.0).all(), out


def test_latent_psnr_known_case():
    # offset=0.1, data_range=2.0 → MSE=0.01 → PSNR=10*log10(4/0.01)≈26.02
    x = torch.zeros(B, 1, 8, 8, 4)
    y = x + 0.1
    out = latent_psnr(y, x, data_range=2.0)
    expected = 10.0 * math.log10(4.0 / 0.01)
    assert out.shape == (B,)
    assert torch.allclose(out, torch.full((B,), expected), atol=1e-3), out


def test_latent_psnr_batch_consistency(lat_rand):
    a, b = lat_rand[:1], lat_rand[1:]
    offset = 0.2
    full = latent_psnr(lat_rand, lat_rand + offset, data_range=3.0)
    s0 = latent_psnr(a, a + offset, data_range=3.0)
    s1 = latent_psnr(b, b + offset, data_range=3.0)
    assert torch.allclose(full[0], s0[0]) and torch.allclose(full[1], s1[0])


# ---------------------------------------------------------------------------
# latent_cosine_similarity
# ---------------------------------------------------------------------------


def test_latent_cosine_shape(lat_rand):
    out = latent_cosine_similarity(lat_rand, lat_rand + 0.01)
    assert out.shape == (B,), out.shape


def test_latent_cosine_identity(lat_rand):
    out = latent_cosine_similarity(lat_rand, lat_rand)
    assert torch.allclose(out, torch.ones(B), atol=1e-5), out


def test_latent_cosine_batch_consistency(lat_rand):
    a, b = lat_rand[:1], lat_rand[1:]
    torch.manual_seed(99)
    noise = torch.randn_like(lat_rand) * 0.1
    full = latent_cosine_similarity(lat_rand, lat_rand + noise)
    s0 = latent_cosine_similarity(a, a + noise[:1])
    s1 = latent_cosine_similarity(b, b + noise[1:])
    assert torch.allclose(full[0], s0[0]) and torch.allclose(full[1], s1[0])


# ---------------------------------------------------------------------------
# compute_latent_data_range
# ---------------------------------------------------------------------------


def test_latent_data_range_scalar(lat_rand):
    out = compute_latent_data_range(lat_rand)
    assert out.shape == torch.Size([]), out.shape
    assert out > 0.0


# ---------------------------------------------------------------------------
# image_psnr_3d
# ---------------------------------------------------------------------------


def test_image_psnr_3d_shape(img_rand):
    out = image_psnr_3d(img_rand, img_rand + 1.0)
    assert out.shape == (B,), out.shape


def test_image_psnr_3d_identity(img_rand):
    out = image_psnr_3d(img_rand, img_rand)
    # monai PSNRMetric returns inf for identical inputs
    assert torch.isinf(out).all(), out


def test_image_psnr_3d_known_case():
    # offset=0.1, data_range=2.0 → PSNR≈26.02
    x = torch.zeros(B, 1, 16, 16, 8)
    y = x + 0.1
    out = image_psnr_3d(y, x, data_range=2.0)
    expected = 10.0 * math.log10(4.0 / 0.01)
    assert out.shape == (B,)
    assert torch.allclose(out, torch.full((B,), expected), atol=1e-3), out


def test_image_psnr_3d_batch_consistency(img_rand):
    a, b = img_rand[:1], img_rand[1:]
    full = image_psnr_3d(img_rand, img_rand + 5.0)
    s0 = image_psnr_3d(a, a + 5.0)
    s1 = image_psnr_3d(b, b + 5.0)
    assert torch.allclose(full[0], s0[0]) and torch.allclose(full[1], s1[0])


# ---------------------------------------------------------------------------
# image_ssim_3d
# ---------------------------------------------------------------------------


def test_image_ssim_3d_shape(img_rand):
    out = image_ssim_3d(img_rand, img_rand + 1.0, win_size=WIN)
    assert out.shape == (B,), out.shape


def test_image_ssim_3d_identity(img_rand):
    out = image_ssim_3d(img_rand, img_rand, win_size=WIN)
    assert torch.allclose(out, torch.ones(B), atol=1e-4), out


def test_image_ssim_3d_batch_consistency(img_rand):
    a, b = img_rand[:1], img_rand[1:]
    torch.manual_seed(7)
    noise = torch.randn_like(img_rand) * 10.0
    full = image_ssim_3d(img_rand, img_rand + noise, win_size=WIN)
    s0 = image_ssim_3d(a, a + noise[:1], win_size=WIN)
    s1 = image_ssim_3d(b, b + noise[1:], win_size=WIN)
    assert torch.allclose(full[0], s0[0], atol=1e-5)
    assert torch.allclose(full[1], s1[0], atol=1e-5)


# ---------------------------------------------------------------------------
# image_ssim_2d_avg
# ---------------------------------------------------------------------------


def test_image_ssim_2d_avg_shape(img_rand):
    out = image_ssim_2d_avg(img_rand, img_rand + 1.0, win_size=WIN)
    assert out.shape == (B,), out.shape


def test_image_ssim_2d_avg_identity(img_rand):
    out = image_ssim_2d_avg(img_rand, img_rand, win_size=WIN)
    assert torch.allclose(out, torch.ones(B), atol=1e-4), out


def test_image_ssim_2d_avg_deterministic(img_rand):
    out1 = image_ssim_2d_avg(img_rand, img_rand + 5.0, seed=42, win_size=WIN)
    out2 = image_ssim_2d_avg(img_rand, img_rand + 5.0, seed=42, win_size=WIN)
    assert torch.allclose(out1, out2), (out1, out2)


def test_image_ssim_2d_avg_batch_consistency(img_rand):
    a, b = img_rand[:1], img_rand[1:]
    torch.manual_seed(7)
    noise = torch.randn_like(img_rand) * 10.0
    full = image_ssim_2d_avg(img_rand, img_rand + noise, seed=7, win_size=WIN)
    s0 = image_ssim_2d_avg(a, a + noise[:1], seed=7, win_size=WIN)
    s1 = image_ssim_2d_avg(b, b + noise[1:], seed=7, win_size=WIN)
    assert torch.allclose(full[0], s0[0], atol=1e-5)
    assert torch.allclose(full[1], s1[0], atol=1e-5)


# ---------------------------------------------------------------------------
# image_lpips_2d_avg
# ---------------------------------------------------------------------------


def test_image_lpips_2d_avg_shape(img_rand):
    out = image_lpips_2d_avg(img_rand, img_rand + 10.0)
    assert out.shape == (B,), out.shape


def test_image_lpips_2d_avg_identity(img_rand):
    out = image_lpips_2d_avg(img_rand, img_rand)
    assert (out < 0.05).all(), f"LPIPS identity not near 0: {out}"


def test_image_lpips_2d_avg_deterministic(img_rand):
    out1 = image_lpips_2d_avg(img_rand, img_rand + 50.0, seed=0)
    out2 = image_lpips_2d_avg(img_rand, img_rand + 50.0, seed=0)
    assert torch.allclose(out1, out2), (out1, out2)


def test_image_lpips_2d_avg_batch_consistency(img_rand):
    a, b = img_rand[:1], img_rand[1:]
    torch.manual_seed(5)
    noise = torch.randn_like(img_rand) * 50.0
    full = image_lpips_2d_avg(img_rand, img_rand + noise, seed=3)
    s0 = image_lpips_2d_avg(a, a + noise[:1], seed=3)
    s1 = image_lpips_2d_avg(b, b + noise[1:], seed=3)
    assert torch.allclose(full[0], s0[0], atol=1e-5)
    assert torch.allclose(full[1], s1[0], atol=1e-5)
