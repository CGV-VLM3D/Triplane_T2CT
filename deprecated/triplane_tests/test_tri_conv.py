"""Tests for src/models/tri_conv.py (TriConv and TriConvResBlock)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.tri_conv import TriConv, TriConvResBlock

# Asymmetric triplane shapes matching the MAISI latent (H=120, W=120, D=64).
# We use small proxies for CPU speed in CI.
H, W, D = 12, 12, 6
B, C = 2, 8
OUT_C = 16


def _make_planes(c: int = C) -> dict[str, torch.Tensor]:
    torch.manual_seed(42)
    return {
        "xy": torch.randn(B, c, H, W),
        "yz": torch.randn(B, c, H, D),
        "xz": torch.randn(B, c, W, D),
    }


# ---------------------------------------------------------------------------
# Shape correctness
# ---------------------------------------------------------------------------

class TestTriConvShapes:
    def test_k1_shapes(self):
        conv = TriConv(C, OUT_C, kernel_size=1)
        out = conv(_make_planes())
        assert out["xy"].shape == (B, OUT_C, H, W)
        assert out["yz"].shape == (B, OUT_C, H, D)
        assert out["xz"].shape == (B, OUT_C, W, D)

    def test_k3_shapes(self):
        conv = TriConv(C, OUT_C, kernel_size=3)
        out = conv(_make_planes())
        assert out["xy"].shape == (B, OUT_C, H, W)
        assert out["yz"].shape == (B, OUT_C, H, D)
        assert out["xz"].shape == (B, OUT_C, W, D)

    def test_same_channels_shapes(self):
        conv = TriConv(C, C, kernel_size=3)
        out = conv(_make_planes())
        assert out["xy"].shape == (B, C, H, W)
        assert out["yz"].shape == (B, C, H, D)
        assert out["xz"].shape == (B, C, W, D)

    def test_asymmetric_grid(self):
        """YZ has different spatial dims than XY; ensure projections are correct."""
        conv = TriConv(C, OUT_C, kernel_size=3)
        planes = _make_planes()
        # YZ is (H, D) = (12, 6) — not square and different from XY (12, 12).
        assert planes["yz"].shape[-2:] != planes["xy"].shape[-2:]
        out = conv(planes)
        # Output shapes preserved.
        assert out["yz"].shape == (B, OUT_C, H, D)
        assert out["xy"].shape == (B, OUT_C, H, W)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

class TestTriConvGradients:
    def test_gradients_nonzero(self):
        conv = TriConv(C, OUT_C, kernel_size=3)
        planes = _make_planes()
        out = conv(planes)
        loss = sum(v.sum() for v in out.values())
        loss.backward()
        for name, p in conv.named_parameters():
            assert p.grad is not None, f"No grad: {name}"
            assert p.grad.abs().sum() > 0, f"Zero grad: {name}"

    def test_k1_gradients_nonzero(self):
        conv = TriConv(C, OUT_C, kernel_size=1)
        planes = _make_planes()
        out = conv(planes)
        loss = sum(v.sum() for v in out.values())
        loss.backward()
        for name, p in conv.named_parameters():
            assert p.grad is not None, f"No grad: {name}"
            assert p.grad.abs().sum() > 0, f"Zero grad: {name}"


# ---------------------------------------------------------------------------
# TriConvResBlock
# ---------------------------------------------------------------------------

class TestTriConvResBlock:
    def test_same_channels_shapes(self):
        blk = TriConvResBlock(C, C)
        out = blk(_make_planes())
        assert out["xy"].shape == (B, C, H, W)
        assert out["yz"].shape == (B, C, H, D)
        assert out["xz"].shape == (B, C, W, D)

    def test_channel_change_shapes(self):
        blk = TriConvResBlock(C, OUT_C)
        out = blk(_make_planes())
        assert out["xy"].shape == (B, OUT_C, H, W)
        assert out["yz"].shape == (B, OUT_C, H, D)
        assert out["xz"].shape == (B, OUT_C, W, D)

    def test_gradients_nonzero(self):
        blk = TriConvResBlock(C, OUT_C)
        planes = _make_planes()
        out = blk(planes)
        loss = sum(v.sum() for v in out.values())
        loss.backward()
        for name, p in blk.named_parameters():
            assert p.grad is not None, f"No grad: {name}"
            assert p.grad.abs().sum() > 0, f"Zero grad: {name}"

    def test_residual_identity_skip(self):
        """With same in/out channels the skip should be identity (no extra params)."""
        blk = TriConvResBlock(C, C)
        assert blk.skip is None

    def test_no_nan(self):
        blk = TriConvResBlock(C, C)
        out = blk(_make_planes())
        for k, v in out.items():
            assert not torch.isnan(v).any(), f"NaN in {k}"
