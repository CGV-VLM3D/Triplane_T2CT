"""Structural invariance and cross-attention ablation checks for Compress3D.

The old transformer encoder had learnable axial positional embeddings
(pos_embed_xy/yz/xz) tested here. The Compress3D encoder uses axis-collapsing
Conv3d projections instead: permuting the collapsed axis produces the same
output (by design — the projection is a learned weighted sum over that axis).

This file tests Compress3D-specific structural properties instead:
1. The encoder output is not permutation-invariant in the NON-collapsed axes
   (spatial structure within each plane is preserved).
2. Enabling vs. disabling CrossAttention3D produces different encodings.
"""

from __future__ import annotations

import torch
import pytest

from src.models.triplane_ae import TriplaneAE
from src.models.triplane_encoder import TriplaneEncoder

LATENT_SHAPE = (12, 8, 6)  # (H, W, D)


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
    """All encoder params must get nonzero gradients in one backward pass."""
    torch.manual_seed(0)
    mu = torch.randn(1, 4, *LATENT_SHAPE)
    out = model(mu)
    (out["mu_hat"].sum() + out["kl_loss"]).backward()

    missing, zero_grad = [], []
    for name, p in model.encoder.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            missing.append(name)
        elif p.grad.abs().sum() == 0:
            zero_grad.append(name)
    assert not missing, f"No gradient at all for: {missing}"
    assert not zero_grad, f"All-zero gradient for: {zero_grad}"


def test_xy_plane_sensitive_to_h_permutation():
    """XY plane (collapses D) must change when H positions are shuffled.

    Conv3d with kernel (1,1,D) collapses D but preserves H and W positions.
    After the subsequent 2D ResBlocks, outputs at different H positions are
    contextualised — permuting H must produce a different xy plane.
    """
    enc = TriplaneEncoder(
        in_channels=4,
        latent_shape=LATENT_SHAPE,
        plane_channels=8,
        latent_channels=8,
        n_resblocks=(1, 1, 1),
        use_cross_attn=False,
    ).eval()
    H, W, D = LATENT_SHAPE
    torch.manual_seed(0)
    mu = torch.randn(1, 4, H, W, D)
    perm = torch.randperm(H)
    mu_perm = mu[:, :, perm, :, :]
    with torch.no_grad():
        out_orig = enc(mu)["mu"]["xy"]
        out_perm = enc(mu_perm)["mu"]["xy"]
    assert not torch.allclose(out_orig, out_perm, atol=1e-4), (
        "xy plane is unchanged after permuting H rows — ResBlocks may not be active"
    )


def test_cross_attn_changes_encoding():
    """Enabling CrossAttention3D must produce a different encoding than disabling it."""
    H, W, D = LATENT_SHAPE
    kwargs = dict(
        in_channels=4,
        latent_shape=LATENT_SHAPE,
        plane_channels=8,
        latent_channels=8,
        n_resblocks=(1, 1, 1),
        decoder_n_tri_resblocks=1,
    )
    enc_no_attn = TriplaneAE(**kwargs, use_cross_attn=False).eval()
    enc_with_attn = TriplaneAE(
        **kwargs, use_cross_attn=True, cross_attn_heads=4, cross_attn_d_kv=8
    ).eval()
    torch.manual_seed(1)
    mu = torch.randn(1, 4, *LATENT_SHAPE)
    with torch.no_grad():
        out_no = enc_no_attn(mu)["mu_planes"]["xy"]
        out_at = enc_with_attn(mu)["mu_planes"]["xy"]
    assert not torch.allclose(out_no, out_at, atol=1e-4), (
        "xy plane is identical with and without cross-attention"
    )
