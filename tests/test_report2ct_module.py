"""Report2CT skeleton tests (Day 4 — 1-step correctness gate).

Acceptance: the DiffusionModelUNetMaisi instantiates from the submodule's
config_maisi_2560.json (no architecture duplication) and a 1-step forward
on a tiny latent returns a tensor of the same shape.

We do NOT load weights — Report2CT does not release them. Full training is
user-driven in Phase B ([[report2ct-training-is-user-owned]]). The Tier-0
overfit smoke (test_overfit_one_batch) lands when the LightningModule's
training_step is wired in Phase B.
"""

from __future__ import annotations

import pytest
import torch

from src.baselines.report2ct_adapter import (
    CONTEXT_DIM,
    LATENT_SHAPE,
    MODEL_DEF_CONFIG,
    Report2CTAdapter,
    build_unet,
    forward_one_step,
)


@pytest.mark.skipif(
    not MODEL_DEF_CONFIG.is_file(),
    reason=f"Report2CT submodule config missing at {MODEL_DEF_CONFIG}",
)
def test_build_unet_from_submodule_config() -> None:
    unet = build_unet()
    n_params_M = sum(p.numel() for p in unet.parameters()) / 1e6
    assert 100 < n_params_M < 500, f"Unexpected UNet size: {n_params_M:.1f}M"


@pytest.mark.skipif(
    not MODEL_DEF_CONFIG.is_file(),
    reason=f"Report2CT submodule config missing at {MODEL_DEF_CONFIG}",
)
def test_forward_one_batch() -> None:
    """1-step forward returns tensor with the same shape as the input latent.

    Smoke is on a tiny spatial size (16, 16, 8) to fit CPU within ~30s.
    Full latent (120, 120, 64) is reserved for the 6/1 wall-clock gate on GPU.
    """
    unet = build_unet().eval()
    latent = torch.randn(1, LATENT_SHAPE[0], 16, 16, 8)
    context = torch.randn(1, 2, CONTEXT_DIM)
    spacing = torch.tensor([[1.0, 1.0, 1.5]])
    with torch.no_grad():
        out = forward_one_step(
            unet, latent, context, spacing, timestep=500, class_label=1
        )
    assert out.shape == latent.shape, f"shape mismatch: {out.shape} vs {latent.shape}"


@pytest.mark.skip(
    reason="Tier-0 overfit gate is Phase B 6/1 [A] deliverable (needs MAISI VAE + text-precompute)."
)
def test_overfit_one_batch() -> None:
    raise NotImplementedError


@pytest.mark.skipif(
    not MODEL_DEF_CONFIG.is_file(),
    reason=f"Report2CT submodule config missing at {MODEL_DEF_CONFIG}",
)
def test_adapter_lightning_module_forward() -> None:
    adapter = Report2CTAdapter(device_str="cpu")
    latent = torch.randn(1, LATENT_SHAPE[0], 16, 16, 8)
    context = torch.randn(1, 2, CONTEXT_DIM)
    spacing = torch.tensor([[1.0, 1.0, 1.5]])
    with torch.no_grad():
        out = adapter(latent, context, spacing)
    assert out.shape == latent.shape
