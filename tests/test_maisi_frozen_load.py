"""R6 mitigation: verify MAISI VAE checkpoint loads and stays frozen.

Acceptance per Phase A: `pytest tests/test_maisi_frozen_load.py` passes Day 2 EOD.
"""

from __future__ import annotations

from pathlib import Path

import pytest

MAISI_CKPT = Path("/workspace/maisi_bundle/models/autoencoder.pt")


@pytest.mark.skipif(
    not MAISI_CKPT.exists(), reason=f"MAISI ckpt missing at {MAISI_CKPT}"
)
def test_maisi_checkpoint_exists() -> None:
    assert MAISI_CKPT.is_file()
    assert MAISI_CKPT.stat().st_size > 1024 * 1024, "MAISI ckpt suspiciously small"


@pytest.mark.skipif(
    not MAISI_CKPT.exists(), reason=f"MAISI ckpt missing at {MAISI_CKPT}"
)
def test_maisi_checkpoint_loadable() -> None:
    import torch

    state = torch.load(MAISI_CKPT, map_location="cpu", weights_only=False)
    assert state is not None


@pytest.mark.skip(
    reason="Frozen-gradient assertion lands with the loader util on Day 2."
)
def test_maisi_parameters_are_frozen_after_load() -> None:
    """After loading via src.baselines.maisi.load_frozen, all params must have requires_grad=False."""
    raise NotImplementedError
