"""GenerateCT LightningModule adapter smoke (lands Day 3)."""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="GenerateCT adapter + ckpt download lands Day 3 (5/28).")
def test_generatect_adapter_text_to_volume_one_sample() -> None:
    """Adapter loads ctvit/transformer/superres weights and produces a non-trivial volume."""
    raise NotImplementedError
