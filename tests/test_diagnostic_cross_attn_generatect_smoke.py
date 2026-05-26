"""Cross-attention hook smoke on GenerateCT (Day 5 deliverable, slip-eligible to Phase B Day 1)."""

from __future__ import annotations

import pytest


@pytest.mark.skip(
    reason="Cross-attn GenerateCT smoke lands Day 5 (or slips to 6/1 per cut-order)."
)
def test_cross_attn_hook_registers_and_dumps_one_png() -> None:
    raise NotImplementedError
