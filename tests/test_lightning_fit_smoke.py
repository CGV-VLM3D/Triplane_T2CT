"""End-of-Phase-A Lightning wiring proof (lands Day 5).

Note: lightning-hydra-template already provides tests/test_train.py which is a richer
fit smoke against actual data+model configs. This file remains as a Phase A acceptance
anchor for our minimal dummy-module smoke.
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(
    reason="Phase A Day 5 dummy fit smoke; template's test_train.py covers richer fit."
)
def test_trainer_fit_one_step_on_dummy_module() -> None:
    raise NotImplementedError
