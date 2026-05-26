"""Report2CT adapter test placeholders.

Per [[report2ct-training-is-user-owned]]: assistant prepares LightningModule + Hydra config +
launcher script; user runs full training. Real impl lands in Phase B (6/1+) wrapping
`third_party/report2ct/src/maisi/scripts/diff_model_train_vlm3D_2560_multi_text.py`.
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Report2CT LightningModule adapter lands Phase B (6/1).")
def test_forward_one_batch() -> None:
    raise NotImplementedError


@pytest.mark.skip(reason="Report2CT Tier-0 overfit smoke lands Phase B (6/1).")
def test_overfit_one_batch() -> None:
    raise NotImplementedError
