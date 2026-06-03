"""GenerateCTSampler save-spacing test (WI-2).

Confirms the eval sampler stamps GenerateCT's truthful native spacing (0.75, 0.75, 1.5) mm —
not the official docker's distorting (1, 1, 1) — and that the docker-parity override still
works. CI-safe: the sampler builds its model lazily in `_init`, so we call `_save_mha`
directly on a tiny dummy volume (no GPU / no weights).
"""

from __future__ import annotations

import SimpleITK as sitk
import torch

from src.eval.samplers.generatect import GenerateCTSampler


def test_save_mha_stamps_truthful_native_spacing(tmp_path) -> None:
    """Default: truthful native (0.75, 0.75, 1.5) in ITK (x, y, z) — 1.5 on the slice axis."""
    sampler = GenerateCTSampler()
    vol = torch.zeros(1, 4, 8, 8)  # (1, D, H, W)
    out = tmp_path / "g.mha"
    sampler._save_mha(vol, out)
    sp = sitk.ReadImage(str(out)).GetSpacing()
    assert tuple(round(s, 4) for s in sp) == (0.75, 0.75, 1.5)


def test_docker_parity_spacing_override(tmp_path) -> None:
    """`final_spacing_mm=[1,1,1]` reproduces the published docker number."""
    sampler = GenerateCTSampler(final_spacing_mm=[1.0, 1.0, 1.0])
    vol = torch.zeros(1, 4, 8, 8)
    out = tmp_path / "g.mha"
    sampler._save_mha(vol, out)
    sp = sitk.ReadImage(str(out)).GetSpacing()
    assert tuple(round(s, 4) for s in sp) == (1.0, 1.0, 1.0)
