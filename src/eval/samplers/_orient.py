"""Orientation helper for saved prediction volumes.

Models we train are encoded through ``Orientationd(axcodes="RAS")``
(``src/baselines/report2ct_image_encoder.py`` / ``scripts/precompute_wan_image_embeddings.py``),
so their decoders emit **RAS** content. The eval ground truth, however, is written straight from
the raw CT-RATE NIfTI (``src/eval/ct_rate_cases.py``) which is **LPS**, and the eval readers
(``evaluate_clip.py`` / FID / ``_fvd_ctclip.py``) compare **raw voxel arrays** — they ignore
direction metadata. So a saved prediction must carry LPS *content* to line up with the GT and the
real CT-RATE data.

LPS→RAS is exactly a flip of the two in-plane axes (X, Y): verified ``RAS == LPS[::-1, ::-1, :]``
on the source NIfTI, and confirmed anatomically (the spine — a posterior landmark — sits at +Y in
the LPS GT but −Y in the RAS predictions). This helper reverses that flip. It operates purely on
the voxel array; it does **not** read or trust any SimpleITK direction/origin metadata (which our
save path leaves at identity and is therefore not meaningful here).
"""

from __future__ import annotations

import numpy as np


def ras_to_lps(arr_zyx: np.ndarray) -> np.ndarray:
    """Flip the two in-plane axes of a ``(Z, Y, X)`` array: RAS content → LPS content.

    Args:
        arr_zyx: voxel array in SimpleITK order ``(Z, Y, X)`` (the array handed to
            ``sitk.GetImageFromArray``), holding RAS-oriented content.

    Returns:
        A contiguous copy with the X and Y axes reversed — ``(Z, Y, X)`` LPS content.
        Involutive: ``ras_to_lps(ras_to_lps(a))`` equals ``a``.
    """
    return arr_zyx[:, ::-1, ::-1].copy()
