"""Shared tiny-synthetic-data helpers for the eval-analysis test suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk


def write_mha(
    path: Path,
    arr_zyx: np.ndarray,
    spacing_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Path:
    """Write a small int16 HU volume as ``.mha`` (sitk array convention: Z,Y,X)."""
    img = sitk.GetImageFromArray(arr_zyx.astype(np.int16))
    img.SetSpacing(spacing_xyz)
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(path))
    return path
