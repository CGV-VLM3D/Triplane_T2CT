"""Roundtrip tests for src/inference.py's per-model truthful NIfTI affine (WI-1).

inference.py builds a diagonal affine from each adapter's ``output_spacing`` (expressed in
the squeezed-array axis order) so the saved ``.nii.gz`` carries the model's real voxel
geometry instead of ``eye(4)``. These tests pin that mapping (catches axis / diag mistakes)
and confirm the adapters expose exactly the spacings inference.py will stamp. No GPU/weights.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest


def _save_with_spacing(arr: np.ndarray, spacing: tuple[float, ...], path) -> None:
    # Mirrors the affine construction in src/inference.py (array-axis-order diagonal).
    affine = np.diag([float(s) for s in spacing] + [1.0])
    nib.save(nib.Nifti1Image(arr, affine), str(path))


@pytest.mark.parametrize(
    "shape, spacing",
    [
        ((512, 512, 128), (0.75, 0.75, 3.0)),  # text2ct (H, W, D)
        ((201, 512, 512), (1.5, 0.75, 0.75)),  # generatect hires (D, H, W)
        ((201, 128, 128), (1.5, 3.0, 3.0)),  # generatect low-res (D, H, W)
    ],
)
def test_output_spacing_roundtrips_to_header_zooms(tmp_path, shape, spacing) -> None:
    """diag(output_spacing) → NIfTI header zooms, position-for-position."""
    arr = np.zeros(shape, dtype=np.int16)
    out = tmp_path / "vol.nii.gz"
    _save_with_spacing(arr, spacing, out)
    zooms = tuple(float(z) for z in nib.load(str(out)).header.get_zooms()[:3])
    assert tuple(round(z, 4) for z in zooms) == spacing


def test_adapter_output_spacing_matches_inference_constants() -> None:
    """The adapters expose exactly the spacings inference.py will diag-stamp."""
    from src.baselines.generatect_adapter import GenerateCTAdapter  # noqa: PLC0415
    from src.baselines.text2ct_adapter import Text2CTAdapter  # noqa: PLC0415

    assert Text2CTAdapter(load_weights=False).output_spacing == (0.75, 0.75, 3.0)
    assert GenerateCTAdapter(load_super_resolution=True).output_spacing == (
        1.5,
        0.75,
        0.75,
    )
    assert GenerateCTAdapter(load_super_resolution=False).output_spacing == (
        1.5,
        3.0,
        3.0,
    )
