"""src/inference.py의 모델별 truthful NIfTI affine roundtrip 테스트.

각 어댑터의 output_spacing(squeezed-array 축 순서)으로 만든 대각 affine이
저장된 .nii.gz 헤더 zoom에 위치 그대로 반영되는지(축·대각 실수 방지), 그리고
어댑터들이 실제로 그 spacing을 노출하는지 확인한다. GPU·가중치 불필요.
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
