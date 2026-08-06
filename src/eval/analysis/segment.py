"""Segmentation backend abstraction for generated CT volumes.

``segment_volume`` produces a raw TotalSegmentator-v2 117-label map (the same label scheme as
the on-disk ``ts_seg`` reference masks) from a generated ``.mha``. Grouping to the 4 chest
organs happens downstream via ``src/data/organ_groups.py::apply_grouping`` — this module never
groups on its own, so callers can also inspect any of the 117 raw labels if needed later.

**Resolution: the default is full-res (``fast=False``), because that is what the reference masks
are.** Measured 2026-08-01 (TotalSegmentator 2.13.0, GPU) by re-segmenting three CT-RATE valid
scans on their own original grid — no resize, so a direct voxel-wise comparison — against the
shipped ``ts_seg`` mask:

===============  ==========================  ==========================
scan             mean organ Dice, ``fast``   mean organ Dice, full-res
===============  ==========================  ==========================
valid_1004_a_1   0.9364                      1.0000
valid_1005_a_1   0.8731                      1.0000
valid_1006_a_1   0.8877                      0.9999
===============  ==========================  ==========================

Full-res reproduces the shipped masks essentially exactly, so they were generated that way.
Segmenting a *prediction* at 3 mm while the reference is 1.5 mm therefore puts a **ceiling** on
every Dice/surface metric that has nothing to do with model quality — per organ, roughly
lung 0.973, heart 0.929, aorta 0.883, **esophagus 0.811** (same three scans). A flawless
generated volume could not score above those, and the historically-recorded lung Dice 0.9645
(``seg_metrics.py`` module docstring) sits at 99 % of its ceiling — i.e. that number was almost
entirely backend mismatch, not model geometry. Matching the backend removes the ceiling; it
costs ~70-115 s/volume instead of ~20-50 s, which ``seg_max_n`` (see :func:`segment_volume`)
exists to bound. ⚠ Numbers produced before 2026-08-01 were all ``fast=True`` and are
ceiling-limited — do not table them next to full-res ones.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk


def _mha_to_nib_xyz(mha_path: str | Path):
    """Read a ``.mha`` (sitk convention, array axes Z,Y,X) into a nibabel image (X,Y,Z).

    ⚠ The affine is derived from the file's own header and is **not** a plain diagonal spacing
    matrix, even though the generated volumes carry a synthetic grid with identity
    origin/direction (plan §3.8). SimpleITK's world is **LPS**, NIfTI/nibabel's is **RAS**, so an
    ITK *identity* direction is ``diag(-1, -1, +1)`` in NIfTI terms — writing
    ``np.diag([sx, sy, sz, 1])`` declares LPS content to be RAS. That matters because
    TotalSegmentator canonicalizes its input by this affine
    (``totalsegmentator/nnunet.py:476`` ``as_closest_canonical``, undone at ``:713``), so a
    wrong-handed affine makes the canonicalization a no-op and feeds nnU-Net a patient rotated
    180° about S — left/right and anterior/posterior swapped.

    Measured cost of getting it wrong (2026-08-01, ``valid_1006_a_1``, one prediction segmented
    twice with **only** the affine sign changed, scored against the same reference mask):

    ===============  =============================  ==========================
    metric           ``diag(+sx, +sy, +sz)`` (RAS)  header-faithful (LPS→RAS)
    ===============  =============================  ==========================
    Dice lung        0.927                          0.966
    Dice heart       0.136                          0.893
    Dice aorta       0.128                          0.898
    Dice esophagus   0.000                          0.654
    hd95_mm mean     87.4                           5.3
    ===============  =============================  ==========================

    **lung barely moves** because ``organ_groups.TS_TO_GROUP`` merges the five lobes into one
    left-right symmetric label — which is exactly why the historical "lung Dice 0.9645"
    validation (``seg_metrics.py`` module docstring) could not see this.
    """
    import nibabel as nib

    img = sitk.ReadImage(str(mha_path))
    arr_zyx = sitk.GetArrayFromImage(img).astype(np.int16)
    arr_xyz = arr_zyx.transpose(2, 1, 0)

    # ITK index -> LPS world:  p_lps = origin + direction @ (spacing * index)
    # NIfTI index -> RAS world: the same map, then LPS -> RAS (negate the first two axes).
    direction = np.array(img.GetDirection()).reshape(
        3, 3
    )  # column j = axis j's direction
    lps_to_ras = np.diag([-1.0, -1.0, 1.0])
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = lps_to_ras @ (direction * np.array(img.GetSpacing()))
    affine[:3, 3] = lps_to_ras @ np.array(img.GetOrigin())
    return nib.Nifti1Image(arr_xyz, affine)


def _segment_totalseg(mha_path: str | Path, fast: bool = False) -> np.ndarray:
    """TotalSegmentator v2, in-process (GPU). Returns a label map ``(X, Y, Z)`` in {0..117}.

    ``output=None, ml=True`` returns a single multi-label ``Nifti1Image`` in-process (verified
    2026-07-27 against a real generated volume: output grid == input grid, labels 0-117 matching
    the on-disk ``ts_seg`` scheme exactly). ``fast`` defaults to ``False`` so predictions are
    segmented in the same resolution as those reference masks — see the module docstring for the
    measurement and for the metric ceiling ``fast=True`` imposes.
    """
    from totalsegmentator.python_api import totalsegmentator

    nib_img = _mha_to_nib_xyz(mha_path)
    result = totalsegmentator(
        nib_img, output=None, ml=True, fast=fast, device="gpu", task="total", quiet=True
    )
    return np.round(result.get_fdata()).astype(np.int64)


def _segment_vista3d(mha_path: str | Path, fast: bool = False) -> np.ndarray:
    """VISTA3D backend — plug-in interface only, not implemented in this phase.

    ``fast`` is part of the shared backend signature (``_BACKENDS`` dispatches positionally) and
    is unused here; this function raises before it could mean anything.

    Would require: downloading the MONAI VISTA3D bundle, a VISTA3D-label -> {0..4} organ-group
    mapping (VISTA3D's label ids differ from TotalSegmentator v2's), and re-segmenting the GT
    reference volumes with VISTA3D too (so pred and reference share one segmentation tool) —
    see plan §7 (deferred; TotalSegmentator is the default because it already matches the
    existing GT/conditioning-mask scheme, giving a common-mode-cancelling comparison).
    """
    raise NotImplementedError(
        "VISTA3D segmentation backend is not implemented (plan: TotalSegmentator is the "
        "default; VISTA3D needs a bundle download + a VISTA3D->organ_groups label map + "
        "re-segmenting the GT reference with VISTA3D for a fair comparison — deferred)."
    )


_BACKENDS = {
    "totalseg": _segment_totalseg,
    "vista3d": _segment_vista3d,
}


def segment_volume(
    mha_path: str | Path, backend: str = "totalseg", fast: bool = False
) -> np.ndarray:
    """Segment a generated CT volume into a raw label map.

    Args:
        mha_path: path to the generated ``.mha`` (int16 HU).
        backend: ``"totalseg"`` (default) or ``"vista3d"`` (not implemented).
        fast: TotalSegmentator's 3 mm model instead of full-res 1.5 mm. ``False`` (default)
            matches the shipped ``ts_seg`` reference masks; ``True`` is ~3x faster but caps
            every downstream metric (module docstring). Ignored by ``vista3d``.

    Returns:
        Label map ``(X, Y, Z)`` — same shape as the input volume, values in the segmentation
        backend's native label scheme (TotalSegmentator v2: {0..117}).

    Cost control: this segmentation pass dominates the whole per-sample analysis. Bound it by
    segmenting **fewer volumes** (``seg_max_n``) rather than by lowering the resolution, so the
    numbers you do get stay comparable against the full-res reference::

        # every generated volume, full-res      (~70-115 s/volume => ~6-10 h at n=300)
        python scripts/score_subgroups.py --pred-dir <run>/predictions --out <run> \\
            --n 300 --dice --hd95

        # first 100 volumes only                (~2-3 h; the rest get
        #                                        failure_reason="seg_max_n_exceeded")
        python scripts/score_subgroups.py --pred-dir <run>/predictions --out <run> \\
            --n 300 --dice --hd95 --seg-max-n 100

        # same knob on the Hydra path
        python scripts/run_eval.py ... task.metrics.dice=true task.metrics.hd95=true \\
            task.seg_max_n=100

        # deliberately reproduce a pre-2026-08-01 3 mm number (ceiling-limited — never table it
        # next to a full-res one)
        segment_volume(path, fast=True)
    """
    if backend not in _BACKENDS:
        raise ValueError(
            f"Unknown segmentation backend {backend!r}; choices: {list(_BACKENDS)}"
        )
    return _BACKENDS[backend](mha_path, fast)


__all__ = ["segment_volume"]
