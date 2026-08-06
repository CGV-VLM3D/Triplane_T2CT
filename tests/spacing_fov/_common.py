"""Shared helpers for the spacing/FOV study: eval preprocessing replicas + body masks.

The two preprocessing chains below are **replicas of the upstream VLM3D-Dockers eval code**.
`third_party/` is read-only (CLAUDE.md P2), so we cannot instrument the originals in place;
instead each chain is transcribed with a `DUPLICATION INTENTIONAL` note and a file:line
citation, and both accept a `spacing_override` so the study can ask "what would the eval see
if this same voxel array were declared at spacing s?" without rewriting any file.

Parity guard: `relabel_sweep.py` gates its fast path by reproducing the production
`CTGenEvaluator` FID at the volumes' native declared spacing.
"""

from __future__ import annotations

import pathlib

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from scipy import ndimage

# --- FID chain constants (compute_fid_2-5d_ct.py:532-601 + ctgen.py:43-46) --------------- #
FID_TARGET_SHAPE = (512, 512, 512)
FID_RESAMPLE_MM = (1.0, 1.0, 1.0)
FID_PAD_VALUE = -1000

# --- CLIP chain constants (evaluate_clip.py:40-43) -------------------------------------- #
CLIP_TARGET_SPACING = (1.5, 0.75, 0.75)  # (z, x, y)
CLIP_TARGET_DHW = (240, 480, 480)
CLIP_HU_MIN, CLIP_HU_MAX = -1000.0, 1000.0
CLIP_PAD_VALUE = (
    -1.0
)  # upstream really does pad with -1.0, not -1000 (evaluate_clip.py:43)

BODY_HU_THRESHOLD = -500.0  # soft tissue / air separation
LUNG_HU_THRESHOLD = -400.0


def read_hu(path: str | pathlib.Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Read a volume as HU. Returns ``(arr_zyx, spacing_xyz)`` — ``arr`` is ``(Z, Y, X)``."""
    img = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(img).astype(np.float32), img.GetSpacing()


def fid_preproc(
    path: str | pathlib.Path,
    spacing_override: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """Run the upstream 2.5D-FID preprocessing chain on one volume.

    DUPLICATION INTENTIONAL — transcribed from
    ``third_party/vlm3d_dockers/ct_challenges/ctgen_evaluation/compute_fid_2-5d_ct.py:580-601``
    (LoadImaged → EnsureChannelFirstd → Orientationd(RAS) → Spacingd → SpatialPadd →
    CenterSpatialCropd → ScaleIntensityRanged). Upstream is read-only, and we need the
    ability to substitute a spacing, which the original cannot do.

    Args:
        path: volume file (``.mha`` / ``.nii.gz``).
        spacing_override: if given, the volume is treated as if its header declared this
            ``(x, y, z)`` spacing. ``None`` uses the real header.

    Returns:
        HU array ``(512, 512, 512)`` in RAS axis order ``(R, A, S)``, clipped to
        ``[-1000, 1000]``, padded with ``-1000``.
    """
    import monai  # noqa: PLC0415 — heavy import, only needed here

    loader = monai.transforms.Compose(
        [
            monai.transforms.LoadImaged(keys=["image"]),
            monai.transforms.EnsureChannelFirstd(keys=["image"]),
        ]
    )
    data = loader({"image": str(path)})
    if spacing_override is not None:
        # Rewrite the affine's scale so downstream Spacingd resamples as if the header said so.
        # Our saved volumes carry an identity direction (report2ct.py:_save_mha), so scaling
        # the diagonal is exactly equivalent to a SetSpacing on the file.
        affine = np.asarray(data["image"].affine, dtype=np.float64).copy()  # (4, 4)
        current = np.linalg.norm(affine[:3, :3], axis=0)  # (3,) per-axis voxel size
        affine[:3, :3] = affine[:3, :3] @ np.diag(
            np.asarray(spacing_override) / current
        )
        data["image"].affine = torch.as_tensor(affine)

    chain = monai.transforms.Compose(
        [
            monai.transforms.Orientationd(keys=["image"], axcodes="RAS"),
            monai.transforms.Spacingd(
                keys=["image"], pixdim=FID_RESAMPLE_MM, mode=["bilinear"]
            ),
            monai.transforms.SpatialPadd(
                keys=["image"],
                spatial_size=FID_TARGET_SHAPE,
                mode="constant",
                value=FID_PAD_VALUE,
            ),
            monai.transforms.CenterSpatialCropd(
                keys=["image"], roi_size=FID_TARGET_SHAPE
            ),
            monai.transforms.ScaleIntensityRanged(
                keys=["image"],
                a_min=-1000,
                a_max=1000,
                b_min=-1000,
                b_max=1000,
                clip=True,
            ),
        ]
    )
    out = chain(data)["image"]  # (1, 512, 512, 512)
    return np.asarray(out[0], dtype=np.float32)  # (512, 512, 512)


def clip_preproc(
    path: str | pathlib.Path,
    spacing_override: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """Run the upstream CLIPScore preprocessing chain on one volume.

    DUPLICATION INTENTIONAL — transcribed from
    ``third_party/vlm3d_dockers/ct_challenges/ctgen_evaluation/evaluate_clip.py:47-91``
    (``_read_mha`` → clip → ``_resize_array`` → ``_centre_crop_or_pad``). Same reason as
    ``fid_preproc``: upstream is read-only and reads spacing straight from the file header.

    Args:
        path: volume file.
        spacing_override: treat the volume as if declared at this ``(x, y, z)`` spacing.

    Returns:
        HU array ``(240, 480, 480)`` = ``(D, H, W)``; regions outside the resampled volume
        are ``-1.0`` (upstream's pad value).
    """
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)  # (Z, Y, X)
    arr = arr.transpose(0, 2, 1)  # (D, H, W)  — upstream _read_mha:50
    arr = np.clip(arr, CLIP_HU_MIN, CLIP_HU_MAX)  # upstream :79

    sx, sy, sz = spacing_override if spacing_override is not None else img.GetSpacing()
    vol = torch.from_numpy(arr)  # (D, H, W)

    # upstream _resize_array:52-63 — scale = cur/target with cur given as (z, x, y)
    cur = (sz, sy, sx)
    scale = [cur[i] / CLIP_TARGET_SPACING[i] for i in range(3)]
    new_size = [int(round(d * s)) for d, s in zip(vol.shape, scale)]
    vol = F.interpolate(
        vol.unsqueeze(0).unsqueeze(0),
        size=new_size,
        mode="trilinear",
        align_corners=False,
    )[0, 0]  # (D', H', W')

    # upstream _centre_crop_or_pad:65-74 — negative pads crop, positive pads fill with -1.0
    d, h, w = CLIP_TARGET_DHW
    pad = [
        (w - vol.shape[2]) // 2,
        w - vol.shape[2] - (w - vol.shape[2]) // 2,
        (h - vol.shape[1]) // 2,
        h - vol.shape[1] - (h - vol.shape[1]) // 2,
        (d - vol.shape[0]) // 2,
        d - vol.shape[0] - (d - vol.shape[0]) // 2,
    ]
    vol = F.pad(vol, pad, value=CLIP_PAD_VALUE)
    return np.asarray(vol[:d, :h, :w], dtype=np.float32)  # (240, 480, 480)


def body_mask(vol_hu: np.ndarray) -> np.ndarray:
    """Largest connected soft-tissue component = the patient's body.

    Args:
        vol_hu: HU volume of any 3-D shape.

    Returns:
        Boolean mask of the same shape; all-False if no component is found.
    """
    binary = vol_hu > BODY_HU_THRESHOLD
    binary = ndimage.binary_opening(binary, iterations=2)
    labels, n = ndimage.label(binary)
    if n == 0:
        return np.zeros_like(binary, dtype=bool)
    counts = np.bincount(labels.ravel())
    counts[0] = 0  # background
    return labels == counts.argmax()


def lung_mask(vol_hu: np.ndarray, body: np.ndarray, slice_axis: int = 2) -> np.ndarray:
    """Air pockets inside the body = lungs (2 largest components).

    Args:
        vol_hu: HU volume.
        body: body mask from :func:`body_mask` (same shape).
        slice_axis: axis to fill holes along, slice by slice. Default 2 = the superior
            axis of a RAS volume, i.e. fill within each axial slice.

    Returns:
        Boolean mask of the same shape.

    Note:
        Hole filling is done **per axial slice, not in 3D**. In 3D the lungs are not a closed
        cavity — the trachea vents them to the air outside the patient at the top of the
        volume — so ``binary_fill_holes`` on the whole volume fills almost nothing and the
        lungs are lost. Within a single axial slice the lung *is* fully enclosed by the chest
        wall, so the 2-D fill is reliable.
    """
    filled = np.moveaxis(
        np.stack(
            [ndimage.binary_fill_holes(sl) for sl in np.moveaxis(body, slice_axis, 0)],
            axis=0,
        ),
        0,
        slice_axis,
    )
    binary = (vol_hu < LUNG_HU_THRESHOLD) & filled
    binary = ndimage.binary_opening(binary, iterations=2)
    labels, n = ndimage.label(binary)
    if n == 0:
        return np.zeros_like(binary, dtype=bool)
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    keep = np.argsort(counts)[-2:]  # two largest = left + right lung
    keep = [k for k in keep if counts[k] > 0]
    return np.isin(labels, keep)


def bbox_extent(mask: np.ndarray) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Bounding-box size (voxels) and centroid (voxels) of a boolean mask.

    Returns:
        ``((ext0, ext1, ext2), (c0, c1, c2))``; all zeros if the mask is empty.
    """
    if not mask.any():
        return (0, 0, 0), (0, 0, 0)
    idx = [
        np.where(mask.any(axis=tuple(j for j in range(mask.ndim) if j != i)))[0]
        for i in range(mask.ndim)
    ]
    extent = tuple(int(a[-1] - a[0] + 1) for a in idx)
    centroid = tuple(int(round(c)) for c in ndimage.center_of_mass(mask))
    return extent, centroid


__all__ = [
    "read_hu",
    "fid_preproc",
    "clip_preproc",
    "body_mask",
    "lung_mask",
    "bbox_extent",
    "FID_TARGET_SHAPE",
    "CLIP_TARGET_DHW",
]
