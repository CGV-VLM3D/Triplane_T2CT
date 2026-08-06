#!/usr/bin/env python3
"""Resample a CT volume onto the CLIPScore evaluation grid and save it as .mha.

Every volume (GT or a model prediction) that passes through here lands on the
*identical* geometry that ``evaluate_clip.py`` feeds to CT-CLIP:

    grid   : (D, H, W) = (240, 480, 480)
    spacing: (z, x, y) = (1.5, 0.75, 0.75) mm
    HU      : clipped to [-1000, 1000], pad regions filled with -1.0

Because the grid, spacing, origin (0,0,0) and direction (identity) are the same
for all outputs, the resampled files overlap perfectly in a medical viewer
(3D Slicer / ITK-SNAP) — which is what the raw predictions do NOT do (native
sizes/spacings differ per model).

The core resampling logic is transcribed 1:1 from the upstream CLIP evaluator so
that what you view here matches exactly what CLIPScore consumed:
    third_party/vlm3d_dockers/ct_challenges/ctgen_evaluation/evaluate_clip.py

Usage
-----
    python resample_to_clip_grid.py <input.mha|.nii.gz> [--out <path.mha>]
    python resample_to_clip_grid.py <input> --outdir <dir> --tag gt

If neither --out nor --outdir is given, writes to ./resampled/<stem>_clipgrid.mha
next to this script.
"""

from __future__ import annotations

import argparse
import pathlib
import typing

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

# ─────────────── CLIP-eval constants (DUPLICATION INTENTIONAL) ───────────────
# Transcribed verbatim from evaluate_clip.py:40-43.
TARGET_SPACING: typing.Tuple[float, float, float] = (1.5, 0.75, 0.75)  # (z, x, y)
TARGET_DHW: typing.Tuple[int, int, int] = (240, 480, 480)  # (D, H, W)
HU_MIN, HU_MAX = -1000.0, 1000.0
PAD_VALUE = -1.0  # keep outside HU range


# ─────────────── CLIP-eval helpers (DUPLICATION INTENTIONAL) ───────────────
def _read_mha(p: pathlib.Path) -> np.ndarray:
    """MetaImage → float32 ndarray (D, H, W).  (evaluate_clip.py:47-50)"""
    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(p))).astype(np.float32)  # (Z, Y, X)
    return arr.transpose(0, 2, 1)  # (D, H, W) = (Z, X, Y)


def _resize_array(
    vol: torch.Tensor, cur_spacing: typing.Tuple[float, float, float]
) -> torch.Tensor:
    """Resample `vol` (D,H,W) to TARGET_SPACING, trilinear.  (evaluate_clip.py:52-63)"""
    d, h, w = vol.shape
    scale = [cur_spacing[i] / TARGET_SPACING[i] for i in range(3)]
    new_size = [int(round(dim * sc)) for dim, sc in zip((d, h, w), scale)]
    return F.interpolate(
        vol.unsqueeze(0).unsqueeze(0),  # → (1,1,D,H,W)
        size=new_size,
        mode="trilinear",
        align_corners=False,
    )[0, 0]  # back to (D,H,W)


def _centre_crop_or_pad(v: torch.Tensor) -> torch.Tensor:
    """Centre-crop or pad to TARGET_DHW with PAD_VALUE.  (evaluate_clip.py:65-74)"""
    D, H, W = TARGET_DHW
    pad = [
        (W - v.shape[2]) // 2,
        W - v.shape[2] - (W - v.shape[2]) // 2,
        (H - v.shape[1]) // 2,
        H - v.shape[1] - (H - v.shape[1]) // 2,
        (D - v.shape[0]) // 2,
        D - v.shape[0] - (D - v.shape[0]) // 2,
    ]
    v = F.pad(v, pad, value=PAD_VALUE)  # PyTorch expects (W1, W2, H1, H2, D1, D2)
    return v[:D, :H, :W]  # guard tiny rounding mismatches


def _load_vol(p: pathlib.Path, flip_xy: bool = False) -> torch.Tensor:
    """Read, HU-clip, resample, crop/pad a single volume.  (evaluate_clip.py:76-90)

    Args:
        flip_xy: flip the two in-plane axes (X and Y). Use for the GT `.mha`,
            which is stored in raw-nifti LPS voxel order while every model
            prediction is decoded in RAS — LPS→RAS is exactly a flip of X and Y
            (verified: RAS == LPS[::-1,::-1,:]). Aligns GT to the predictions so
            all files overlap in a viewer.

    Returns:
        ``(240, 480, 480)`` float32 tensor in (D,H,W)=(Z,X,Y) order, HU units.
    """
    arr = _read_mha(p)  # (Z,X,Y)
    if flip_xy:
        arr = arr[:, ::-1, ::-1].copy()  # flip X,Y → LPS→RAS
    arr = np.clip(arr, HU_MIN, HU_MAX)

    itk_img = sitk.ReadImage(str(p))  # need spacing (x,y,z)
    sx, sy, sz = itk_img.GetSpacing()  # (x, y, z)
    v = torch.from_numpy(arr)  # (Z,X,Y)

    v = _resize_array(v, (sz, sy, sx))  # spacing order → (z,·,·)
    v = _centre_crop_or_pad(v)  # (240,480,480)
    # NOTE: CLIPScore additionally does `v / 1000.0` before the network
    # (evaluate_clip.py loop); we keep HU units here so the saved file is
    # window/level-friendly in a viewer.
    return v


# ─────────────────────────── save ───────────────────────────
def resample_and_save(
    in_path: pathlib.Path, out_path: pathlib.Path, flip_xy: bool = False
) -> None:
    """Resample `in_path` onto the CLIP grid and write `out_path` (.mha)."""
    v = _load_vol(in_path, flip_xy=flip_xy)  # (Z,X,Y)=(240,480,480)
    # Undo the _read_mha X/Y transpose so the saved volume keeps the source's
    # anatomical orientation: (Z,X,Y) → (Z,Y,X), the order sitk expects.
    arr_zyx = v.numpy().transpose(0, 2, 1).copy()  # (Z,Y,X)=(240,480,480)

    img = sitk.GetImageFromArray(arr_zyx)  # size (X,Y,Z)=(480,480,240)
    img.SetSpacing(
        (TARGET_SPACING[1], TARGET_SPACING[2], TARGET_SPACING[0])
    )  # (x,y,z)=(0.75,0.75,1.5)
    img.SetOrigin((0.0, 0.0, 0.0))
    img.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(out_path), useCompression=True)
    print(f"[saved] {out_path}  size={img.GetSize()} spacing={img.GetSpacing()}")


def _default_out(in_path: pathlib.Path) -> pathlib.Path:
    return (
        pathlib.Path(__file__).parent / "resampled" / f"{_stem(in_path)}_clipgrid.mha"
    )


def _stem(p: pathlib.Path) -> str:
    n = p.name
    for ext in (".nii.gz", ".nii", ".mha"):
        if n.lower().endswith(ext):
            return n[: -len(ext)]
    return p.stem


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Resample a CT volume onto the CLIPScore eval grid."
    )
    ap.add_argument("input", type=pathlib.Path, help="input .mha / .nii.gz volume")
    ap.add_argument("--out", type=pathlib.Path, help="explicit output .mha path")
    ap.add_argument(
        "--outdir", type=pathlib.Path, help="output directory (filename auto)"
    )
    ap.add_argument(
        "--tag", type=str, help="suffix tag, e.g. 'gt' → <stem>__gt.mha (with --outdir)"
    )
    ap.add_argument(
        "--flip-xy",
        action="store_true",
        help="flip in-plane X,Y (LPS→RAS). Use for the GT .mha so it aligns "
        "with RAS-decoded predictions.",
    )
    args = ap.parse_args()

    if args.out:
        out = args.out
    elif args.outdir:
        name = (
            _stem(args.input) + (f"__{args.tag}" if args.tag else "") + "_clipgrid.mha"
        )
        out = args.outdir / name
    else:
        out = _default_out(args.input)

    resample_and_save(args.input, out, flip_xy=args.flip_xy)


if __name__ == "__main__":
    main()
