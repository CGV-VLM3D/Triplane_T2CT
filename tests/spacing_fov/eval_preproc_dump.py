#!/usr/bin/env python3
"""E2 — what the evaluator actually sees, after its own preprocessing.

Comparing raw generated volumes tells you little: the metrics never see them. FID resamples
to 1 mm and pads to 512^3; CLIPScore resamples to (1.5, 0.75, 0.75) mm and center-crops to
(240, 480, 480). This script pushes real / gen@1.0 / gen@0.8 through both chains, dumps the
results as NIfTI (inspectable in MITK), and renders the 3x2 comparison panels.

The dashed rectangle in each panel is the extent of the resampled volume inside the eval's
fixed frame — everything outside it is padding (FID) or was cropped away (CLIP).

Usage:
    python tests/spacing_fov/eval_preproc_dump.py                  # auto-pick a case
    python tests/spacing_fov/eval_preproc_dump.py --case valid_598_a_1
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import SimpleITK as sitk  # noqa: E402

sys.path.insert(0, "/workspace")

from tests.spacing_fov._common import (  # noqa: E402
    CLIP_PAD_VALUE,
    CLIP_TARGET_DHW,
    CLIP_TARGET_SPACING,
    FID_TARGET_SHAPE,
    body_mask,
    bbox_extent,
    clip_preproc,
    fid_preproc,
)

HERE = pathlib.Path(__file__).parent
FIGS = HERE / "figs"
DUMPS = HERE / "dumps"

GT_DIR = pathlib.Path("/workspace/data/vlm3d_eval/_valid_full_3001")
ARMS = {
    "real (GT)": GT_DIR,
    "gen@1.0": HERE / "preds" / "gen_sp1.0",
    "gen@0.8": HERE / "preds" / "gen_sp0.8",
}
HU_WINDOW = (-1000, 400)  # display window; wide enough to show body outline + lung


def _data_extent_fid(path: pathlib.Path) -> tuple[int, int, int]:
    """Size (voxels) the volume occupies inside the 512^3 FID frame after 1 mm resampling."""
    r = sitk.ImageFileReader()
    r.SetFileName(str(path))
    r.ReadImageInformation()
    size, spacing = r.GetSize(), r.GetSpacing()
    return tuple(
        min(FID_TARGET_SHAPE[i], int(round(size[i] * spacing[i]))) for i in range(3)
    )


def _resampled_size_clip(path: pathlib.Path) -> tuple[int, int, int]:
    """Size (voxels) after the CLIP chain's resample, BEFORE its center crop/pad, as (D, H, W).

    Anything larger than ``CLIP_TARGET_DHW`` on an axis is cropped away — that loss is
    invisible in the final volume, so it has to be reported as a number.
    """
    r = sitk.ImageFileReader()
    r.SetFileName(str(path))
    r.ReadImageInformation()
    sx, sy, sz = r.GetSpacing()
    nx, ny, nz = r.GetSize()
    # Upstream's `_resize_array` scales `vol.shape`, and after its transpose that shape is
    # (Z, X, Y), while `cur_spacing` is passed as (sz, sy, sx) — the x/y pairing is crossed in
    # the original. Reproduce it exactly rather than "fixing" it: identical for these volumes
    # (nx == ny, sx == sy) but the annotation must describe what upstream really does.
    cur, n = (sz, sy, sx), (nz, nx, ny)
    return tuple(int(round(n[i] * cur[i] / CLIP_TARGET_SPACING[i])) for i in range(3))


def _data_extent_clip(path: pathlib.Path) -> tuple[int, int, int]:
    """Size (voxels) the volume occupies inside the (240,480,480) CLIP frame, as (D, H, W)."""
    resampled = _resampled_size_clip(path)
    return tuple(min(CLIP_TARGET_DHW[i], resampled[i]) for i in range(3))


def _rect(ax, width: int, height: int, frame_w: int, frame_h: int) -> None:
    """Dashed rectangle marking the real-data extent, centered in the eval frame."""
    x0 = (frame_w - min(width, frame_w)) / 2
    y0 = (frame_h - min(height, frame_h)) / 2
    ax.add_patch(
        mpatches.Rectangle(
            (x0, y0),
            min(width, frame_w),
            min(height, frame_h),
            fill=False,
            ec="tab:orange",
            ls="--",
            lw=1.4,
        )
    )


def _render(
    chain: str,
    vols: dict,
    extents: dict,
    out_png: pathlib.Path,
    case: str,
    notes: dict | None = None,
) -> None:
    """Render the 3-row (arm) x 2-col (axial, coronal) comparison for one chain."""
    fig, axes = plt.subplots(len(vols), 2, figsize=(9.5, 4.4 * len(vols)))
    for row, (name, vol) in enumerate(vols.items()):
        ext = extents[name]
        if chain == "fid":
            # RAS axes: (R, A, S). axial = [:, :, S], coronal = [:, A, :]
            axial = vol[:, :, vol.shape[2] // 2].T
            coronal = vol[:, vol.shape[1] // 2, :].T
            ax_wh = (ext[0], ext[1])  # R, A
            co_wh = (ext[0], ext[2])  # R, S
            frame_ax = (FID_TARGET_SHAPE[0], FID_TARGET_SHAPE[1])
            frame_co = (FID_TARGET_SHAPE[0], FID_TARGET_SHAPE[2])
        else:
            # CLIP axes: (D, H, W) = (z, x, y). Coronal means "fixed y" → index the LAST axis;
            # no transpose, so the panel comes out with x horizontal and z (superior) up.
            axial = vol[vol.shape[0] // 2, :, :].T
            coronal = vol[:, :, vol.shape[2] // 2]
            ax_wh = (ext[1], ext[2])  # H, W
            co_wh = (ext[1], ext[0])  # H, D
            frame_ax = (CLIP_TARGET_DHW[1], CLIP_TARGET_DHW[2])
            frame_co = (CLIP_TARGET_DHW[1], CLIP_TARGET_DHW[0])

        # The CLIP chain pads with -1.0 HU (evaluate_clip.py:43), which sits ABOVE the
        # body threshold — mask those voxels out or the padding is counted as tissue.
        mask = body_mask(
            vol if chain == "fid" else np.where(vol == CLIP_PAD_VALUE, -1000.0, vol)
        )
        bb, _ = bbox_extent(mask)
        for col, (img, wh, frame, title) in enumerate(
            [
                (axial, ax_wh, frame_ax, "axial mid"),
                (coronal, co_wh, frame_co, "coronal mid"),
            ]
        ):
            ax = axes[row, col]
            ax.imshow(
                img, cmap="gray", vmin=HU_WINDOW[0], vmax=HU_WINDOW[1], origin="lower"
            )
            _rect(ax, wh[0], wh[1], frame[0], frame[1])
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                label = f"{name}\nbody bbox {bb}\nbody frac {mask.mean():.3f}"
                if chain == "clip" and notes is not None:
                    label += f"\nresampled {notes[name]}\n→ crop/pad {CLIP_TARGET_DHW}"
                ax.set_ylabel(label, fontsize=8)
            ax.set_title(f"{title}  ({img.shape[1]}x{img.shape[0]})", fontsize=9)
    chain_label = (
        "FID chain: RAS -> 1mm -> pad/crop 512^3"
        if chain == "fid"
        else "CLIP chain: -> (1.5,0.75,0.75)mm -> crop/pad (240,480,480)"
    )
    fig.suptitle(f"{chain_label}   |   case {case}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[E2] {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--case", default=None, help="scan_id; default = first case present in all arms"
    )
    ap.add_argument(
        "--no-dump", action="store_true", help="skip writing the NIfTI dumps"
    )
    args = ap.parse_args()

    FIGS.mkdir(parents=True, exist_ok=True)
    DUMPS.mkdir(parents=True, exist_ok=True)

    if args.case:
        case = args.case
    else:
        common = None
        for d in ARMS.values():
            ids = {p.stem for p in d.glob("*.mha")}
            common = ids if common is None else (common & ids)
        if not common:
            sys.exit(
                "no case present in all three arms yet — wait for generation to finish"
            )
        case = sorted(common)[0]
    print(f"[E2] case = {case}")

    paths = {name: d / f"{case}.mha" for name, d in ARMS.items()}
    missing = [n for n, p in paths.items() if not p.is_file()]
    if missing:
        sys.exit(f"missing volume for arms: {missing}")

    for chain, fn, ext_fn in (
        ("fid", fid_preproc, _data_extent_fid),
        ("clip", clip_preproc, _data_extent_clip),
    ):
        vols, extents, notes = {}, {}, {}
        for name, path in paths.items():
            vols[name] = fn(path)
            extents[name] = ext_fn(path)
            notes[name] = _resampled_size_clip(path) if chain == "clip" else None
            print(
                f"  {chain:4s} {name:10s} -> {vols[name].shape}  data extent {extents[name]}"
                + (
                    f"  (resampled {notes[name]} before crop)"
                    if chain == "clip"
                    else ""
                )
            )
            if not args.no_dump:
                tag = name.replace(" ", "").replace("(", "").replace(")", "")
                nib.save(
                    nib.Nifti1Image(vols[name], affine=np.eye(4)),
                    str(DUMPS / f"{case}__{chain}__{tag}.nii.gz"),
                )
        assert all(
            v.shape == (FID_TARGET_SHAPE if chain == "fid" else CLIP_TARGET_DHW)
            for v in vols.values()
        ), "preprocessed shape does not match the eval's fixed frame"
        _render(chain, vols, extents, FIGS / f"F3_{chain}_chain.png", case, notes)


if __name__ == "__main__":
    main()
