#!/usr/bin/env python3
"""E3 — quantify body/lung geometry in the space the FID metric actually works in.

Every volume is pushed through the FID preprocessing chain (RAS → 1 mm → pad/crop 512^3),
so 1 voxel == 1 mm and the numbers are directly comparable across arms and to GT.

The load-bearing column is **body_ap_grid**: the anterior-posterior body extent converted
back into *original grid voxels* (mm / declared spacing). That separates the two effects —

  * body_ap_mm    : what the evaluator sees  → moves with the DECLARED spacing (hypothesis A)
  * body_ap_grid  : what the model drew      → moves only if the CONDITIONING changed the
                    content (hypothesis B). Identical content ⇒ ratio 1.00 between arms;
                    a spacing-faithful model would show a 1.25x ratio between s=1.0 and 0.8.

Left-right extent is reported but is NOT a scale measure: chest CT FOVs cut through the arms,
so the body touches the frame edge and the extent saturates at the FOV.

Masks are computed on a 2x-downsampled copy (the extents are then in 2 mm units and scaled
back); at 1 mm isotropic that costs ~2 mm of precision on a ~300 mm measurement and makes the
n=100 x 3-arm sweep tractable.

Usage:
    python tests/spacing_fov/body_metrics.py            # all cases present in every arm
    python tests/spacing_fov/body_metrics.py --limit 20
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import SimpleITK as sitk  # noqa: E402

sys.path.insert(0, "/workspace")

from tests.spacing_fov._common import (  # noqa: E402
    FID_TARGET_SHAPE,
    bbox_extent,
    body_mask,
    fid_preproc,
    lung_mask,
)

HERE = pathlib.Path(__file__).parent
FIGS = HERE / "figs"
RESULTS = HERE / "results"

GT_DIR = pathlib.Path("/workspace/data/vlm3d_eval/_valid_full_3001")
ARMS = {
    "real": GT_DIR,
    "gen@1.0": HERE / "preds" / "gen_sp1.0",
    "gen@0.8": HERE / "preds" / "gen_sp0.8",
}
DOWNSAMPLE = 2  # mask computation stride; extents are multiplied back to mm


def _declared_spacing(path: pathlib.Path) -> tuple[float, float, float]:
    """Header spacing (x, y, z) in mm."""
    r = sitk.ImageFileReader()
    r.SetFileName(str(path))
    r.ReadImageInformation()
    return tuple(float(s) for s in r.GetSpacing())


def _extent_mm(path: pathlib.Path, axis: int) -> float:
    """Size (mm) the volume occupies inside the FID frame along one axis.

    Capped at the frame: a few GT volumes have a z FOV beyond 512 mm (max 708) and are
    center-cropped, so their usable extent is the frame, not the FOV.
    """
    r = sitk.ImageFileReader()
    r.SetFileName(str(path))
    r.ReadImageInformation()
    return min(float(FID_TARGET_SHAPE[axis]), r.GetSize()[axis] * r.GetSpacing()[axis])


def measure(args: tuple[str, str]) -> dict | None:
    """Measure one (arm, case) volume. Returns a flat dict of metrics, or None on failure."""
    arm, case = args
    path = ARMS[arm] / f"{case}.mha"
    try:
        vol = fid_preproc(path)  # (512, 512, 512), 1 mm isotropic, RAS
    except Exception as exc:  # noqa: BLE001 — a bad file must not kill the sweep
        print(f"  !! {arm} {case}: {exc}", file=sys.stderr)
        return None

    small = vol[::DOWNSAMPLE, ::DOWNSAMPLE, ::DOWNSAMPLE]  # (256, 256, 256)
    body = body_mask(small)
    if not body.any():
        return None
    (lr, ap, si), centroid = bbox_extent(body)
    lung = lung_mask(small, body)
    lung_si = bbox_extent(lung)[0][2] if lung.any() else 0

    mm = DOWNSAMPLE  # one downsampled voxel == DOWNSAMPLE mm
    frame_centre = FID_TARGET_SHAPE[0] / (2 * DOWNSAMPLE)
    sx, sy, sz = _declared_spacing(path)

    return {
        "arm": arm,
        "case": case,
        "declared_inplane_mm": sx,
        "declared_z_mm": sz,
        # what the evaluator sees (mm == px after the 1 mm resample)
        "body_lr_mm": lr * mm,  # saturates at the FOV — not a scale measure
        "body_ap_mm": ap * mm,
        "body_si_mm": si * mm,
        "body_frac": float(body.mean()),  # fraction of the 512^3 FID frame that is body
        "lung_si_mm": lung_si * mm,
        "centroid_off_lr_mm": (centroid[0] - frame_centre) * mm,
        "centroid_off_ap_mm": (centroid[1] - frame_centre) * mm,
        "centroid_off_si_mm": (centroid[2] - frame_centre) * mm,
        # What the model drew, in its own 480x480x256 grid units. Only comparable BETWEEN the
        # generated arms: GT lives on its own native grid (512 or 1024 px), so dividing its mm
        # extent by its native spacing yields a number on a different scale entirely.
        # divide by sy, not sx: the extent being converted is the anterior-posterior one, and
        # AP is the y axis. (sx == sy for every volume here, but the pairing should be right.)
        "body_ap_grid": float("nan") if arm == "real" else ap * mm / sy,
        # True when the body touches the volume's data extent on that axis, i.e. the FOV cuts
        # through the patient — the extent then measures the FOV, not the body.
        "sat_lr": bool(lr * mm >= _extent_mm(path, 0) - 2 * mm),
        "sat_ap": bool(ap * mm >= _extent_mm(path, 1) - 2 * mm),
        "sat_si": bool(si * mm >= _extent_mm(path, 2) - 2 * mm),
    }


def _agg(rows: list[dict], arm: str, key: str) -> tuple[float, float]:
    """mean, sd of one metric for one arm (NaN entries, e.g. body_ap_grid for GT, are dropped)."""
    vals = np.array([r[key] for r in rows if r["arm"] == arm], dtype=np.float64)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    return float(vals.mean()), float(vals.std())


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--limit", type=int, default=None)
    ap_.add_argument("--workers", type=int, default=8)
    args = ap_.parse_args()

    FIGS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    common = None
    for d in ARMS.values():
        ids = {p.stem for p in d.glob("*.mha")}
        common = ids if common is None else (common & ids)
    cases = sorted(common)
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        sys.exit("no case present in all arms yet")
    print(f"[E3] {len(cases)} cases x {len(ARMS)} arms")

    jobs = [(arm, c) for c in cases for arm in ARMS]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = [r for r in ex.map(measure, jobs) if r is not None]

    metrics = [
        "body_ap_mm",
        "body_si_mm",
        "body_lr_mm",
        "body_frac",
        "lung_si_mm",
        "body_ap_grid",
        "centroid_off_ap_mm",
        "centroid_off_si_mm",
    ]
    summary = {
        "n_cases": len(cases),
        "downsample": DOWNSAMPLE,
        "per_arm": {
            arm: {m: dict(zip(("mean", "sd"), _agg(rows, arm, m))) for m in metrics}
            for arm in ARMS
        },
        # How often each axis' body extent is FOV-limited. An axis that saturates for most
        # volumes cannot be read as a body-size measurement.
        "saturation_rate": {
            arm: {
                k: float(np.mean([r[k] for r in rows if r["arm"] == arm]))
                for k in ("sat_lr", "sat_ap", "sat_si")
            }
            for arm in ARMS
        },
    }
    (RESULTS / "body_metrics.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2)
    )

    # ---- console table ----
    print(f"\n{'metric':22s}" + "".join(f"{a:>16s}" for a in ARMS))
    for m in metrics:
        line = f"{m:22s}"
        for arm in ARMS:
            mu, sd = (
                summary["per_arm"][arm][m]["mean"],
                summary["per_arm"][arm][m]["sd"],
            )
            line += f"{mu:>10.2f}±{sd:<5.2f}"
        print(line)

    # ---- figure: distributions, GT as the reference ----
    show = ["body_ap_mm", "body_frac", "lung_si_mm", "body_ap_grid"]
    fig, axes = plt.subplots(1, len(show), figsize=(4.2 * len(show), 3.8))
    colors = {"real": "0.35", "gen@1.0": "tab:red", "gen@0.8": "tab:olive"}
    for ax, m in zip(axes, show):
        for arm in ARMS:
            vals = [r[m] for r in rows if r["arm"] == arm and not np.isnan(r[m])]
            if not vals:
                continue  # body_ap_grid is undefined for GT (different native grid)
            ax.hist(vals, bins=25, alpha=0.55, label=arm, color=colors[arm])
        ax.set_title(m, fontsize=10)
        ax.legend(fontsize=7)
    axes[-1].set_xlabel("original-grid voxels (content measure)")
    fig.suptitle(
        f"E3 body/lung geometry after the FID chain (n={len(cases)}) — "
        "body_ap_mm = what the metric sees, body_ap_grid = what the model drew",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIGS / "F4_body_metrics.png", dpi=150)
    plt.close(fig)
    print(f"\n[E3] → {RESULTS / 'body_metrics.json'}, {FIGS / 'F4_body_metrics.png'}")


if __name__ == "__main__":
    main()
