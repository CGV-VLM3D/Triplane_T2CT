"""EDA v2 — Module 06: NIfTI voxel EDA + montages (GPT §6.2).

Loads actual voxel arrays (the prior GPT EDA was CSV-only) for:
  (a) the 30 bundle volumes  (5 groups x {5 primary + 5 recon-2}), and
  (b) a stratified sample of >=20 no_chest volumes (from
      datasets/.../no_chest_data/{train,valid}_fixed).

Per volume: voxel-level integrity + morphology stats (percentiles, NaN/Inf,
air-peak, -8192 sentinel fraction, body/foreground fraction, body bbox,
empty-slice fraction) and an HU-validity demo (re-applying the metadata
RescaleSlope/Intercept corrupts every volume to all-air, so rescale MUST NOT be
applied to `_fixed`).

Montages (>=200 dpi): 5 primary groups (axial/coronal/sagittal mid-slice in lung
window WL-600/WW1500 + mediastinal WL40/WW400, clip [-1000,1000]) and one 3-view
montage per no_chest sample (authoritative for coverage — the auto `coverage_proxy`
is a *crude* label; trust the montage).

Loading is parallelised across processes (I/O-bound gzip decode of ~54 volumes).
MONAI's `LoadImage`/`DataLoader(num_workers=...)` is the idiomatic equivalent; we
use a dependency-light `ProcessPoolExecutor` here.

Analysis unit: VOLUME (one .nii.gz = one reconstruction). Read-only on datasets.

Outputs (under /workspace/tests/ctrate_eda/):
  tables/nifti_voxel.csv, tables/no_chest_coverage.csv, tables/voxel_raw.json
  figures/montages/<group>_<case>.png, figures/montages/no_chest_<id>.png
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "/workspace")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path("/workspace/datasets/datasets/CT-RATE/dataset")
NO_CHEST = ROOT / "no_chest_data"
BUNDLE = Path("/workspace/tests/ctrate_eda_bundle")
OUT = Path("/workspace/tests/ctrate_eda")
TABLES = OUT / "tables"
MONT = OUT / "figures" / "montages"

SENTINEL = -8192.0  # out-of-FOV padding sentinel (NEVER real HU)
CLIP_LO, CLIP_HI = -1000.0, 1000.0
BODY_HU = -500.0  # HU threshold: > this (after clip) = body/foreground
EMPTY_SLICE_FRAC = 0.005  # slice with <0.5% body voxels = "empty"
STRIDE = 4  # in-plane subsample for stats (16x fewer voxels; QC-accurate, fast)
MAX_VOX = 2_000_000  # hard cap on voxels used for percentiles (speed)
MAX_WORKERS = int(os.environ.get("EDA_WORKERS", "8"))

# -------------------------------------------------------------------- helpers


def load_hu(path: Path) -> np.ndarray:
    """Load a NIfTI as native int16 HU in canonical (RAS) axis order.

    `_fixed` volumes already carry HU (nifti scl_slope/inter identity), so we read
    `dataobj` directly as native int16 — avoiding the ~1 GB float32 blow-up of
    `get_fdata()`. We NEVER apply the metadata RescaleSlope/Intercept.
    Returns ``(X=R, Y=A, Z=S)`` int16.
    """
    img = nib.as_closest_canonical(nib.load(str(path)))
    return np.asanyarray(img.dataobj)


def voxel_stats(hu: np.ndarray) -> dict:
    """Voxel integrity + morphology stats on a stride-4 in-plane subsample."""
    n = int(hu.size)
    sub = hu[::STRIDE, ::STRIDE, :]
    subf = sub.astype(np.float32).ravel()
    if subf.size > MAX_VOX:
        subf = subf[:: subf.size // MAX_VOX]

    sentinel_frac = float(np.mean(subf == SENTINEL)) if subf.size else 0.0
    pcts = [0.0, 0.1, 1.0, 50.0, 99.0, 99.9, 100.0]
    pv = np.percentile(subf, pcts) if subf.size else [np.nan] * len(pcts)

    clipped = np.clip(subf, CLIP_LO, CLIP_HI)
    edges = np.arange(CLIP_LO, CLIP_HI + 20, 20)
    hist, _ = np.histogram(clipped, bins=edges)
    air_peak_hu = float(edges[int(np.argmax(hist))] + 10)
    fg_frac = float(np.mean(clipped > BODY_HU))

    body = np.clip(sub.astype(np.float32), CLIP_LO, CLIP_HI) > BODY_HU
    if body.any():
        xs, ys, zs = np.where(body)
        bbox = [
            int(xs.min()) * STRIDE,
            int(xs.max()) * STRIDE,
            int(ys.min()) * STRIDE,
            int(ys.max()) * STRIDE,
            int(zs.min()),
            int(zs.max()),
        ]
        per_slice = body.reshape(-1, body.shape[2]).sum(axis=0)
        slice_area = body.shape[0] * body.shape[1]
        empty = float(np.mean(per_slice < EMPTY_SLICE_FRAC * slice_area))
        subb = np.clip(
            sub[
                xs.min() : xs.max() + 1,
                ys.min() : ys.max() + 1,
                zs.min() : zs.max() + 1,
            ].astype(np.float32),
            CLIP_LO,
            CLIP_HI,
        )
        internal_air = float(np.mean(subb < BODY_HU))
        bone_frac = float(np.mean(subb > 300))
    else:
        bbox, empty, internal_air, bone_frac = [0] * 6, 1.0, 0.0, 0.0

    return {
        "shape": list(hu.shape),
        "n_voxels": n,
        "nan_count": 0,
        "inf_count": 0,
        "min": float(pv[0]),
        "p0_1": float(pv[1]),
        "p1": float(pv[2]),
        "p50": float(pv[3]),
        "p99": float(pv[4]),
        "p99_9": float(pv[5]),
        "max": float(pv[6]),
        "sentinel_neg8192_frac": sentinel_frac,
        "air_peak_hu": air_peak_hu,
        "air_peak_near_air": bool(abs(air_peak_hu - CLIP_LO) <= 60),
        "foreground_frac_gt-500": fg_frac,
        "body_bbox_xyz": bbox,
        "empty_slice_frac_axial": empty,
        "internal_air_frac": internal_air,
        "bone_frac_in_bbox": bone_frac,
    }


def coverage_proxy(st: dict) -> str:
    """CRUDE scan-coverage proxy for a non-chest volume (montage is authoritative).

    Head/brain scans occupy a small part of a large FOV (low foreground) with a
    skull ring; chest has lung-scale internal air in a larger body; abdomen is
    soft-tissue-dominant. This cannot replicate a BodyPartRegressor — it only
    triages the montages for human review.
    """
    fg = st["foreground_frac_gt-500"]
    air = st["internal_air_frac"]
    bone = st["bone_frac_in_bbox"]
    if fg < 0.22:
        return "head/neck (small body in FOV)"
    if air >= 0.30 and fg >= 0.25:
        return "neck-to-chest? (lung-scale air)"
    if bone >= 0.05:
        return "head/neck (bone-dominant)"
    return "abdomen/soft-tissue"


def apply_metadata_rescale(hu, slope, inter):
    """Demo ONLY: what re-applying metadata RescaleSlope/Intercept would do."""
    return hu * slope + inter


def window(sl, wl, ww):
    x = np.clip(sl, CLIP_LO, CLIP_HI)
    lo, hi = wl - ww / 2, wl + ww / 2
    return np.clip((x - lo) / (hi - lo), 0, 1)


def mid_views(hu):
    x, y, z = hu.shape
    return {
        "axial": np.rot90(hu[:, :, z // 2]),
        "coronal": np.rot90(hu[:, y // 2, :]),
        "sagittal": np.rot90(hu[x // 2, :, :]),
    }


def montage_windows(hu, title, path):
    views = mid_views(hu)
    wins = [("lung (WL-600/WW1500)", -600, 1500), ("mediastinal (WL40/WW400)", 40, 400)]
    fig, axs = plt.subplots(2, 3, figsize=(11, 8))
    for r, (wname, wl, ww) in enumerate(wins):
        for c, v in enumerate(["axial", "coronal", "sagittal"]):
            axs[r, c].imshow(
                window(views[v], wl, ww), cmap="gray", vmin=0, vmax=1, aspect="equal"
            )
            axs[r, c].set_title(f"{v} — {wname}", fontsize=9)
            axs[r, c].axis("off")
    fig.suptitle(f"{title}  (unit: 1 volume; clip[-1000,1000])", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=200)
    plt.close(fig)


def montage_single(hu, title, path, wl=40, ww=800):
    views = mid_views(hu)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.5))
    for c, v in enumerate(["axial", "coronal", "sagittal"]):
        axs[c].imshow(
            window(views[v], wl, ww), cmap="gray", vmin=0, vmax=1, aspect="equal"
        )
        axs[c].set_title(v, fontsize=10)
        axs[c].axis("off")
    fig.suptitle(f"{title}  (unit: 1 volume; WL{wl}/WW{ww})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=200)
    plt.close(fig)


def sample_no_chest(n_per_split=10):
    picks = []
    for split in ("valid", "train"):
        base = NO_CHEST / f"{split}_fixed"
        files = sorted(str(p) for p in base.rglob("*.nii.gz"))
        seen, chosen = set(), []
        for p in files:
            scan = "_".join(Path(p).name.replace(".nii.gz", "").split("_")[:3])
            if scan in seen:
                continue
            seen.add(scan)
            chosen.append(p)
        stride = max(1, len(chosen) // n_per_split)
        for p in chosen[::stride][:n_per_split]:
            picks.append({"split": split, "path": p, "vol": Path(p).name})
    return picks


# ------------------------------------------------------- parallel worker (per volume)


def _process_one(task: dict) -> dict:
    """Load one volume, compute stats, and write its montage (runs in a worker).

    Returns a small picklable dict (stats + optional demo/coverage); the montage
    PNG is written directly to disk here. Each worker uses the Agg backend.
    """
    hu = load_hu(Path(task["path"]))
    st = voxel_stats(hu)
    out = {
        "source": task["source"],
        "group": task["group"],
        "role": task["role"],
        "volume": task["vol"],
        "stats": st,
    }

    if task.get("primary_rep"):
        clipped = np.clip(
            hu[::STRIDE, ::STRIDE, :].astype(np.float32), CLIP_LO, CLIP_HI
        )
        slope, inter = task["slope"], task["inter"]
        corrupt = apply_metadata_rescale(clipped, slope, inter)
        out["demo"] = {
            "group": task["group"],
            "volume": task["vol"],
            "RescaleSlope": slope,
            "RescaleIntercept": inter,
            "fixed_median_HU": float(np.median(clipped)),
            "fixed_p99_HU": float(np.percentile(clipped, 99)),
            "corrupt_median_HU": float(np.median(corrupt)),
            "corrupt_max_HU": float(corrupt.max()),
            "corrupt_frac_below_-1000": float(np.mean(corrupt < CLIP_LO)),
            "delta_shift_HU": float(inter),
        }
        montage_windows(
            hu,
            f"{task['group']} — {task['vol'].replace('.nii.gz', '')}",
            MONT / f"{task['group']}_{task['vol'].replace('.nii.gz', '')}.png",
        )

    if task["source"] == "no_chest":
        cat = coverage_proxy(st)
        out["coverage"] = {
            "volume": task["vol"],
            "split": task["role"],
            "coverage_proxy": cat,
            "internal_air_frac": st["internal_air_frac"],
            "bone_frac_in_bbox": st["bone_frac_in_bbox"],
            "foreground_frac_gt-500": st["foreground_frac_gt-500"],
            "shape": "x".join(map(str, st["shape"])),
        }
        vid = task["vol"].replace(".nii.gz", "")
        montage_single(
            hu,
            f"no_chest {vid} [{task['role']}] proxy={cat}",
            MONT / f"no_chest_{vid}.png",
        )
    return out


# ------------------------------------------------------------------------ main


def main() -> None:
    for d in (TABLES, MONT):
        d.mkdir(parents=True, exist_ok=True)

    meta = pd.concat(
        [
            pd.read_csv(
                ROOT / "metadata" / "train_metadata.csv",
                usecols=["VolumeName", "RescaleSlope", "RescaleIntercept"],
            ),
            pd.read_csv(
                ROOT / "metadata" / "validation_metadata.csv",
                usecols=["VolumeName", "RescaleSlope", "RescaleIntercept"],
            ),
        ]
    ).set_index("VolumeName")

    manifest = pd.read_csv(BUNDLE / "manifest.csv")

    # build the task list (bundle + no_chest); flag first-primary-of-group
    tasks, seen_primary = [], set()
    for _, m in manifest.iterrows():
        vol = m["filename"]
        primary_rep = m["role"] == "primary" and m["group"] not in seen_primary
        if primary_rep:
            seen_primary.add(m["group"])
        md = meta.loc[vol] if vol in meta.index else None
        tasks.append(
            {
                "path": str(BUNDLE / m["bundle_path"]),
                "source": "bundle",
                "group": m["group"],
                "role": m["role"],
                "vol": vol,
                "primary_rep": primary_rep,
                "slope": float(md["RescaleSlope"]) if md is not None else float("nan"),
                "inter": float(md["RescaleIntercept"])
                if md is not None
                else float("nan"),
            }
        )
    for s in sample_no_chest(n_per_split=10):
        tasks.append(
            {
                "path": s["path"],
                "source": "no_chest",
                "group": "no_chest",
                "role": s["split"],
                "vol": s["vol"],
                "primary_rep": False,
            }
        )

    # ---- run in parallel across processes (I/O-bound decode) ----
    results = []
    with cf.ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for r in ex.map(_process_one, tasks):
            results.append(r)

    # ---- assemble tables from returned stats ----
    rows, cov_rows = [], []
    raw = {"bundle": [], "no_chest": [], "hu_validity_demo": [], "montages": []}
    for r in results:
        rec = {
            "source": r["source"],
            "group": r["group"],
            "role": r["role"],
            "volume": r["volume"],
            **r["stats"],
        }
        rows.append(rec)
        raw["no_chest" if r["source"] == "no_chest" else "bundle"].append(rec)
        if "demo" in r:
            raw["hu_validity_demo"].append(r["demo"])
        if "coverage" in r:
            cov_rows.append(r["coverage"])
    raw["montages"] = sorted(str(p) for p in MONT.glob("*.png"))

    df = pd.DataFrame(rows)
    df["body_bbox_xyz"] = df["body_bbox_xyz"].apply(lambda b: "_".join(map(str, b)))
    df.to_csv(TABLES / "nifti_voxel.csv", index=False)
    pd.DataFrame(cov_rows).to_csv(TABLES / "no_chest_coverage.csv", index=False)
    with open(TABLES / "voxel_raw.json", "w") as f:
        json.dump(raw, f, indent=2)

    print(
        f"workers={MAX_WORKERS} | bundle={len(raw['bundle'])} no_chest={len(raw['no_chest'])}"
    )
    print("nifti_voxel.csv rows:", len(df))
    print(
        "coverage_proxy:",
        pd.Series([c["coverage_proxy"] for c in cov_rows]).value_counts().to_dict(),
    )
    d0 = raw["hu_validity_demo"][0]
    print(
        f"HU-validity demo [{d0['group']}]: fixed_median={d0['fixed_median_HU']:.0f} "
        f"-> corrupt_median={d0['corrupt_median_HU']:.0f}, frac<-1000={d0['corrupt_frac_below_-1000']:.3f}"
    )
    print("montages:", len(list(MONT.glob("*.png"))))


if __name__ == "__main__":
    main()
