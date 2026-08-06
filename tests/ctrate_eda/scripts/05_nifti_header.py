#!/usr/bin/env python
"""05_nifti_header - NIfTI header QC (GPT EDA extension, section 6.1).

Header-ONLY QC (never loads voxel arrays) for the full on-disk valid_fixed set
(3,001 clean volumes) plus the 30 curated bundle files, with an optional
stratified train_fixed header scan (--sample-train N).

Per-file: split / patient / scan / recon ids, path, size, header shape, dtype,
zooms, axcodes, affine determinant, qform/sform codes, ndim, read_error, no_chest
flag, and metadata-vs-header spacing/shape diffs. Flags corrupt reads, non-3D
volumes, non-positive spacing, non-LPS/RAS orientation, near-singular affines,
and metadata/header mismatches.

Self-contained + re-runnable. Read-only on datasets.
"""

import sys

sys.path.insert(0, "/workspace")

import argparse
import json
import os
import re
import random
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib

DATA_ROOT = Path("/workspace/datasets/datasets/CT-RATE/dataset")
VALID_DIR = DATA_ROOT / "valid_fixed"
TRAIN_DIR = DATA_ROOT / "train_fixed"
META_DIR = DATA_ROOT / "metadata"
BUNDLE_FILES_DIR = Path("/workspace/tests/ctrate_eda_bundle/files")
BUNDLE_MANIFEST = Path("/workspace/tests/ctrate_eda_bundle/manifest.csv")

OUT = Path("/workspace/tests/ctrate_eda")
TBL = OUT / "tables"
FIG = OUT / "figures"
for d in (TBL, FIG):
    d.mkdir(parents=True, exist_ok=True)

FNAME_RE = re.compile(r"^(train|valid)_(\d+)_([a-z]+)_(\d+)$")


def parse_name(volname: str):
    """Split a CT-RATE stem (train_1_a_2) into split/patient/scan/recon ids."""
    stem = volname
    for suf in (".nii.gz", ".nii"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
    m = FNAME_RE.match(stem)
    if not m:
        return None, None, None, None
    split, pid, scan, recon = m.groups()
    return split, f"{split}_{pid}", f"{split}_{pid}_{scan}", int(recon)


def load_no_chest():
    """Return set of no_chest basenames (VolumeName incl .nii.gz) for both splits."""
    s = set()
    for fn in ("no_chest_train.txt", "no_chest_valid.txt"):
        p = META_DIR / fn
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            line = line.strip()
            if line:
                s.add(os.path.basename(line))
    return s


def load_metadata():
    """Build VolumeName -> (spacing_triple, shape_triple) from the two metadata CSVs."""
    lut = {}
    for fn in ("train_metadata.csv", "validation_metadata.csv"):
        p = META_DIR / fn
        if not p.exists():
            continue
        m = pd.read_csv(p)
        for _, r in m.iterrows():
            xy = r.get("XYSpacing")
            try:
                xy_vals = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+", str(xy))]
            except Exception:
                xy_vals = []
            xspc = xy_vals[0] if len(xy_vals) >= 1 else np.nan
            yspc = xy_vals[1] if len(xy_vals) >= 2 else xspc
            zspc = float(r["ZSpacing"]) if not pd.isna(r.get("ZSpacing")) else np.nan
            spacing = (xspc, yspc, zspc)
            try:
                shape = (int(r["Rows"]), int(r["Columns"]), int(r["NumberofSlices"]))
            except Exception:
                shape = (np.nan, np.nan, np.nan)
            lut[str(r["VolumeName"])] = (spacing, shape)
    return lut


def sorted_maxdiff(a, b):
    """Max abs diff of the two triples after sorting each (orientation-robust)."""
    try:
        a = np.sort(np.asarray(a, dtype=float))
        b = np.sort(np.asarray(b, dtype=float))
        if np.any(np.isnan(a)) or np.any(np.isnan(b)):
            return np.nan
        return float(np.max(np.abs(a - b)))
    except Exception:
        return np.nan


def scan_file(path: Path, no_chest_set, meta_lut):
    """Read one NIfTI header (no voxels) and return the QC record dict."""
    volname = path.name
    split, pid, scan, recon = parse_name(volname)
    rec = {
        "split": split,
        "patient_id": pid,
        "scan_id": scan,
        "recon_id": recon,
        "volume_name": volname,
        "absolute_path": str(path),
        "file_size": path.stat().st_size if path.exists() else np.nan,
        "shape": None,
        "dtype": None,
        "zooms": None,
        "axcodes": None,
        "affine_determinant": np.nan,
        "qform_code": np.nan,
        "sform_code": np.nan,
        "ndim": np.nan,
        "read_error": False,
        "no_chest": volname in no_chest_set,
        "metadata_spacing_vs_header_zoom_diff": np.nan,
        "metadata_shape_vs_header_shape_diff": np.nan,
    }
    try:
        img = nib.load(str(path))
        hdr = img.header
        shape = tuple(int(x) for x in hdr.get_data_shape())
        zooms = tuple(float(z) for z in hdr.get_zooms())
        aff = img.affine
        rec["shape"] = list(shape)
        rec["dtype"] = str(hdr.get_data_dtype())
        rec["zooms"] = list(zooms)
        rec["axcodes"] = "".join(nib.aff2axcodes(aff))
        rec["affine_determinant"] = float(np.linalg.det(aff[:3, :3]))
        rec["qform_code"] = int(hdr["qform_code"])
        rec["sform_code"] = int(hdr["sform_code"])
        rec["ndim"] = len(shape)
        # metadata comparison (spatial dims only)
        if volname in meta_lut:
            mspc, mshape = meta_lut[volname]
            zt = zooms[:3] if len(zooms) >= 3 else zooms
            st = shape[:3] if len(shape) >= 3 else shape
            if len(zt) == 3:
                rec["metadata_spacing_vs_header_zoom_diff"] = sorted_maxdiff(mspc, zt)
            if len(st) == 3:
                rec["metadata_shape_vs_header_shape_diff"] = sorted_maxdiff(mshape, st)
    except Exception as e:
        rec["read_error"] = True
        rec["read_error_msg"] = str(e)[:200]
    return rec


def flag_record(rec):
    """Return list of QC flag strings for a record (empty if clean)."""
    flags = []
    if rec["read_error"]:
        flags.append("read_failure")
        return flags
    if rec["ndim"] != 3:
        flags.append("non_3d")
    z = rec["zooms"] or []
    if any((zz is None) or (zz <= 0) for zz in z[:3]):
        flags.append("nonpositive_spacing")
    ax = rec["axcodes"] or ""
    if set(ax) - set("LRAPSI") or len(ax) != 3:
        flags.append("abnormal_orientation")
    else:
        # flag orientations that are neither canonical LPS nor RAS
        if ax not in ("LPS", "RAS"):
            flags.append("nonstandard_orientation")
    det = rec["affine_determinant"]
    if det is None or (isinstance(det, float) and (np.isnan(det) or abs(det) < 1e-3)):
        flags.append("near_singular_affine")
    sd = rec["metadata_spacing_vs_header_zoom_diff"]
    if sd is not None and not (isinstance(sd, float) and np.isnan(sd)) and sd > 1e-2:
        flags.append("spacing_metadata_mismatch")
    shd = rec["metadata_shape_vs_header_shape_diff"]
    if shd is not None and not (isinstance(shd, float) and np.isnan(shd)) and shd > 0:
        flags.append("shape_metadata_mismatch")
    return flags


def stratified_train_paths(n, seed=0):
    """Pick ~n train_fixed volumes stratified by ZSpacing bins (fallback: uniform)."""
    all_paths = sorted(TRAIN_DIR.rglob("*.nii.gz"))
    if n <= 0 or n >= len(all_paths):
        return all_paths if n >= len(all_paths) else []
    meta = load_metadata()
    # stratify by z-spacing quantile bins
    buckets = {}
    for p in all_paths:
        _, mshape = (
            meta.get(p.name, ((np.nan, np.nan, np.nan), None))
            if False
            else (None, None)
        )
        z = (
            meta.get(p.name, ((np.nan, np.nan, np.nan),))[0][2]
            if p.name in meta
            else np.nan
        )
        key = "nan" if (isinstance(z, float) and np.isnan(z)) else round(float(z), 1)
        buckets.setdefault(key, []).append(p)
    rng = random.Random(seed)
    picked = []
    keys = list(buckets.keys())
    per = max(1, n // max(1, len(keys)))
    for k in keys:
        b = buckets[k]
        rng.shuffle(b)
        picked.extend(b[:per])
    rng.shuffle(picked)
    return picked[:n]


def make_figure(df):
    """Zoom & shape distributions colored by split; saved to figures/nifti_spacing_shape.png."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = df[~df["read_error"]].copy()
    zx, zy, zz = [], [], []
    sz = []
    splits = []
    for _, r in ok.iterrows():
        z = r["zooms"]
        s = r["shape"]
        if isinstance(z, list) and len(z) >= 3 and isinstance(s, list) and len(s) >= 3:
            zx.append(z[0])
            zy.append(z[1])
            zz.append(z[2])
            sz.append(s[2])
            splits.append(r["split"])
    zx = np.array(zx)
    zy = np.array(zy)
    zz = np.array(zz)
    sz = np.array(sz)
    splits = np.array(splits)
    color = {"valid": "tab:orange", "train": "tab:blue"}
    cvec = np.array([color.get(s, "tab:green") for s in splits])

    fig, axs = plt.subplots(2, 2, figsize=(13, 10))
    # in-plane spacing histogram
    ax = axs[0, 0]
    for sp in ("valid", "train"):
        mask = splits == sp
        if mask.sum():
            ax.hist(
                zx[mask],
                bins=40,
                alpha=0.6,
                color=color[sp],
                label=f"{sp} (n={mask.sum()})",
            )
    ax.set_xlabel("in-plane zoom X (mm)")
    ax.set_ylabel("count")
    ax.set_title("Header in-plane spacing (per volume)")
    ax.legend()
    # z spacing histogram
    ax = axs[0, 1]
    for sp in ("valid", "train"):
        mask = splits == sp
        if mask.sum():
            ax.hist(
                zz[mask],
                bins=40,
                alpha=0.6,
                color=color[sp],
                label=f"{sp} (n={mask.sum()})",
            )
    ax.set_xlabel("z zoom (mm)")
    ax.set_ylabel("count")
    ax.set_title("Header slice spacing (per volume)")
    ax.legend()
    # in-plane vs z spacing scatter
    ax = axs[1, 0]
    ax.scatter(zx, zz, s=6, c=cvec, alpha=0.4)
    ax.set_xlabel("in-plane zoom X (mm)")
    ax.set_ylabel("z zoom (mm)")
    ax.set_title("Spacing scatter (per volume; blue=train, orange=valid)")
    # n_slices histogram
    ax = axs[1, 1]
    for sp in ("valid", "train"):
        mask = splits == sp
        if mask.sum():
            ax.hist(
                sz[mask],
                bins=40,
                alpha=0.6,
                color=color[sp],
                label=f"{sp} (n={mask.sum()})",
            )
    ax.set_xlabel("header shape[2] (slices)")
    ax.set_ylabel("count")
    ax.set_title("Header n-slices (per volume)")
    ax.legend()
    fig.suptitle(
        "CT-RATE _fixed NIfTI header QC - zooms & shapes (analysis unit: volume)",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG / "nifti_spacing_shape.png", dpi=220)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sample-train",
        type=int,
        default=0,
        help="stratified N of train_fixed to header-scan (0 = skip)",
    )
    args = ap.parse_args()

    no_chest_set = load_no_chest()
    meta_lut = load_metadata()

    records = []

    # 1) full valid_fixed
    valid_paths = sorted(VALID_DIR.rglob("*.nii.gz"))
    print(f"[valid_fixed] scanning {len(valid_paths)} volumes ...", flush=True)
    for i, p in enumerate(valid_paths):
        records.append(scan_file(p, no_chest_set, meta_lut))
        if (i + 1) % 500 == 0:
            print(f"  valid {i + 1}/{len(valid_paths)}", flush=True)

    # 2) 30 bundle files
    bundle_paths = (
        sorted(BUNDLE_FILES_DIR.rglob("*.nii.gz")) if BUNDLE_FILES_DIR.exists() else []
    )
    print(f"[bundle] scanning {len(bundle_paths)} volumes ...", flush=True)
    for p in bundle_paths:
        rec = scan_file(p, no_chest_set, meta_lut)
        rec["source"] = "bundle"
        records.append(rec)

    # 3) optional stratified train sample
    if args.sample_train > 0:
        tpaths = stratified_train_paths(args.sample_train)
        print(
            f"[train_fixed] scanning {len(tpaths)} stratified volumes ...", flush=True
        )
        for i, p in enumerate(tpaths):
            records.append(scan_file(p, no_chest_set, meta_lut))
            if (i + 1) % 500 == 0:
                print(f"  train {i + 1}/{len(tpaths)}", flush=True)

    for r in records:
        r.setdefault("source", "on_disk")
        r["flags"] = ";".join(flag_record(r))

    df = pd.DataFrame(records)

    # serialize list cols as JSON strings for CSV
    df_csv = df.copy()
    for col in ("shape", "zooms"):
        df_csv[col] = df_csv[col].apply(
            lambda v: json.dumps(v) if isinstance(v, list) else v
        )
    csv_cols = [
        "split",
        "patient_id",
        "scan_id",
        "recon_id",
        "volume_name",
        "source",
        "absolute_path",
        "file_size",
        "shape",
        "dtype",
        "zooms",
        "axcodes",
        "affine_determinant",
        "qform_code",
        "sform_code",
        "ndim",
        "read_error",
        "no_chest",
        "metadata_spacing_vs_header_zoom_diff",
        "metadata_shape_vs_header_shape_diff",
        "flags",
    ]
    csv_cols = [c for c in csv_cols if c in df_csv.columns]
    df_csv[csv_cols].to_csv(TBL / "nifti_qc.csv", index=False)

    flagged = df_csv[df_csv["flags"].astype(bool) & (df_csv["flags"] != "")]
    flagged[
        csv_cols + (["read_error_msg"] if "read_error_msg" in df_csv.columns else [])
    ].to_csv(TBL / "nifti_qc_flags.csv", index=False)

    make_figure(df)

    # raw summary json
    def flag_counts(sub):
        c = {}
        for f in sub["flags"]:
            for tok in f.split(";") if f else []:
                if tok:
                    c[tok] = c.get(tok, 0) + 1
        return c

    valid_df = df[df["split"] == "valid"]
    ondisk_valid = valid_df[valid_df["source"] == "on_disk"]
    summary = {
        "n_files_total": int(len(df)),
        "n_valid_on_disk": int(len(ondisk_valid)),
        "n_bundle": int((df["source"] == "bundle").sum()),
        "n_train_sampled": int(
            ((df["split"] == "train") & (df["source"] == "on_disk")).sum()
        ),
        "n_read_error": int(df["read_error"].sum()),
        "n_no_chest_on_disk": int(df["no_chest"].sum()),
        "n_flagged": int(len(flagged)),
        "flag_counts_all": flag_counts(df),
        "flag_counts_valid_on_disk": flag_counts(ondisk_valid),
        "dtype_counts": {str(k): int(v) for k, v in df["dtype"].value_counts().items()},
        "axcodes_counts": {
            str(k): int(v) for k, v in df["axcodes"].value_counts().items()
        },
        "ndim_counts": {
            str(k): int(v) for k, v in df["ndim"].value_counts(dropna=False).items()
        },
        "qform_code_counts": {
            str(k): int(v)
            for k, v in df["qform_code"].value_counts(dropna=False).items()
        },
        "sform_code_counts": {
            str(k): int(v)
            for k, v in df["sform_code"].value_counts(dropna=False).items()
        },
        "affine_det_stats": {
            "min": float(np.nanmin(df["affine_determinant"])),
            "max": float(np.nanmax(df["affine_determinant"])),
            "mean": float(np.nanmean(df["affine_determinant"])),
            "n_negative": int((df["affine_determinant"] < 0).sum()),
        },
        "spacing_metadata_diff_stats": {
            "max": float(np.nanmax(df["metadata_spacing_vs_header_zoom_diff"])),
            "n_gt_1e-2": int((df["metadata_spacing_vs_header_zoom_diff"] > 1e-2).sum()),
            "n_compared": int(df["metadata_spacing_vs_header_zoom_diff"].notna().sum()),
        },
        "shape_metadata_diff_stats": {
            "max": float(np.nanmax(df["metadata_shape_vs_header_shape_diff"])),
            "n_gt_0": int((df["metadata_shape_vs_header_shape_diff"] > 0).sum()),
            "n_compared": int(df["metadata_shape_vs_header_shape_diff"].notna().sum()),
        },
        "zoom_x_stats_on_disk_valid": {
            "min": float(
                ondisk_valid["metadata_spacing_vs_header_zoom_diff"].notna().sum()
            ),
        },
    }
    # zoom summary from actual header zooms (valid on-disk)
    zx = [z[0] for z in ondisk_valid["zooms"] if isinstance(z, list) and len(z) >= 3]
    zz = [z[2] for z in ondisk_valid["zooms"] if isinstance(z, list) and len(z) >= 3]
    if zx:
        summary["header_zoom_valid_on_disk"] = {
            "in_plane_x": {
                "min": float(np.min(zx)),
                "median": float(np.median(zx)),
                "max": float(np.max(zx)),
            },
            "z": {
                "min": float(np.min(zz)),
                "median": float(np.median(zz)),
                "max": float(np.max(zz)),
            },
        }
    with open(TBL / "nifti_header_raw.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote: {TBL / 'nifti_qc.csv'} ({len(df)} rows)")
    print(f"Wrote: {TBL / 'nifti_qc_flags.csv'} ({len(flagged)} rows)")
    print(f"Wrote: {FIG / 'nifti_spacing_shape.png'}")
    print(f"Wrote: {TBL / 'nifti_header_raw.json'}")


if __name__ == "__main__":
    main()
