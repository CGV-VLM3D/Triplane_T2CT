"""EDA v2 — Module 08: reconstruction (recon-1 vs recon-2) comparison (GPT §6.4).

A CT *scan* is reconstructed multiple times from the same raw projection data with
different convolution kernels / slice geometries. Report and 18 abnormality labels
are constant across a scan's reconstructions — they differ only in kernel family
(sharp/lung vs soft/mediastinal), slice thickness, and in-plane matrix. This module
takes the 5 recon-1 / recon-2 pairs vendored in the EDA bundle and quantifies those
differences: convolution kernel, series description, geometry, noise, edge sharpness,
and intensity percentiles, plus a resample-aware lung-window montage at matched
anatomical (fractional-depth) levels.

Pairs (all from valid split):
  all-zero          valid_1000_a  (Philips iCT 256)
  lung-nodule-only  valid_1001_a  (Philips Brilliance Big Bore)
  diffuse-low-burden valid_1022_a (Philips Brilliance Big Bore)
  multi-abnormality valid_1016_b  (SIEMENS SOMATOM Force)  <- identical geometry pair
  medical-material  valid_1288_a  (Philips iCT 256)

KEY QUESTION (task): does recon index map consistently to a kernel family? Answer is
computed empirically (noise + edge) and cross-checked against ConvolutionKernel.

valid_1016_b flag: recon-1 and recon-2 have IDENTICAL geometry (512x512x209, same
spacing) — a matched-geometry pair. We verify voxel-wise whether it is a true
duplicate or a same-geometry sharp/soft kernel pair.

Read-only on /workspace/datasets. HU already baked into _fixed NIfTI — NEVER reapply
RescaleSlope/Intercept. -8192 sentinel = out-of-FOV padding, clipped to [-1000,1000].

Outputs (under /workspace/tests/ctrate_eda/):
  tables/recon_compare.csv  — one row per (pair, recon) with all metrics
  tables/recon_raw.json     — full pre-rounding machine-readable digest
  figures/recon_compare.png — matched-level lung montages + noise/edge bars
"""

from __future__ import annotations

import sys

sys.path.insert(0, "/workspace")

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

BUNDLE = Path("/workspace/tests/ctrate_eda_bundle")
META_CSV = Path(
    "/workspace/datasets/datasets/CT-RATE/dataset/metadata/validation_metadata.csv"
)
OUT_T = Path("/workspace/tests/ctrate_eda/tables")
OUT_F = Path("/workspace/tests/ctrate_eda/figures")
OUT_T.mkdir(parents=True, exist_ok=True)
OUT_F.mkdir(parents=True, exist_ok=True)

SENTINEL = -8192
HU_CLIP = (-1000.0, 1000.0)

# (group, scan_key, manufacturer/model, clinical group label)
PAIRS = [
    ("all-zero", "valid_1000_a", "Philips iCT 256"),
    ("lung-nodule-only", "valid_1001_a", "Philips Brilliance Big Bore"),
    ("diffuse-low-burden", "valid_1022_a", "Philips Brilliance Big Bore"),
    ("multi-abnormality", "valid_1016_b", "SIEMENS SOMATOM Force"),
    ("medical-material", "valid_1288_a", "Philips iCT 256"),
]


def load_hu(path: Path) -> np.ndarray:
    """Load a _fixed CT volume as float32 HU, sentinel->clip. Returns ``(X, Y, Z)``."""
    vol = np.asanyarray(nib.load(str(path)).dataobj).astype(np.float32)
    vol[vol <= SENTINEL] = HU_CLIP[0]  # out-of-FOV padding -> air
    return np.clip(vol, HU_CLIP[0], HU_CLIP[1])


def noise_soft_tissue(vol: np.ndarray) -> tuple[float, int]:
    """High-frequency residual noise std in a flat soft-tissue ROI.

    Soft-tissue mask = HU in [-50, 120] (muscle/blood/organ), eroded to interior.
    Residual = vol - gaussian(vol, sigma=2); noise = std of residual in mask.

    Returns (noise_std_HU, n_voxels_in_mask).
    """
    mask = (vol >= -50) & (vol <= 120)  # (X, Y, Z) soft tissue
    mask = ndimage.binary_erosion(mask, iterations=2)  # avoid edge voxels
    if mask.sum() < 500:
        return float("nan"), int(mask.sum())
    resid = vol - ndimage.gaussian_filter(vol, sigma=2.0)  # high-pass
    return float(resid[mask].std()), int(mask.sum())


def edge_sharpness(vol: np.ndarray) -> tuple[float, float]:
    """Edge/detail energy over the body region (HU > -500).

    Returns (laplacian_variance, mean_gradient_magnitude) in HU units.
    Sharp (lung/bone) kernels produce markedly larger values than soft kernels.
    """
    body = vol > -500  # (X, Y, Z) exclude air
    lap = ndimage.laplace(vol)  # (X, Y, Z)
    lap_var = float(lap[body].var()) if body.sum() else float("nan")
    gx, gy, gz = np.gradient(vol)  # each (X, Y, Z)
    gmag = np.sqrt(gx * gx + gy * gy + gz * gz)
    grad_mean = float(gmag[body].mean()) if body.sum() else float("nan")
    return lap_var, grad_mean


def pctiles(vol: np.ndarray) -> dict:
    """Intensity percentiles over the body region (HU > -900)."""
    body = vol[vol > -900]
    p = np.percentile(body, [1, 5, 50, 95, 99])
    return {
        "p01": float(p[0]),
        "p05": float(p[1]),
        "p50": float(p[2]),
        "p95": float(p[3]),
        "p99": float(p[4]),
    }


def lung_slice(vol: np.ndarray, frac: float) -> np.ndarray:
    """Axial slice at fractional depth `frac`, lung window WL=-600 WW=1500 -> [0,1].

    Returns a display-normalized 2D array ``(X, Y)``. Fractional-depth indexing gives
    resample-aware matched anatomical levels across differing slice counts.
    """
    z = int(round(frac * (vol.shape[2] - 1)))
    sl = vol[:, :, z]
    lo, hi = -600 - 1500 / 2, -600 + 1500 / 2  # [-1350, 150]
    return np.clip((sl - lo) / (hi - lo), 0, 1)


def main() -> None:
    meta = pd.read_csv(META_CSV)
    meta["stem"] = meta["VolumeName"].str.replace(".nii.gz", "", regex=False)
    meta = meta.set_index("stem")

    rows: list[dict] = []
    raw: dict = {}
    montage_data: dict = {}  # scan_key -> list of (recon, [slice035, slice05, slice065])

    for group, scan, model in PAIRS:
        raw[scan] = {"group": group, "model": model, "recons": {}}
        montage_data[scan] = []
        for recon in (1, 2):
            stem = f"{scan}_{recon}"
            path = BUNDLE / "files" / group / f"{stem}.nii.gz"
            vol = load_hu(path)  # (X, Y, Z)
            X, Y, Z = vol.shape
            m = meta.loc[stem]
            kernel = str(m["ConvolutionKernel"])
            series = str(m["SeriesDescription"])
            xy_raw = m["XYSpacing"]  # e.g. "[0.3935, 0.3935]" or a bare float
            xy = (
                float(str(xy_raw).strip("[]").split(",")[0])
                if not pd.isna(xy_raw)
                else float("nan")
            )
            zsp = float(m["ZSpacing"]) if not pd.isna(m["ZSpacing"]) else float("nan")

            noise, nvox = noise_soft_tissue(vol)
            lap_var, grad_mean = edge_sharpness(vol)
            pc = pctiles(vol)

            row = {
                "group": group,
                "scan_key": scan,
                "manufacturer_model": model,
                "recon": recon,
                "convolution_kernel": kernel,
                "series_description": series,
                "shape": f"{X}x{Y}x{Z}",
                "n_slices": Z,
                "in_plane": f"{X}x{Y}",
                "xy_spacing_mm": xy,
                "z_spacing_mm": zsp,
                "noise_softtissue_HU": noise,
                "noise_roi_voxels": nvox,
                "laplacian_var": lap_var,
                "grad_mag_mean": grad_mean,
                **pc,
            }
            rows.append(row)
            raw[scan]["recons"][recon] = row
            montage_data[scan].append(
                (recon, [lung_slice(vol, f) for f in (0.35, 0.5, 0.65)])
            )

        # per-pair derived: which recon is empirically sharper + geometry-identical flag
        r1, r2 = raw[scan]["recons"][1], raw[scan]["recons"][2]
        sharper = 1 if r1["laplacian_var"] >= r2["laplacian_var"] else 2
        geom_identical = (
            r1["shape"] == r2["shape"]
            and abs(r1["xy_spacing_mm"] - r2["xy_spacing_mm"]) < 1e-4
            and abs(r1["z_spacing_mm"] - r2["z_spacing_mm"]) < 1e-4
        )
        raw[scan]["empirically_sharper_recon"] = sharper
        raw[scan]["geometry_identical"] = bool(geom_identical)
        if geom_identical:
            # matched geometry -> direct voxel-wise comparison possible
            va = load_hu(BUNDLE / "files" / group / f"{scan}_1.nii.gz")
            vb = load_hu(BUNDLE / "files" / group / f"{scan}_2.nii.gz")
            diff = va - vb
            raw[scan]["voxelwise"] = {
                "mean_abs_diff_HU": float(np.abs(diff).mean()),
                "max_abs_diff_HU": float(np.abs(diff).max()),
                "identical_voxels": bool(np.array_equal(va, vb)),
            }

    df = pd.DataFrame(rows)
    df.to_csv(OUT_T / "recon_compare.csv", index=False)

    # ---- figure: montages (top) + noise/edge bars (bottom) ----
    n = len(PAIRS)
    fig = plt.figure(figsize=(4 * n, 12))
    gs = fig.add_gridspec(4, n, height_ratios=[1, 1, 0.9, 0.9], hspace=0.35, wspace=0.1)

    for j, (group, scan, model) in enumerate(PAIRS):
        for i, (recon, slices) in enumerate(montage_data[scan]):
            ax = fig.add_subplot(gs[i, j])
            # horizontal filmstrip of the 3 matched levels
            strip = np.concatenate([np.rot90(s) for s in slices], axis=1)
            ax.imshow(strip, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            r = raw[scan]["recons"][recon]
            if i == 0:
                ax.set_title(f"{group}\n{scan} ({model})", fontsize=8)
            ax.set_ylabel(
                f"recon-{recon}\n{r['convolution_kernel']}\n{r['n_slices']}sl "
                f"{r['z_spacing_mm']}mm",
                fontsize=7,
            )

    labels = [p[1].replace("valid_", "") for p in PAIRS]
    x = np.arange(n)
    w = 0.38
    # noise bars
    axn = fig.add_subplot(gs[2, :])
    n1 = [raw[s]["recons"][1]["noise_softtissue_HU"] for _, s, _ in PAIRS]
    n2 = [raw[s]["recons"][2]["noise_softtissue_HU"] for _, s, _ in PAIRS]
    axn.bar(x - w / 2, n1, w, label="recon-1", color="#1f77b4")
    axn.bar(x + w / 2, n2, w, label="recon-2", color="#ff7f0e")
    axn.set_xticks(x)
    axn.set_xticklabels(labels, fontsize=8)
    axn.set_ylabel("soft-tissue\nnoise (HU std)", fontsize=9)
    axn.set_title(
        "Soft-tissue noise (high-pass residual std) — higher = sharper kernel",
        fontsize=9,
    )
    axn.legend(fontsize=8)
    # edge bars (laplacian var, log)
    axe = fig.add_subplot(gs[3, :])
    e1 = [raw[s]["recons"][1]["laplacian_var"] for _, s, _ in PAIRS]
    e2 = [raw[s]["recons"][2]["laplacian_var"] for _, s, _ in PAIRS]
    axe.bar(x - w / 2, e1, w, label="recon-1", color="#1f77b4")
    axe.bar(x + w / 2, e2, w, label="recon-2", color="#ff7f0e")
    axe.set_xticks(x)
    axe.set_xticklabels(labels, fontsize=8)
    axe.set_yscale("log")
    axe.set_ylabel("edge energy\nLaplacian var", fontsize=9)
    axe.set_title(
        "Edge sharpness (Laplacian variance, log) — higher = sharper kernel", fontsize=9
    )
    axe.legend(fontsize=8)

    fig.suptitle(
        "Reconstruction comparison: recon-1 vs recon-2 (5 scans, valid split; "
        "lung window WL-600/WW1500, matched fractional levels 0.35/0.50/0.65)\n"
        "Unit = reconstruction (volume). Report/labels constant within a scan; recons "
        "differ only in kernel/geometry.",
        fontsize=11,
    )
    fig.savefig(OUT_F / "recon_compare.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # ---- generalization summary ----
    summary = {}
    for group, scan, model in PAIRS:
        r1, r2 = raw[scan]["recons"][1], raw[scan]["recons"][2]
        summary[scan] = {
            "manufacturer_model": model,
            "recon1_kernel": r1["convolution_kernel"],
            "recon1_series": r1["series_description"],
            "recon2_kernel": r2["convolution_kernel"],
            "recon2_series": r2["series_description"],
            "empirically_sharper_recon": raw[scan]["empirically_sharper_recon"],
            "geometry_identical": raw[scan]["geometry_identical"],
        }
    raw["_generalization"] = summary
    with open(OUT_T / "recon_raw.json", "w") as f:
        json.dump(raw, f, indent=2)

    # ---- console digest ----
    print("=== recon_compare (read back) ===")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(
            df[
                [
                    "scan_key",
                    "recon",
                    "convolution_kernel",
                    "shape",
                    "z_spacing_mm",
                    "noise_softtissue_HU",
                    "laplacian_var",
                    "grad_mag_mean",
                    "p50",
                    "p99",
                ]
            ].to_string(index=False)
        )
    print("\n=== generalization: which recon is sharper (by Laplacian var) ===")
    for scan, s in summary.items():
        print(
            f"{scan:15s} model={s['manufacturer_model']:26s} "
            f"sharper=recon-{s['empirically_sharper_recon']} "
            f"geom_identical={s['geometry_identical']} "
            f"r1={s['recon1_kernel']}/{s['recon2_kernel']}=r2"
        )
    if "voxelwise" in raw["valid_1016_b"]:
        v = raw["valid_1016_b"]["voxelwise"]
        print(
            f"\nvalid_1016_b matched-geometry voxelwise: mean|diff|={v['mean_abs_diff_HU']:.2f} HU, "
            f"max|diff|={v['max_abs_diff_HU']:.0f} HU, identical={v['identical_voxels']}"
        )
    print(f"\nrows={len(df)}  csv={OUT_T / 'recon_compare.csv'}")


if __name__ == "__main__":
    main()
