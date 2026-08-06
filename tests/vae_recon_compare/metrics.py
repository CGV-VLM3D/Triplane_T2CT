"""Quantify the distortion-vs-detail tradeoff from the aligned npz slices.

Answers "is MAISI's overall diff actually larger?" with numbers:
  * MAE / RMSE (recon - GT), whole slice + lung region  -> pixelwise error (blur wins = lower).
  * grad-retention = mean|nabla recon| / mean|nabla GT|  -> detail kept (1.0 = full; <1 = smoothed).
    Reported raw AND on a sigma=1 low-pass (structure only, noise-realization removed).
All on the 3 saved axial slices per patient (512x512 HU). Prints per-patient + cohort means.
"""

from __future__ import annotations

import json

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_gradient_magnitude

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from tests.vae_recon_compare import common as C  # noqa: E402

HU_RANGE = 2000.0  # [-1000,1000] -> PSNR peak
LUNG_MAX_HU = (
    -400.0
)  # GT voxels below this ~ lung/air parenchyma (where fine detail lives)


def _metrics(gt: np.ndarray, rec: np.ndarray) -> dict:
    d = rec - gt
    mae = float(np.abs(d).mean())
    rmse = float(np.sqrt((d**2).mean()))
    psnr = float(20 * np.log10(HU_RANGE / rmse)) if rmse > 0 else float("inf")
    lung = gt < LUNG_MAX_HU
    lung_mae = float(np.abs(d[lung]).mean()) if lung.any() else float("nan")
    # detail retention: mean edge strength recon / GT (per slice, then averaged)
    g_gt = gaussian_gradient_magnitude(gt, 1.5)
    g_re = gaussian_gradient_magnitude(rec, 1.5)
    grad_ret = float(g_re.mean() / g_gt.mean())
    gs_gt = gaussian_gradient_magnitude(gaussian_filter(gt, 1.0), 1.5)
    gs_re = gaussian_gradient_magnitude(gaussian_filter(rec, 1.0), 1.5)
    grad_ret_struct = float(gs_re.mean() / gs_gt.mean())
    # detail retention INSIDE the lung parenchyma only (where fine vessels live) — the global
    # gradient is dominated by chest-wall/mediastinal boundaries and can hide lung smearing.
    grad_ret_lung = (
        float(g_re[lung].mean() / g_gt[lung].mean()) if lung.any() else float("nan")
    )
    return {
        "MAE": mae,
        "RMSE": rmse,
        "PSNR": psnr,
        "lungMAE": lung_mae,
        "gradRet": grad_ret,
        "gradRetStruct": grad_ret_struct,
        "gradRetLung": grad_ret_lung,
    }


def main() -> None:
    patients = json.loads((C.OUT_DIR / "manifest.json").read_text())
    rows: list[tuple] = []
    agg: dict[tuple[str, str], list[dict]] = {}
    for p in patients:
        sid, cohort = p["id"], p["cohort"]
        mg = np.load(C.OUT_DIR / f"{sid}_maisi_gt.npz")
        wn = np.load(C.OUT_DIR / f"{sid}_wan.npz")
        gt = mg["gt"]  # (3,512,512) HU
        for method, rec in [("MAISI", mg["maisi"]), ("Wan", wn["wan"])]:
            m = {
                k: np.mean([_metrics(gt[i], rec[i])[k] for i in range(gt.shape[0])])
                for k in [
                    "MAE",
                    "RMSE",
                    "PSNR",
                    "lungMAE",
                    "gradRet",
                    "gradRetStruct",
                    "gradRetLung",
                ]
            }
            rows.append((cohort, sid, method, m))
            agg.setdefault((cohort, method), []).append(m)

    hdr = f"{'cohort':9s} {'scan':16s} {'method':6s} | {'MAE':>7s} {'RMSE':>7s} {'PSNR':>6s} {'lungMAE':>8s} | {'gradRet':>8s} {'gradRet*':>9s} {'gRetLung':>9s}"
    print(hdr)
    print("-" * len(hdr))
    for cohort, sid, method, m in rows:
        print(
            f"{cohort:9s} {sid:16s} {method:6s} | {m['MAE']:7.1f} {m['RMSE']:7.1f} "
            f"{m['PSNR']:6.2f} {m['lungMAE']:8.1f} | {m['gradRet']:8.3f} {m['gradRetStruct']:9.3f} {m['gradRetLung']:9.3f}"
        )
    print("-" * len(hdr))
    print(
        "MEANS (gRetLung<1 => lung fine-detail smoothed; MAE/RMSE lower => smaller pixel diff):"
    )
    for (cohort, method), ms in sorted(agg.items()):
        mean = {k: float(np.mean([x[k] for x in ms])) for k in ms[0]}
        print(
            f"{cohort:9s} {'':16s} {method:6s} | {mean['MAE']:7.1f} {mean['RMSE']:7.1f} "
            f"{mean['PSNR']:6.2f} {mean['lungMAE']:8.1f} | {mean['gradRet']:8.3f} {mean['gradRetStruct']:9.3f} {mean['gradRetLung']:9.3f}"
        )


if __name__ == "__main__":
    main()
