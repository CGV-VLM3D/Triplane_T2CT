#!/usr/bin/env python3
"""Multi-metric sharpness measurement for GT vs generated CT, per case.

Answers "is wan inherently blurrier, or is it the clip_grid resample?" robustly,
over many samples and several independent sharpness definitions.

Fairness
--------
Native predictions have different voxel spacing per model (report2ct 0.80 mm,
wan/wan_mask 0.73 mm, GT 0.34 mm), so native numbers are not directly comparable
across models. The primary cross-model comparison is therefore done on the COMMON
clip_grid (0.75 mm isotropic in-plane, 480x480x240) that CLIPScore uses — same
resolution for every source. Native HF is also reported (common physical band) to
isolate the resample effect (native vs clip_grid per model).

Metrics (per case, per source)
------------------------------
On clip_grid (fair cross-model):
    hf_clip   spectral high-freq fraction, physical band [0.15, 0.667] cyc/mm
    grad_clip mean gradient energy per mm (Tenengrad-style)
    lap_clip  variance of Laplacian per mm^2
    edge_clip fraction of body pixels with |grad| above a physical threshold
On native (for resample effect):
    hf_native spectral HF fraction, same physical band [0.15, 0.667] cyc/mm

Higher = sharper for every metric. A robust "wan is blurriest" conclusion means wan
ranks lowest on ALL clip-grid metrics, across most samples.

Usage
-----
    python blur_spectrum.py --cases valid_1_a_1 valid_1000_a_1 --out out.json
    python blur_spectrum.py --n 12 --out out.json          # first 12 valid cases
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from resample_to_clip_grid import (
    _load_vol,
)  # (Z,X,Y)=(240,480,480) HU on clip grid  # noqa: E402

GT_ROOT = pathlib.Path("/workspace/data/vlm3d_eval/_valid_full_3001")
MODELS: dict[str, pathlib.Path] = {
    "report2ct": pathlib.Path(
        "/workspace/outputs/report2ct/eval_report2ct_spacing0.8_full/predictions"
    ),
    "wan": pathlib.Path(
        "/workspace/outputs/report2ct_wan/eval_ep299_sp0.73_1.34_cfg5/predictions"
    ),
    "wan_mask": pathlib.Path(
        "/workspace/outputs/report2ct_wan_mask/eval_ep299_sp0.73_1.34_cfg5/predictions"
    ),
}
CLIP_MM = 0.75  # clip_grid in-plane spacing
F_LO, F_HI = (
    0.15,
    0.667,
)  # physical band (cyc/mm); F_HI = clip_grid Nyquist → fair across resolutions
GRAD_EDGE_THRESH = 50.0  # HU/mm; |grad| above this counts as an edge


def _central_axial(vol_zax: np.ndarray) -> list[np.ndarray]:
    """Central 40-60% axial slices, cropped to a central ~340mm-ish in-plane box is done by caller."""
    Z = vol_zax.shape[0]
    return [vol_zax[z] for z in range(int(Z * 0.40), int(Z * 0.60))]


def _crop_center(s: np.ndarray, keep: float = 0.75) -> np.ndarray:
    """Central crop of a 2D slice to `keep` fraction (focus on body, drop air border)."""
    h, w = s.shape
    rh, rw = int(h * keep / 2), int(w * keep / 2)
    return s[h // 2 - rh : h // 2 + rh, w // 2 - rw : w // 2 + rw]


def _hf_fraction(s: np.ndarray, mm: float) -> float:
    """High-freq energy fraction in physical band [F_LO, F_HI] cyc/mm on a 2D slice."""
    s = np.clip(s, -1000, 400).astype(np.float64)
    s = s - s.mean()
    F = np.fft.fftshift(np.fft.fft2(s))
    P = np.abs(F) ** 2
    ny, nx = P.shape
    cy, cx = ny // 2, nx // 2
    y, x = np.indices((ny, nx))
    fx = (x - cx) / nx / mm
    fy = (y - cy) / ny / mm
    rho = np.sqrt(fx**2 + fy**2).ravel()
    pp = P.ravel()
    tot = pp[(rho > 0) & (rho <= F_HI)].sum()
    band = pp[(rho >= F_LO) & (rho <= F_HI)].sum()
    return float(band / tot) if tot > 0 else float("nan")


def _grad_metrics(s: np.ndarray, mm: float) -> tuple[float, float, float]:
    """Return (mean grad energy /mm, Laplacian variance /mm^2, edge-density) on a body-masked slice."""
    s = np.clip(s, -1000, 400).astype(np.float64)
    gy, gx = np.gradient(s, mm, mm)  # per-mm derivatives
    gmag = np.sqrt(gx**2 + gy**2)
    body = s > -500
    if body.sum() < 100:
        return float("nan"), float("nan"), float("nan")
    grad_energy = float((gmag[body] ** 2).mean())
    # Laplacian per mm^2
    lap = np.gradient(gx, mm, axis=1) + np.gradient(gy, mm, axis=0)
    lap_var = float(lap[body].var())
    edge_density = float((gmag[body] > GRAD_EDGE_THRESH).mean())
    return grad_energy, lap_var, edge_density


def _measure_source(vol_zyx: np.ndarray, mm: float) -> dict:
    """Average sharpness metrics over central axial slices of a volume (Z, Y, X)."""
    hf, ge, lv, ed = [], [], [], []
    for s in _central_axial(vol_zyx):
        s = _crop_center(s)
        hf.append(_hf_fraction(s, mm))
        g, l, e = _grad_metrics(s, mm)
        ge.append(g)
        lv.append(l)
        ed.append(e)
    return {
        "hf": float(np.nanmean(hf)),
        "grad": float(np.nanmean(ge)),
        "lap": float(np.nanmean(lv)),
        "edge": float(np.nanmean(ed)),
    }


def measure_case(case: str) -> dict:
    """Measure GT + every model for one case: native HF + clip_grid multi-metric."""
    out: dict[str, dict] = {}
    sources = {"gt": GT_ROOT / f"{case}.mha"}
    for tag, d in MODELS.items():
        sources[tag] = d / f"{case}.mha"

    for tag, path in sources.items():
        if not path.is_file():
            continue
        # native
        im = sitk.ReadImage(str(path))
        nat = sitk.GetArrayFromImage(im).astype(np.float32)  # (Z,Y,X)
        sp = im.GetSpacing()
        mm_nat = (sp[0] + sp[1]) / 2.0
        hf_native = _measure_source(nat, mm_nat)["hf"]
        del nat
        # clip_grid (common 0.75mm) — flip GT to RAS so body region matches (orientation
        # doesn't affect isotropic sharpness metrics, but keeps crop consistent)
        cg = _load_vol(path, flip_xy=(tag == "gt")).numpy()  # (Z,X,Y)=(240,480,480)
        m = _measure_source(cg, CLIP_MM)
        del cg
        out[tag] = {
            "hf_native": hf_native,
            "hf_clip": m["hf"],
            "grad_clip": m["grad"],
            "lap_clip": m["lap"],
            "edge_clip": m["edge"],
            "mm_native": mm_nat,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*", default=[])
    ap.add_argument(
        "--n", type=int, default=0, help="use first N valid cases if --cases empty"
    )
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()

    cases = args.cases
    if not cases:
        allc = sorted(p.stem for p in GT_ROOT.glob("*.mha"))
        cases = [
            c for c in allc if all((d / f"{c}.mha").exists() for d in MODELS.values())
        ][: args.n]

    results = {}
    for c in cases:
        try:
            results[c] = measure_case(c)
            print(f"[ok] {c}")
        except Exception as e:  # noqa: BLE001
            print(f"[err] {c}: {e}", file=sys.stderr)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"[done] {len(results)} cases → {args.out}")


if __name__ == "__main__":
    main()
