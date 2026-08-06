"""Shared config + pure-numpy/torch helpers for the MAISI-vs-Wan VAE recon comparison.

Goal: visually compare the encode->decode round-trip of the two latent substrates
(MAISI VAE 480x480x256 [0,1]; Wan2.1 VAE 512x512x253 HU) against ground truth for a
handful of *detail-heavy* abnormal patients, to judge whether Wan smears fine
parenchymal texture / small structures.

This module holds only env-agnostic helpers (numpy + torch), so it imports cleanly in
both the main env (MAISI decode + montage) and the `wan` conda env (Wan decode).
Anything needing monai (GT load, MAISI VAE) or matplotlib (montage) lives in the
per-stage scripts.

Axis convention (both VAEs precompute with Orientationd axcodes="RAS" before resize, so
all volumes share RAS in-plane orientation):
  * volume array is (X, Y, Z) = (L->R, P->A, I->S).
  * an axial slice fixes Z -> (X, Y).
  * `orient_axial` rotates it to a conventional anterior-up display (SAME transform for
    every method, so any residual convention quirk is identical across columns).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Final

import numpy as np
import torch
import torch.nn.functional as F

ROOT: Final[Path] = Path("/workspace")
VALID_IDS: Final[Path] = ROOT / "data/ctrate_toy_v2/valid_v2/ids.json"
LABELS_CSV: Final[Path] = (
    ROOT
    / "datasets/datasets/CT-RATE/dataset/multi_abnormality_labels/valid_predicted_labels.csv"
)
MAISI_LAT_DIR: Final[Path] = ROOT / "data/report2ct_work_dir/image_embeddings"
WAN_LAT_DIR: Final[Path] = ROOT / "data/report2ct_wan/latents_512x512x253"
GT_DIR: Final[Path] = ROOT / "datasets/datasets/CT-RATE/dataset/valid_fixed"
OUT_DIR: Final[Path] = ROOT / "tests/vae_recon_compare/out"

# Findings whose visual signature is fine texture / small structures — the ones a lossy
# VAE would blur first. Names must match the CSV header exactly.
DETAIL_LABELS: Final[tuple[str, ...]] = (
    "Emphysema",
    "Atelectasis",
    "Lung nodule",
    "Lung opacity",
    "Pulmonary fibrotic sequela",
    "Mosaic attenuation pattern",
    "Peribronchial thickening",
    "Consolidation",
    "Bronchiectasis",
    "Interlobular septal thickening",
)

# Display grid + which axial levels + zoom crop (normalized on the 512 display slice).
DISPLAY: Final[int] = 512
Z_FRACTIONS: Final[tuple[float, ...]] = (0.35, 0.50, 0.65)
# (row0, row1, col0, col1) in [0,1] on the oriented display slice — a lung-parenchyma box.
CROP_NORM: Final[tuple[float, float, float, float]] = (0.30, 0.78, 0.10, 0.58)

# (window level, window width) in HU.
HU_WINDOWS: Final[dict[str, tuple[float, float]]] = {
    "lung": (-600.0, 1500.0),
    "mediastinal": (40.0, 400.0),
}


def load_valid_ids() -> list[str]:
    """Return the 1304 valid_v2 scan ids (e.g. ``valid_1000_a_1``)."""
    return list(json.loads(VALID_IDS.read_text())["ids"])


def load_labels() -> tuple[list[str], dict[str, list[int]]]:
    """Read the multi-abnormality CSV.

    Returns:
        ``(label_names, {scan_id: 0/1 vector})`` where scan_id drops the ``.nii.gz``.
    """
    with LABELS_CSV.open() as f:
        r = csv.reader(f)
        header = next(r)
        names = header[1:]
        table: dict[str, list[int]] = {}
        for row in r:
            sid = row[0].replace(".nii.gz", "")
            table[sid] = [int(v) for v in row[1:]]
    return names, table


def select_detail_patients(n: int = 5) -> list[dict]:
    """Pick the n valid_v2 scans with the most detail-heavy positive findings.

    Requires both latents + the GT NIfTI on disk. Ranks by (# detail labels positive,
    then # total labels positive). Returns manifest dicts with id + positive-label names.
    """
    ids = set(load_valid_ids())
    names, table = load_labels()
    detail_idx = [names.index(x) for x in DETAIL_LABELS]

    cand: list[dict] = []
    for sid in ids:
        vec = table.get(sid)
        if vec is None:
            continue
        if not (MAISI_LAT_DIR / f"{sid}_emb.nii.gz").is_file():
            continue
        if not (WAN_LAT_DIR / f"{sid}_emb.nii.gz").is_file():
            continue
        if not gt_path(sid).is_file():
            continue
        detail_score = sum(vec[i] for i in detail_idx)
        total = sum(vec)
        pos = [names[i] for i, v in enumerate(vec) if v]
        cand.append(
            {"id": sid, "detail_score": detail_score, "total": total, "positive": pos}
        )

    cand.sort(key=lambda d: (d["detail_score"], d["total"]), reverse=True)
    out = cand[:n]
    for d in out:
        d["cohort"] = "abnormal"
    return out


def select_healthy_patients(n: int = 1) -> list[dict]:
    """Pick n valid_v2 scans with ZERO positive findings (all-18 labels 0) as controls.

    Deterministic (id-sorted). Requires both latents + the GT NIfTI on disk. Serves as the
    baseline: how much does each VAE smear a *normal* scan, absent any pathology?
    """
    ids = sorted(load_valid_ids())
    _, table = load_labels()
    out: list[dict] = []
    for sid in ids:
        vec = table.get(sid)
        if vec is None or sum(vec) != 0:
            continue
        if not (MAISI_LAT_DIR / f"{sid}_emb.nii.gz").is_file():
            continue
        if not (WAN_LAT_DIR / f"{sid}_emb.nii.gz").is_file():
            continue
        if not gt_path(sid).is_file():
            continue
        out.append(
            {
                "id": sid,
                "detail_score": 0,
                "total": 0,
                "positive": [],
                "cohort": "healthy",
            }
        )
        if len(out) == n:
            break
    return out


def gt_path(sid: str) -> Path:
    """valid_1000_a_1 -> valid_fixed/valid_1000/valid_1000_a/valid_1000_a_1.nii.gz."""
    parts = sid.split("_")
    return GT_DIR / "_".join(parts[:2]) / "_".join(parts[:3]) / f"{sid}.nii.gz"


# -- slice extraction / display -------------------------------------------------


def axial_slice(vol_xyz: np.ndarray, frac: float) -> np.ndarray:
    """Extract one axial slice (fix Z) from an ``(X, Y, Z)`` volume -> ``(X, Y)``."""
    z = int(round(frac * (vol_xyz.shape[2] - 1)))
    return np.asarray(vol_xyz[:, :, z])


def orient_axial(slice_xy: np.ndarray) -> np.ndarray:
    """Rotate an ``(X=L->R, Y=P->A)`` slice to conventional anterior-up, patient-right-left.

    ``rot90(k=1)`` maps rows=X,cols=Y -> rows=A->P(top=anterior after flip), then we flip
    vertically so anterior is up. Applied identically to GT / MAISI / Wan.
    """
    return np.flipud(np.rot90(slice_xy, k=1))


def resize2d(arr: np.ndarray, size: int = DISPLAY) -> np.ndarray:
    """Bilinear-resize a 2D array to ``(size, size)`` (float32)."""
    t = torch.from_numpy(np.ascontiguousarray(arr)).float()[None, None]  # (1,1,H,W)
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t[0, 0].numpy()


def slices_hu(vol_xyz_hu: np.ndarray) -> np.ndarray:
    """Extract + orient + resize the Z_FRACTIONS axial slices -> ``(len(fracs), 512, 512)`` HU."""
    out = [resize2d(orient_axial(axial_slice(vol_xyz_hu, f))) for f in Z_FRACTIONS]
    return np.stack(out).astype(np.float32)  # (n_frac, 512, 512)


def apply_window(hu: np.ndarray, window: str) -> np.ndarray:
    """Map HU to [0,1] under a named CT window (WL, WW)."""
    wl, ww = HU_WINDOWS[window]
    lo, hi = wl - ww / 2.0, wl + ww / 2.0
    return np.clip((hu - lo) / (hi - lo), 0.0, 1.0)


def crop_box(size: int = DISPLAY) -> tuple[int, int, int, int]:
    """CROP_NORM -> integer (r0, r1, c0, c1) pixel box on a ``size`` grid."""
    r0, r1, c0, c1 = CROP_NORM
    return int(r0 * size), int(r1 * size), int(c0 * size), int(c1 * size)
