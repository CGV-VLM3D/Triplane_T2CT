#!/usr/bin/env python3
"""Qualitative abnormality-fidelity figures: GT vs 4 ctgen models, 3 planes per case.

Orientation (fixed 2026-07-23): GT (_valid_full_3001) is raw CT-RATE **LPS**; the model
top-level predictions are **RAS** (in-plane X/Y flipped) — verified anatomically (heart sits
on opposite array-X sides). So every non-GT volume is flipped to LPS before display, and all
volumes share z=max=Superior (confirmed via ITK direction +Z=S). Canonical LPS display:
  axial   = arr[z]           -> (Y,X)  anterior at top, patient-left at viewer-right
  coronal = arr[::-1, y, :]  -> (Z,X)  superior at top
  sagittal= arr[::-1, :, x]  -> (Z,Y)  superior at top, anterior at left
Generations are text-conditioned, NOT spatially registered; slices come from each volume's own
body center of mass — judge "does the finding appear", not voxel registration.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

GT_DIR = "/workspace/data/vlm3d_eval/_valid_full_3001"
# (label, predictions dir, is_lps_source)
# is_lps=False = "on-disk pred is RAS, flip it here". These dirs were generated BEFORE the
# samplers gained the save-side ras_to_lps flip (report2ct 2026-07-21, text2ct 2026-07-24), so
# they hold RAS content and this is correct. ⚠ If you REGENERATE any of these with the current
# sampler (now writes LPS at save), set that row's flag to True — else make_figures double-flips it.
MODELS = [
    ("GT", GT_DIR, True),
    (
        "report2ct_toy_v2",
        "outputs/report2ct/eval_cfg5_spacing0.8_toy_v2/predictions",
        False,
    ),
    (
        "report2ct_wan",
        "outputs/report2ct_wan/eval_ep299_sp0.73_1.34_cfg5/predictions",
        False,
    ),
    (
        "report2ct_wan_mask",
        "outputs/report2ct_wan_mask/eval_ep299_sp0.73_1.34_cfg5/predictions",
        False,
    ),
    ("text2ct_v2", "outputs/text2ct_toy_v2/eval_2026-06-28/predictions", False),
]
DR = "/workspace/datasets/datasets/CT-RATE/dataset"
FIGROOT = Path("tests/abnormality_fidelity/figures")

LUNG, MED, CALC = (-550, 1500), (40, 400), (100, 700)

# hand-picked showcase cases: id -> (title, window, axial z-fraction from inferior)
SHOWCASE = {
    "valid_27_a_1": ("Cardiomegaly + pericardial/pleural effusion", MED, 0.42),
    "valid_144_a_1": ("Giant right mediastinal/hilar mass (~15 cm)", (40, 450), 0.55),
    "valid_155_a_1": (
        "Focal lung nodule 15x13 mm (RUL, horizontal fissure)",
        LUNG,
        0.62,
    ),
    "valid_322_a_1": ("Coronary artery (LAD) calcification", CALC, 0.48),
}

# abnormality subfolders: folder -> (label column | None for normal, window, axial frac)
ABNORMALITIES = {
    "cardiomegaly": ("Cardiomegaly", MED, 0.42),
    "pleural_effusion": ("Pleural effusion", MED, 0.32),
    "emphysema": ("Emphysema", LUNG, 0.55),
    "lung_nodule": ("Lung nodule", LUNG, 0.60),
    "consolidation": ("Consolidation", (-450, 1400), 0.45),
    "arterial_calcification": ("Arterial wall calcification", CALC, 0.50),
    "normal_all_zero": (None, LUNG, 0.55),
}


def load_lps(path: str, is_lps: bool):
    """Read a volume as (Z,Y,X) LPS-content HU array + (sx,sy,sz) mm. RAS sources are flipped in-plane."""
    im = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(im).astype(np.float32)  # (Z, Y, X)
    if not is_lps:  # RAS content -> LPS: reverse the two in-plane axes
        arr = arr[:, ::-1, ::-1]
    return np.ascontiguousarray(arr), im.GetSpacing()


def body_centroid(arr: np.ndarray):
    body = arr > -500
    if not body.any():
        return tuple(s // 2 for s in arr.shape)
    zc, yc, xc = (int(round(c)) for c in np.array(np.where(body)).mean(axis=1))
    return zc, yc, xc


def win(sl, wl, ww):
    lo, hi = wl - ww / 2, wl + ww / 2
    return np.clip((sl - lo) / (hi - lo), 0, 1)


def render_case(case_id, title, window, axial_frac, out_png):
    wl, ww = window
    fig, axes = plt.subplots(len(MODELS), 3, figsize=(11, 3.4 * len(MODELS)))
    for r, (name, d, is_lps) in enumerate(MODELS):
        arr, (sx, sy, sz) = load_lps(f"{d}/{case_id}.mha", is_lps)
        _, yc, xc = body_centroid(arr)
        zs = np.where((arr > -500).any(axis=(1, 2)))[0]
        z0, z1 = (int(zs.min()), int(zs.max())) if zs.size else (0, arr.shape[0] - 1)
        za = int(
            np.clip(z0 + round(axial_frac * (z1 - z0)), 0, arr.shape[0] - 1)
        )  # z0=inferior

        panels = [
            (win(arr[za, :, :], wl, ww), sy / sx, "axial"),  # (Y,X) anterior top
            (win(arr[::-1, yc, :], wl, ww), sz / sx, "coronal"),  # (Z,X) superior top
            (
                win(arr[::-1, :, xc], wl, ww),
                sz / sy,
                "sagittal",
            ),  # (Z,Y) superior top, anterior left
        ]
        for c, (img, asp, plane) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(img, cmap="gray", aspect=asp, vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(plane, fontsize=13, fontweight="bold")
            if c == 0:
                ax.set_ylabel(name, fontsize=11, fontweight="bold")
    fig.suptitle(
        f"{case_id}  —  {title}\nwindow WL={wl} WW={ww}   (all volumes LPS, superior up)",
        fontsize=13,
        y=0.997,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_png)


def select_abnormality_cases():
    """Pick 2 cleanest (lowest-n_pos) valid_v2 cases per abnormality; 2 all-normal cases too."""
    ids = set(json.load(open("/workspace/data/ctrate_toy_v2/valid_v2/ids.json"))["ids"])
    lab = pd.read_csv(f"{DR}/multi_abnormality_labels/valid_predicted_labels.csv")
    lab["id"] = lab["VolumeName"].str.replace(".nii.gz", "", regex=False)
    lab = lab[lab["id"].isin(ids)].copy()
    cols = [c for c in lab.columns if c not in ("VolumeName", "id")]
    lab["n_pos"] = lab[cols].sum(axis=1)
    rep = pd.read_csv(f"{DR}/radiology_text_reports/validation_reports.csv")
    rep["id"] = rep["VolumeName"].str.replace(".nii.gz", "", regex=False)
    rep = rep.set_index("id")
    picks = {}
    for folder, (label, _, _) in ABNORMALITIES.items():
        if label is None:  # normal
            cand = lab[lab["n_pos"] == 0].sort_values("id")
        else:
            cand = lab[lab[label] == 1].sort_values(["n_pos", "id"])
        chosen = list(cand["id"].head(2))
        picks[folder] = [
            (cid, str(rep.loc[cid, "Impressions_EN"]).strip()) for cid in chosen
        ]
    return picks


if __name__ == "__main__":
    # 1) showcase cases (top level)
    for cid, (title, window, frac) in SHOWCASE.items():
        render_case(cid, title, window, frac, FIGROOT / f"{cid}.png")
    # 2) abnormality subfolders, 2 cases each (+ normal all-zero)
    picks = select_abnormality_cases()
    manifest = {}
    for folder, (label, window, frac) in ABNORMALITIES.items():
        manifest[folder] = []
        for cid, impression in picks[folder]:
            title = (
                "NORMAL (no abnormality labels)" if label is None else label
            ) + f" — {cid}"
            render_case(cid, title, window, frac, FIGROOT / folder / f"{cid}.png")
            manifest[folder].append({"case": cid, "impression": impression})
    (FIGROOT / "abnormality_cases.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )
    print("\nwrote", FIGROOT / "abnormality_cases.json")
