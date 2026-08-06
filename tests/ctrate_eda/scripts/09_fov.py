"""FOV (field-of-view) statistics for CT-RATE train/valid.

CLEAN population (matches 02_metadata / spacing_shape.csv): exclude no_chest
(volume-level) + unencodable volumes -> train 46,393 / valid 3,001.

Two FOV axes, both volume-level:
  - in-plane FOV (mm)  = ReconstructionDiameter  (~= Columns * XYSpacing)
  - Z coverage   (mm)  = NumberofSlices * ZSpacing  (craniocaudal extent)

Key relationship surfaced: in-plane FOV is anatomy-clamped (~368 mm median),
so the acquisition knob is the matrix (512/768/1024). Since spacing = FOV/matrix,
finer XY spacing <-> larger matrix at ~constant FOV (corr(matrix, xy) ~= -0.92).

Writes: figures/nifti_fov.png, tables/fov.csv
"""

import ast
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/workspace/datasets/datasets/CT-RATE/dataset")
META = ROOT / "metadata"
OUT = Path("/workspace/tests/ctrate_eda")
TRAIN_BLUE = "#1f77b4"
VALID_ORANGE = "#ff7f0e"

# Same exclusion sets as 02_metadata.py (kept in sync by hand).
UNENCODABLE = {
    "train_14384_a_2",
    "valid_251_a_2",
    "train_1267_a_4",
    "train_11755_a_3",
    "train_11755_a_4",
}


def vol_stem(name: str) -> str:
    return name[:-7] if name.endswith(".nii.gz") else name


def parse_spacing_xy(raw):
    try:
        return float(ast.literal_eval(raw)[0])
    except Exception:
        return np.nan


def load_no_chest(split_txt: Path) -> set:
    """no_chest file holds v1 full paths; take basename stems."""
    return {
        vol_stem(line.strip().split("/")[-1])
        for line in split_txt.read_text().splitlines()
        if line.strip()
    }


def load_clean(split):
    meta_csv = META / (
        "train_metadata.csv" if split == "train" else "validation_metadata.csv"
    )
    nc_txt = META / ("no_chest_train.txt" if split == "train" else "no_chest_valid.txt")
    df = pd.read_csv(meta_csv, dtype=str)
    df["stem"] = df["VolumeName"].map(vol_stem)
    no_chest = load_no_chest(nc_txt)
    clean = ~(df["stem"].isin(no_chest) | df["stem"].isin(UNENCODABLE))
    df = df[clean].copy()

    df["xy"] = df["XYSpacing"].map(parse_spacing_xy)
    df["z"] = pd.to_numeric(df["ZSpacing"], errors="coerce")
    df["nslices"] = pd.to_numeric(df["NumberofSlices"], errors="coerce")
    df["cols"] = pd.to_numeric(df["Columns"], errors="coerce")
    df["recon"] = pd.to_numeric(df["ReconstructionDiameter"], errors="coerce")
    df["fov_xy"] = df["recon"]  # DICOM-authoritative in-plane FOV diameter (mm)
    df["fov_z"] = df["nslices"] * df["z"]  # craniocaudal coverage (mm)
    return df


PCTS = [1, 25, 50, 75, 99]


def qdict(s):
    s = s.dropna()
    return {str(p): float(np.percentile(s, p)) for p in PCTS}


train = load_clean("train")
valid = load_clean("valid")

# ---- table ----
rows = []
for name, d in [("train", train), ("valid", valid)]:
    rows.append(
        {
            "split": name,
            "n_volumes": int(len(d)),
            "fov_xy_recon_missing": int(d["recon"].isna().sum()),
            "fov_xy_median": float(d["fov_xy"].median()),
            "fov_xy_q": json.dumps(qdict(d["fov_xy"])),
            "fov_xy_min": float(d["fov_xy"].min()),
            "fov_xy_max": float(d["fov_xy"].max()),
            "fov_z_median": float(d["fov_z"].median()),
            "fov_z_q": json.dumps(qdict(d["fov_z"])),
            "fov_z_min": float(d["fov_z"].min()),
            "fov_z_max": float(d["fov_z"].max()),
            "corr_matrix_xyspacing": float(d["cols"].corr(d["xy"])),
            "corr_zspacing_nslices": float(d["z"].corr(d["nslices"])),
            "fov_xy_by_matrix": json.dumps(
                {
                    str(int(m)): {
                        "n": int((d["cols"] == m).sum()),
                        "xy_mean": round(float(d.loc[d["cols"] == m, "xy"].mean()), 4),
                        "fov_xy_mean": round(
                            float(d.loc[d["cols"] == m, "fov_xy"].mean()), 2
                        ),
                    }
                    for m in [512, 768, 1024]
                }
            ),
        }
    )
tbl = pd.DataFrame(rows)
(OUT / "tables").mkdir(exist_ok=True)
tbl.to_csv(OUT / "tables/fov.csv", index=False)
print("Wrote:", OUT / "tables/fov.csv")
print(
    tbl[
        ["split", "n_volumes", "fov_xy_median", "fov_z_median", "corr_matrix_xyspacing"]
    ].to_string(index=False)
)

# ---- figure ----
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# (0,0) in-plane FOV distribution.
ax = axes[0, 0]
bins = np.linspace(150, 550, 60)
ax.hist(
    train["fov_xy"].dropna(),
    bins=bins,
    color=TRAIN_BLUE,
    alpha=0.6,
    density=True,
    label="train",
)
ax.hist(
    valid["fov_xy"].dropna(),
    bins=bins,
    color=VALID_ORANGE,
    alpha=0.6,
    density=True,
    label="valid",
)
ax.axvline(train["fov_xy"].median(), color=TRAIN_BLUE, ls="--", lw=1)
ax.set_title("In-plane FOV = ReconstructionDiameter (volume-level)")
ax.set_xlabel("in-plane FOV (mm)")
ax.set_ylabel("density")
ax.legend()

# (0,1) Z coverage distribution.
ax = axes[0, 1]
binsz = np.linspace(0, 700, 60)
ax.hist(
    train["fov_z"].clip(upper=700).dropna(),
    bins=binsz,
    color=TRAIN_BLUE,
    alpha=0.6,
    density=True,
    label="train",
)
ax.hist(
    valid["fov_z"].clip(upper=700).dropna(),
    bins=binsz,
    color=VALID_ORANGE,
    alpha=0.6,
    density=True,
    label="valid",
)
ax.set_title("Z coverage = nslices x ZSpacing (clip<=700mm)")
ax.set_xlabel("craniocaudal extent (mm)")
ax.set_ylabel("density")
ax.legend()

# (1,0) spacing vs matrix: finer spacing <-> larger matrix at ~const FOV.
ax = axes[1, 0]
for d, c, lab in [(train, TRAIN_BLUE, "train"), (valid, VALID_ORANGE, "valid")]:
    ax.scatter(d["cols"], d["xy"], s=4, alpha=0.15, color=c, label=lab)
ax.set_title(f"XY spacing vs matrix (train corr={train['cols'].corr(train['xy']):.2f})")
ax.set_xlabel("matrix (Columns)")
ax.set_ylabel("XY spacing (mm)")
ax.set_xlim(400, 1100)
ax.legend()

# (1,1) mean in-plane FOV by matrix group -> FOV is ~constant across matrices.
ax = axes[1, 1]
groups = [512, 768, 1024]
x = np.arange(len(groups))
w = 0.35
tv = [train.loc[train["cols"] == m, "fov_xy"].mean() for m in groups]
vv = [valid.loc[valid["cols"] == m, "fov_xy"].mean() for m in groups]
ax.bar(x - w / 2, tv, w, color=TRAIN_BLUE, label="train")
ax.bar(x + w / 2, vv, w, color=VALID_ORANGE, label="valid")
ax.set_xticks(x)
ax.set_xticklabels([f"{m}x{m}" for m in groups])
ax.set_title("Mean in-plane FOV by matrix (anatomy-clamped ~const)")
ax.set_ylabel("mean FOV (mm)")
ax.set_ylim(300, 420)
ax.legend()

fig.suptitle(
    f"CT-RATE FOV (CLEAN: no_chest + unencodable excluded; train n={len(train)}, valid n={len(valid)})",
    fontsize=13,
)
fig.tight_layout(rect=[0, 0, 1, 0.97])
(OUT / "figures").mkdir(exist_ok=True)
fig.savefig(OUT / "figures/nifti_fov.png", dpi=200)
print("Wrote:", OUT / "figures/nifti_fov.png")
