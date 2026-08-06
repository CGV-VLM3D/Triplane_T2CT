"""CLEAN (no_chest+error 제외) train+val 합산 ConvolutionKernel 분포 — volume-level.

figure의 scan-level 패널과 달리 재구성(volume)당 1개로 집계(커널의 자연 단위).
"""

import ast
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path("/workspace/datasets/datasets/CT-RATE/dataset")
META = ROOT / "metadata"
OUT = Path("/workspace/tests/ctrate_eda/figures/kernel_volume_clean.png")

# ---- CLEAN 제외 집합: no_chest(789) + error(3), recon-level basename ----
excl: set[str] = set()
for d in ("no_chest_data", "error_ctrate_data"):
    for p in glob.glob(str(ROOT / d / "*" / "*" / "*" / "*.nii.gz")):
        excl.add(Path(p).name[: -len(".nii.gz")])

# 벤더 판정: 정규화된 커널명 -> 벤더 (metadata crosstab에서 verified)
PHILIPS = {"B", "YA", "A", "L", "YB", "UB", "D", "E", "YC", "C", "UA"}
PNMS = {"EA", "SA", "SB", "LungB"}


def norm(k: str) -> str:
    """['Br40f', '3'] -> 'Br40f'; 단일 코드는 그대로."""
    k = str(k)
    if k.startswith("["):
        try:
            return ast.literal_eval(k)[0]
        except (ValueError, SyntaxError):
            return k
    return k


def vendor(k: str) -> str:
    if k in PHILIPS:
        return "Philips"
    if k in PNMS:
        return "PNMS"
    return "Siemens"  # Br*/Bl*/B31s/B70s/... = Siemens


frames = []
for f, split in (("train_metadata.csv", "train"), ("validation_metadata.csv", "valid")):
    df = pd.read_csv(META / f)
    df["stem"] = df["VolumeName"].str.replace(".nii.gz", "", regex=False)
    df = df[~df["stem"].isin(excl)].copy()  # CLEAN
    df["k"] = df["ConvolutionKernel"].map(norm)
    df["split"] = split
    frames.append(df)
allc = pd.concat(frames, ignore_index=True)
n_vol = len(allc)

# top-N by combined count, train/valid split 안에서 stacked
TOPN = 14
order = allc["k"].value_counts().head(TOPN).index.tolist()
piv = (
    allc[allc["k"].isin(order)]
    .groupby(["k", "split"])
    .size()
    .unstack(fill_value=0)
    .reindex(order)
)
piv = piv.iloc[::-1]  # barh는 아래->위, 큰 값이 위로 오게

vcol = {"Philips": "#1f77b4", "Siemens": "#ff7f0e", "PNMS": "#2ca02c"}
edge = [vcol[vendor(k)] for k in piv.index]

fig, ax = plt.subplots(figsize=(10, 6.5))
y = range(len(piv))
tr = ax.barh(
    y,
    piv["train"],
    color=[vcol[vendor(k)] for k in piv.index],
    alpha=0.95,
    label="train",
)
ax.barh(y, piv["valid"], left=piv["train"], color="#888888", alpha=0.9, label="valid")
ax.set_yticks(list(y))
ax.set_yticklabels(piv.index)
ax.set_xlabel("volumes (reconstructions)")
ax.set_title(
    "CT-RATE ConvolutionKernel — CLEAN, train+val combined, VOLUME-level\n"
    f"{n_vol} volumes  (no_chest+error excluded)   bar color = vendor, grey = valid share",
    fontsize=11,
)

# 각 막대에 combined 개수 + % 라벨
for i, k in enumerate(piv.index):
    total = int(piv.loc[k].sum())
    ax.text(
        total + n_vol * 0.005,
        i,
        f"{total:,} ({100 * total / n_vol:.1f}%)",
        va="center",
        fontsize=8.5,
    )

# 벤더 범례
from matplotlib.patches import Patch

handles = [Patch(color=vcol[v], label=v) for v in ("Philips", "Siemens", "PNMS")]
handles.append(Patch(color="#888888", label="valid share"))
ax.legend(handles=handles, loc="lower right", fontsize=9)
ax.set_xlim(0, allc["k"].value_counts().max() * 1.18)
plt.tight_layout()
plt.savefig(OUT, dpi=150)
print("saved:", OUT)
print("volumes:", n_vol)
