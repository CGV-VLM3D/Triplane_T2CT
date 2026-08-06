"""03_label — Label EDA (GPT §4) for CT-RATE 18 abnormality labels.

Unit = SCAN (dedup reconstructions). Default population = chest-only (clean):
exclude no_chest (volume-level) + unencodable volumes. Also reports
no_chest incl/excl and volume-level variants for comparison.

Self-contained + re-runnable. Read-only on datasets.
"""

import sys

sys.path.insert(0, "/workspace")

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

DATA = Path("/workspace/datasets/datasets/CT-RATE/dataset")
LABELS = DATA / "multi_abnormality_labels"
META = DATA / "metadata"
OUT = Path("/workspace/tests/ctrate_eda")
(OUT / "tables").mkdir(parents=True, exist_ok=True)
(OUT / "figures").mkdir(parents=True, exist_ok=True)

LABEL_COLS = [
    "Medical material",
    "Arterial wall calcification",
    "Cardiomegaly",
    "Pericardial effusion",
    "Coronary artery wall calcification",
    "Hiatal hernia",
    "Lymphadenopathy",
    "Emphysema",
    "Atelectasis",
    "Lung nodule",
    "Lung opacity",
    "Pulmonary fibrotic sequela",
    "Pleural effusion",
    "Mosaic attenuation pattern",
    "Peribronchial thickening",
    "Consolidation",
    "Bronchiectasis",
    "Interlobular septal thickening",
]

# Unencodable volumes (fact sheet)
UNENCODABLE = {
    "train_14384_a_2",
    "valid_251_a_2",
    "train_1267_a_4",
    "train_11755_a_3",
    "train_11755_a_4",
}


def vol_stem(vn: str) -> str:
    return vn.replace(".nii.gz", "")


def scan_key(stem: str) -> str:
    # split_patient_scan_recon -> split_patient_scan
    return "_".join(stem.split("_")[:3])


def patient_key(stem: str) -> str:
    return "_".join(stem.split("_")[:2])


def load_no_chest(split: str) -> set:
    """Basenames (stem) of no_chest volumes for a split."""
    p = META / f"no_chest_{split}.txt"
    s = set()
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        s.add(vol_stem(Path(line).name))
    return s


def load_labels(split: str) -> pd.DataFrame:
    csv = (
        "train_predicted_labels.csv"
        if split == "train"
        else "valid_predicted_labels.csv"
    )
    df = pd.read_csv(LABELS / csv)
    df["stem"] = df["VolumeName"].map(vol_stem)
    df["scan"] = df["stem"].map(scan_key)
    df["patient"] = df["stem"].map(patient_key)
    return df


def dedup_scan(df: pd.DataFrame) -> pd.DataFrame:
    """One row per scan. Labels are constant per scan; take first (verified)."""
    return df.drop_duplicates(subset="scan", keep="first").reset_index(drop=True)


def prevalence(df: pd.DataFrame) -> dict:
    n = len(df)
    L = df[LABEL_COLS].values.astype(int)
    per = L.mean(axis=0) * 100.0
    allzero = (L.sum(axis=1) == 0).mean() * 100.0
    labels_per = L.sum(axis=1)
    return {
        "n": n,
        "allzero_pct": float(allzero),
        "mean_pos": float(labels_per.mean()),
        "median_pos": float(np.median(labels_per)),
        "prev_pct": {c: float(per[i]) for i, c in enumerate(LABEL_COLS)},
        "labels_per_hist": {
            int(k): int(v) for k, v in zip(*np.unique(labels_per, return_counts=True))
        },
    }


def main():
    raw = {}

    # ---------- load ----------
    tr_vol = load_labels("train")
    va_vol = load_labels("valid")

    nc_tr = load_no_chest("train")
    nc_va = load_no_chest("valid")

    # verify labels constant per scan
    def scan_const_violations(df):
        g = df.groupby("scan")[LABEL_COLS].nunique()
        return int((g > 1).any(axis=1).sum())

    raw["scan_label_const_violations"] = {
        "train": scan_const_violations(tr_vol),
        "valid": scan_const_violations(va_vol),
    }

    # ---------- population variants ----------
    def clean_vol(df, nc):
        mask = ~df["stem"].isin(nc | UNENCODABLE)
        return df[mask].reset_index(drop=True)

    tr_vol_clean = clean_vol(tr_vol, nc_tr)
    va_vol_clean = clean_vol(va_vol, nc_va)

    tr_scan_clean = dedup_scan(tr_vol_clean)
    va_scan_clean = dedup_scan(va_vol_clean)
    tr_scan_all = dedup_scan(tr_vol)  # incl no_chest
    va_scan_all = dedup_scan(va_vol)

    raw["counts"] = {
        "train": {
            "vol_csv": len(tr_vol),
            "vol_clean": len(tr_vol_clean),
            "scan_clean": len(tr_scan_clean),
            "scan_incl_nochest": len(tr_scan_all),
        },
        "valid": {
            "vol_csv": len(va_vol),
            "vol_clean": len(va_vol_clean),
            "scan_clean": len(va_scan_clean),
            "scan_incl_nochest": len(va_scan_all),
        },
    }

    # ---------- prevalence (all variants) ----------
    variants = {
        "train_scan_clean": prevalence(tr_scan_clean),  # DEFAULT
        "valid_scan_clean": prevalence(va_scan_clean),  # DEFAULT
        "train_scan_incl_nochest": prevalence(tr_scan_all),
        "valid_scan_incl_nochest": prevalence(va_scan_all),
        "train_vol_clean": prevalence(tr_vol_clean),
        "valid_vol_clean": prevalence(va_vol_clean),
    }
    raw["prevalence_variants"] = variants

    # imbalance ratio (default scan_clean, train)
    pv = variants["train_scan_clean"]["prev_pct"]
    raw["imbalance_ratio_train_scan_clean"] = max(pv.values()) / min(pv.values())

    # ---------- prevalence table ----------
    rows = []
    dtr = variants["train_scan_clean"]["prev_pct"]
    dva = variants["valid_scan_clean"]["prev_pct"]
    for c in LABEL_COLS:
        rows.append(
            {
                "label": c,
                "train_scan_clean": dtr[c],
                "valid_scan_clean": dva[c],
                "delta": dva[c] - dtr[c],
                "train_scan_incl_nochest": variants["train_scan_incl_nochest"][
                    "prev_pct"
                ][c],
                "train_vol_clean": variants["train_vol_clean"]["prev_pct"][c],
            }
        )
    prev_df = pd.DataFrame(rows).sort_values("train_scan_clean", ascending=False)
    # append summary rows
    summ = pd.DataFrame(
        [
            {
                "label": "__all_zero_pct",
                "train_scan_clean": variants["train_scan_clean"]["allzero_pct"],
                "valid_scan_clean": variants["valid_scan_clean"]["allzero_pct"],
                "delta": variants["valid_scan_clean"]["allzero_pct"]
                - variants["train_scan_clean"]["allzero_pct"],
            },
            {
                "label": "__mean_pos_labels",
                "train_scan_clean": variants["train_scan_clean"]["mean_pos"],
                "valid_scan_clean": variants["valid_scan_clean"]["mean_pos"],
                "delta": variants["valid_scan_clean"]["mean_pos"]
                - variants["train_scan_clean"]["mean_pos"],
            },
        ]
    )
    pd.concat([prev_df, summ], ignore_index=True).to_csv(
        OUT / "tables/label_prevalence.csv", index=False
    )

    # ---------- co-occurrence (train scan clean) ----------
    L = tr_scan_clean[LABEL_COLS].values.astype(float)
    n = L.shape[0]
    k = len(LABEL_COLS)
    co = L.T @ L  # joint counts (k,k)
    p = L.mean(axis=0)  # marginal prob
    joint = co / n  # P(i&j)
    # Jaccard
    jac = np.zeros((k, k))
    lift = np.zeros((k, k))
    phi = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            union = co[i, i] + co[j, j] - co[i, j]
            jac[i, j] = co[i, j] / union if union > 0 else 0.0
            denom = p[i] * p[j]
            lift[i, j] = joint[i, j] / denom if denom > 0 else 0.0
            # phi (Pearson for binary)
            num = joint[i, j] - p[i] * p[j]
            d = np.sqrt(p[i] * (1 - p[i]) * p[j] * (1 - p[j]))
            phi[i, j] = num / d if d > 0 else 0.0

    raw["cooccurrence"] = {
        "unit": "train_scan_clean",
        "n": int(n),
        "arterial_coronary_joint_count": int(
            co[
                LABEL_COLS.index("Arterial wall calcification"),
                LABEL_COLS.index("Coronary artery wall calcification"),
            ]
        ),
        "arterial_coronary_phi": float(
            phi[
                LABEL_COLS.index("Arterial wall calcification"),
                LABEL_COLS.index("Coronary artery wall calcification"),
            ]
        ),
        "lungopacity_consolidation_phi": float(
            phi[LABEL_COLS.index("Lung opacity"), LABEL_COLS.index("Consolidation")]
        ),
    }

    # long-form co-occurrence table (upper triangle pairs)
    long_rows = []
    for i in range(k):
        for j in range(i + 1, k):
            long_rows.append(
                {
                    "label_a": LABEL_COLS[i],
                    "label_b": LABEL_COLS[j],
                    "count": int(co[i, j]),
                    "jaccard": float(jac[i, j]),
                    "lift": float(lift[i, j]),
                    "phi": float(phi[i, j]),
                }
            )
    pd.DataFrame(long_rows).sort_values("phi", ascending=False).to_csv(
        OUT / "tables/label_cooccurrence.csv", index=False
    )

    # hierarchical clustering order on phi
    dist = 1.0 - phi
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0
    Z = linkage(squareform(dist, checks=False), method="average")
    order = leaves_list(Z)
    clust_order = [LABEL_COLS[i] for i in order]
    raw["cooccurrence"]["cluster_order"] = clust_order

    # ---------- stratified: by scanner / sex / age ----------
    mtr = pd.read_csv(META / "train_metadata.csv")
    mtr["stem"] = mtr["VolumeName"].map(vol_stem)
    mtr["scan"] = mtr["stem"].map(scan_key)
    # scan-level metadata (first recon)
    mtr_scan = mtr.drop_duplicates(subset="scan", keep="first")[
        ["scan", "Manufacturer", "PatientSex", "PatientAge"]
    ]
    strat = tr_scan_clean.merge(mtr_scan, on="scan", how="left")

    def parse_age(a):
        try:
            return int(str(a).rstrip("Yy").lstrip("0") or "0")
        except Exception:
            return np.nan

    strat["age"] = strat["PatientAge"].map(parse_age)
    bins = [0, 40, 55, 70, 200]
    labs = ["<40", "40-54", "55-69", "70+"]
    strat["age_band"] = pd.cut(strat["age"], bins=bins, labels=labs, right=False)

    def group_prev(df, col):
        out = {}
        for key, sub in df.groupby(col, observed=True):
            L2 = sub[LABEL_COLS].values.astype(int)
            out[str(key)] = {
                "n": len(sub),
                **{c: float(L2[:, i].mean() * 100) for i, c in enumerate(LABEL_COLS)},
                "all_zero_pct": float((L2.sum(1) == 0).mean() * 100),
            }
        return out

    strat_scanner = group_prev(strat, "Manufacturer")
    strat_sex = group_prev(strat.dropna(subset=["PatientSex"]), "PatientSex")
    strat_age = group_prev(strat.dropna(subset=["age_band"]), "age_band")
    raw["stratified"] = {
        "scanner": strat_scanner,
        "sex": strat_sex,
        "age_band": strat_age,
    }

    # by-scanner wide table
    sc_rows = []
    for scanner, d in strat_scanner.items():
        row = {"scanner": scanner, "n": d["n"], "all_zero_pct": d["all_zero_pct"]}
        row.update({c: d[c] for c in LABEL_COLS})
        sc_rows.append(row)
    pd.DataFrame(sc_rows).to_csv(OUT / "tables/label_by_scanner.csv", index=False)

    # ---------- rare / long-tail ----------
    sorted_prev = sorted(dtr.items(), key=lambda x: x[1])
    raw["long_tail"] = {"rarest3": sorted_prev[:3], "commonest3": sorted_prev[-3:]}

    # ---------- repeat-scan label change rate (same patient, multiple scans) ----------
    # among patients with >1 distinct scan, fraction of scan-pairs with differing label vectors
    def repeat_change(df):
        multi = df.groupby("patient").filter(lambda g: g["scan"].nunique() > 1)
        n_pat = multi["patient"].nunique()
        changed = 0
        total = 0
        for _, g in multi.groupby("patient"):
            vecs = g[LABEL_COLS].values.astype(int)
            # any variation across this patient's scans?
            total += 1
            if len(np.unique(vecs, axis=0)) > 1:
                changed += 1
        return {
            "n_multiscan_patients": int(n_pat),
            "pct_patients_label_changed": float(changed / total * 100)
            if total
            else 0.0,
        }

    raw["repeat_scan"] = {
        "train": repeat_change(tr_scan_clean),
        "valid": repeat_change(va_scan_clean),
    }

    # ---------- save raw json ----------
    with open(OUT / "tables/label_raw.json", "w") as f:
        json.dump(raw, f, indent=1)

    # ---------- figure: prevalence horizontal bar ----------
    order_lbl = prev_df["label"].tolist()  # sorted by train desc
    y = np.arange(len(order_lbl))
    tvals = [dtr[c] for c in order_lbl]
    vvals = [dva[c] for c in order_lbl]
    fig, ax = plt.subplots(figsize=(9, 8))
    h = 0.4
    ax.barh(y + h / 2, tvals, height=h, color="tab:blue", label="train (scan, clean)")
    ax.barh(y - h / 2, vvals, height=h, color="tab:orange", label="valid (scan, clean)")
    ax.set_yticks(y)
    ax.set_yticklabels(order_lbl, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Prevalence (% of scans)")
    ax.set_title(
        "CT-RATE label prevalence — unit=SCAN, chest-only (clean)\n"
        f"train n={len(tr_scan_clean)} scans, valid n={len(va_scan_clean)} scans"
    )
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "figures/label_prevalence.png", dpi=220)
    plt.close(fig)

    # ---------- figure: clustered phi heatmap ----------
    idx = order
    phi_ord = phi[np.ix_(idx, idx)]
    lbl_ord = [LABEL_COLS[i] for i in idx]
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(phi_ord, cmap="coolwarm", vmin=-0.4, vmax=0.8)
    ax.set_xticks(range(k))
    ax.set_xticklabels(lbl_ord, rotation=90, fontsize=8)
    ax.set_yticks(range(k))
    ax.set_yticklabels(lbl_ord, fontsize=8)
    ax.set_title(
        "Label co-occurrence (phi correlation), hierarchically clustered\n"
        f"unit=SCAN, train chest-only clean, n={len(tr_scan_clean)}"
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="phi")
    fig.tight_layout()
    fig.savefig(OUT / "figures/label_cooccurrence.png", dpi=220)
    plt.close(fig)

    # ---------- console summary ----------
    print("=== counts ===")
    print(json.dumps(raw["counts"], indent=1))
    print("=== train scan-clean prevalence (top) ===")
    for c, v in sorted(dtr.items(), key=lambda x: -x[1]):
        print(f"  {c:38s} {v:6.2f}")
    print(
        f"all_zero train={variants['train_scan_clean']['allzero_pct']:.2f} "
        f"valid={variants['valid_scan_clean']['allzero_pct']:.2f}"
    )
    print(
        f"mean_pos train={variants['train_scan_clean']['mean_pos']:.2f} "
        f"valid={variants['valid_scan_clean']['mean_pos']:.2f}"
    )
    print(
        f"imbalance ratio train scan-clean = {raw['imbalance_ratio_train_scan_clean']:.2f}"
    )
    print(
        "=== co-occ checks ===",
        json.dumps(
            {
                k2: raw["cooccurrence"][k2]
                for k2 in [
                    "arterial_coronary_joint_count",
                    "arterial_coronary_phi",
                    "lungopacity_consolidation_phi",
                ]
            },
            indent=1,
        ),
    )
    print("cluster_order:", clust_order)
    print("repeat_scan:", raw["repeat_scan"])
    print("DONE")


if __name__ == "__main__":
    main()
