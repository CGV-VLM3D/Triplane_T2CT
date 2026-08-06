"""02_metadata — CT-RATE metadata EDA (GPT §3 verify+extend).

CLEAN population (default): exclude no_chest (volume-level) + unencodable volumes.
Distinguishes volume-level (spacing/shape/kernel) vs scan-level (demographics/
manufacturer, constant per scan) vs patient-level statistics.

Outputs under /workspace/tests/ctrate_eda/:
  tables/demographics.csv, scanner_protocol.csv, spacing_shape.csv,
  metadata_missingness.csv, kernel_table.csv, metadata_raw.json
  figures/metadata_overview.png
All raw pre-rounding numbers live in metadata_raw.json.
"""

import sys

sys.path.insert(0, "/workspace")

import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/workspace/datasets/datasets/CT-RATE/dataset")
META = ROOT / "metadata"
LABELS = ROOT / "multi_abnormality_labels"
OUT = Path("/workspace/tests/ctrate_eda")
(OUT / "tables").mkdir(parents=True, exist_ok=True)
(OUT / "figures").mkdir(parents=True, exist_ok=True)

TRAIN_BLUE = "#1f77b4"
VALID_ORANGE = "#ff7f0e"

# Unencodable volumes (from authoritative facts).
UNENCODABLE = {
    "train_14384_a_2",
    "valid_251_a_2",
    "train_1267_a_4",
    "train_11755_a_3",
    "train_11755_a_4",
}

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


# ---------------------------------------------------------------- helpers
def vol_stem(name: str) -> str:
    """'train_1_a_1.nii.gz' -> 'train_1_a_1'."""
    return str(name).replace(".nii.gz", "")


def scan_key(name: str) -> str:
    """volume -> scan key: 'train_1_a_1' -> 'train_1_a' (drop recon idx)."""
    return "_".join(vol_stem(name).split("_")[:3])


def patient_key(name: str) -> str:
    """volume -> patient key: 'train_1_a_1' -> 'train_1'."""
    return "_".join(vol_stem(name).split("_")[:2])


def load_no_chest(split_txt: Path) -> set:
    """no_chest file holds v1 full paths; take basename stems."""
    out = set()
    if not split_txt.exists():
        return out
    for line in split_txt.read_text().splitlines():
        line = line.strip()
        if line:
            out.add(vol_stem(Path(line).name))
    return out


def parse_age(raw):
    """Robust PatientAge parse: '049Y' -> 49; strip non-digits; None if empty."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    digits = re.sub(r"\D", "", str(raw))
    if not digits:
        return None
    return int(digits)


def parse_spacing_xy(raw):
    """XYSpacing stored as '[0.82, 0.82]' -> first component float."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    s = str(raw).strip()
    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple)):
            return float(v[0])
        return float(v)
    except Exception:
        s2 = s.strip("[]").split(",")[0].strip()
        try:
            return float(s2)
        except Exception:
            return None


def canon_manufacturer(raw):
    """Merge SIEMENS + Siemens Healthineers -> Siemens; keep Philips, PNMS."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return "missing"
    s = str(raw).strip()
    u = s.upper()
    if "SIEMENS" in u:
        return "Siemens"
    if "PHILIPS" in u:
        return "Philips"
    if u == "PNMS" or "PNMS" in u:
        return "PNMS"
    return s


def raw_manufacturer(raw):
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return "missing"
    return str(raw).strip()


def q(series, ps):
    s = pd.Series(series).dropna().astype(float)
    if len(s) == 0:
        return {str(p): None for p in ps}
    return {str(p): float(np.percentile(s, p)) for p in ps}


# ---------------------------------------------------------------- load + clean
def load_split(split):
    meta_csv = META / (
        "train_metadata.csv" if split == "train" else "validation_metadata.csv"
    )
    lab_csv = LABELS / (
        "train_predicted_labels.csv"
        if split == "train"
        else "valid_predicted_labels.csv"
    )
    nc_txt = META / ("no_chest_train.txt" if split == "train" else "no_chest_valid.txt")

    df = pd.read_csv(meta_csv, dtype=str)
    df["stem"] = df["VolumeName"].map(vol_stem)
    df["scan"] = df["VolumeName"].map(scan_key)
    df["patient"] = df["VolumeName"].map(patient_key)

    no_chest = load_no_chest(nc_txt)
    df["is_no_chest"] = df["stem"].isin(no_chest)
    df["is_unencodable"] = df["stem"].isin(UNENCODABLE)
    df["clean"] = ~(df["is_no_chest"] | df["is_unencodable"])

    lab = pd.read_csv(lab_csv)
    lab["stem"] = lab["VolumeName"].map(vol_stem)
    df = df.merge(lab[["stem"] + LABEL_COLS], on="stem", how="left")

    # Derived numeric columns.
    df["age"] = df["PatientAge"].map(parse_age)
    df["xy"] = df["XYSpacing"].map(parse_spacing_xy)
    df["z"] = pd.to_numeric(df["ZSpacing"], errors="coerce")
    df["nslices"] = pd.to_numeric(df["NumberofSlices"], errors="coerce")
    df["rows"] = pd.to_numeric(df["Rows"], errors="coerce")
    df["cols"] = pd.to_numeric(df["Columns"], errors="coerce")
    df["rescale_int"] = pd.to_numeric(df["RescaleIntercept"], errors="coerce")
    df["mfr"] = df["Manufacturer"].map(canon_manufacturer)
    df["mfr_raw"] = df["Manufacturer"].map(raw_manufacturer)
    df["studydate"] = pd.to_datetime(df["StudyDate"], format="%Y%m%d", errors="coerce")
    return df


train = load_split("train")
valid = load_split("valid")

# Clean subsets.
tc = train[train["clean"]].copy()
vc = valid[valid["clean"]].copy()

raw = {"population": {}}
raw["population"]["train"] = {
    "csv_volumes": int(len(train)),
    "clean_volumes": int(len(tc)),
    "no_chest_volumes": int(train["is_no_chest"].sum()),
    "unencodable_volumes": int(train["is_unencodable"].sum()),
    "clean_scans": int(tc["scan"].nunique()),
    "clean_patients": int(tc["patient"].nunique()),
}
raw["population"]["valid"] = {
    "csv_volumes": int(len(valid)),
    "clean_volumes": int(len(vc)),
    "no_chest_volumes": int(valid["is_no_chest"].sum()),
    "unencodable_volumes": int(valid["is_unencodable"].sum()),
    "clean_scans": int(vc["scan"].nunique()),
    "clean_patients": int(vc["patient"].nunique()),
}
print("Population:", json.dumps(raw["population"], indent=2))


# ---------------------------------------------------------------- DEMOGRAPHICS (scan-level)
def scan_level(df):
    """One row per scan (demographics constant per scan)."""
    return df.drop_duplicates("scan")


def patient_level(df):
    return df.drop_duplicates("patient")


demo = {}
demo_rows = []
for name, df in [("train", tc), ("valid", vc)]:
    s = scan_level(df)
    p = patient_level(df)
    ages_scan = s["age"].dropna()
    ages_pat = p["age"].dropna()
    # Sex distribution at scan level.
    sex = s["PatientSex"].fillna("missing").replace("", "missing")
    sex_counts = sex.value_counts().to_dict()
    sex_counts = {
        ("missing" if (k is None or str(k).strip() == "") else str(k)): int(v)
        for k, v in sex_counts.items()
    }
    # Scans per patient.
    spp = s.groupby("patient")["scan"].nunique()
    d = {
        "n_scans": int(len(s)),
        "n_patients": int(len(p)),
        "age_scan_median": float(ages_scan.median()),
        "age_scan_q25": float(np.percentile(ages_scan, 25)),
        "age_scan_q75": float(np.percentile(ages_scan, 75)),
        "age_scan_min": int(ages_scan.min()),
        "age_scan_max": int(ages_scan.max()),
        "age_scan_missing": int(s["age"].isna().sum()),
        "age_gt100_count": int((ages_scan > 100).sum()),
        "age_patient_median": float(ages_pat.median()),
        "age_patient_q25": float(np.percentile(ages_pat, 25)),
        "age_patient_q75": float(np.percentile(ages_pat, 75)),
        "sex_counts": sex_counts,
        "scans_per_patient_mean": float(spp.mean()),
        "scans_per_patient_max": int(spp.max()),
        "patients_multi_scan": int((spp > 1).sum()),
    }
    # Repeat-scan interval (days) from StudyDate for multi-scan patients.
    intervals = []
    for pat, grp in s[s["studydate"].notna()].groupby("patient"):
        dates = sorted(grp["studydate"].tolist())
        for i in range(1, len(dates)):
            intervals.append((dates[i] - dates[i - 1]).days)
    if intervals:
        d["repeat_interval_days_median"] = float(np.median(intervals))
        d["repeat_interval_days_mean"] = float(np.mean(intervals))
        d["repeat_interval_n_pairs"] = int(len(intervals))
    demo[name] = d
    row = {"split": name}
    row.update({k: v for k, v in d.items() if k != "sex_counts"})
    row.update({f"sex_{k}": v for k, v in sex_counts.items()})
    demo_rows.append(row)

raw["demographics"] = demo
pd.DataFrame(demo_rows).to_csv(OUT / "tables/demographics.csv", index=False)
print(
    "Demographics train age median:",
    demo["train"]["age_scan_median"],
    "IQR",
    demo["train"]["age_scan_q25"],
    demo["train"]["age_scan_q75"],
)
print("Sex train:", demo["train"]["sex_counts"], "valid:", demo["valid"]["sex_counts"])


# ---------------------------------------------------------------- SCANNER / PROTOCOL (scan-level)
scanner = {}
sp_rows = []
for name, df in [("train", tc), ("valid", vc)]:
    s = scan_level(df)
    mfr_raw_counts = s["mfr_raw"].value_counts().to_dict()
    mfr_canon_counts = s["mfr"].value_counts().to_dict()
    total = int(len(s))
    mfr_canon_pct = {
        k: round(100.0 * v / total, 2) for k, v in mfr_canon_counts.items()
    }
    model_counts = (
        s["ManufacturerModelName"].fillna("missing").value_counts().head(15).to_dict()
    )
    series_counts = (
        s["SeriesDescription"].fillna("missing").value_counts().head(15).to_dict()
    )
    kernel_counts = (
        s["ConvolutionKernel"].fillna("missing").value_counts().head(20).to_dict()
    )
    scanner[name] = {
        "n_scans": total,
        "manufacturer_raw": {str(k): int(v) for k, v in mfr_raw_counts.items()},
        "manufacturer_canon": {str(k): int(v) for k, v in mfr_canon_counts.items()},
        "manufacturer_canon_pct": {str(k): v for k, v in mfr_canon_pct.items()},
        "model_top": {str(k): int(v) for k, v in model_counts.items()},
        "series_top": {str(k): int(v) for k, v in series_counts.items()},
        "kernel_top": {str(k): int(v) for k, v in kernel_counts.items()},
    }
    for mk, mv in mfr_raw_counts.items():
        sp_rows.append(
            {
                "split": name,
                "level": "scan",
                "field": "manufacturer_raw",
                "value": str(mk),
                "count": int(mv),
            }
        )
    for mk, mv in mfr_canon_counts.items():
        sp_rows.append(
            {
                "split": name,
                "level": "scan",
                "field": "manufacturer_canon",
                "value": str(mk),
                "count": int(mv),
                "pct": mfr_canon_pct[mk],
            }
        )

# Manufacturer x label prevalence (shortcut risk), scan-level, train.
s_tr = scan_level(tc)
mfr_label = {}
for mfr, grp in s_tr.groupby("mfr"):
    n = len(grp)
    mfr_label[mfr] = {"n_scans": int(n)}
    for lc in LABEL_COLS:
        vals = pd.to_numeric(grp[lc], errors="coerce").dropna()
        if len(vals):
            mfr_label[mfr][lc] = round(float(vals.mean()), 4)
scanner["train_mfr_label_prevalence"] = mfr_label

# Kernel <-> manufacturer <-> series cross (train scan-level), full kernel table.
kernel_rows = []
kt = s_tr.groupby(["ConvolutionKernel", "mfr"]).size().reset_index(name="count")
kt = kt.sort_values("count", ascending=False)
for _, r in kt.iterrows():
    kernel_rows.append(
        {
            "kernel_string": str(r["ConvolutionKernel"]),
            "manufacturer": str(r["mfr"]),
            "count": int(r["count"]),
            "family": "",  # BLANK for user to fill sharp/medium/soft
        }
    )
pd.DataFrame(kernel_rows).to_csv(OUT / "tables/kernel_table.csv", index=False)

raw["scanner_protocol"] = scanner
pd.DataFrame(sp_rows).to_csv(OUT / "tables/scanner_protocol.csv", index=False)
print("Manufacturer raw train:", scanner["train"]["manufacturer_raw"])
print("Manufacturer canon train pct:", scanner["train"]["manufacturer_canon_pct"])
print("Manufacturer raw valid:", scanner["valid"]["manufacturer_raw"])


# ---------------------------------------------------------------- SPACING / SHAPE (volume-level)
spacing = {}
ss_rows = []
for name, df in [("train", tc), ("valid", vc)]:
    xy = df["xy"].dropna()
    z = df["z"].dropna()
    ns = df["nslices"].dropna()
    d = {
        "n_volumes": int(len(df)),
        "xy_median": float(xy.median()),
        "xy_q": q(xy, [1, 25, 50, 75, 99]),
        "xy_min": float(xy.min()),
        "xy_max": float(xy.max()),
        "z_median": float(z.median()),
        "z_q": q(z, [1, 25, 50, 75, 99]),
        "z_min": float(z.min()),
        "z_max": float(z.max()),
        "z_value_counts_top": {
            str(k): int(v)
            for k, v in z.round(3).value_counts().head(12).to_dict().items()
        },
        "nslices_median": float(ns.median()),
        "nslices_q": q(ns, [1, 25, 50, 75, 99]),
        "nslices_min": int(ns.min()),
        "nslices_max": int(ns.max()),
        "nslices_gt1000_count": int((ns > 1000).sum()),
        "matrix_counts": {
            str(k): int(v)
            for k, v in (
                df["rows"].astype("Int64").astype(str)
                + "x"
                + df["cols"].astype("Int64").astype(str)
            )
            .value_counts()
            .to_dict()
            .items()
        },
        "rescale_intercept_counts": {
            str(k): int(v)
            for k, v in df["rescale_int"].value_counts().to_dict().items()
        },
    }
    spacing[name] = d
    ss_rows.append(
        {
            "split": name,
            **{k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in d.items()},
        }
    )

# Z=0.035 min investigation. GPT's claimed 0.035mm min is a RAW-population artifact
# excluded by the clean filter — record both raw and clean views.
z_min_clean = tc["z"].min()
z_low_clean = tc[tc["z"] < 0.1][
    ["VolumeName", "z", "nslices", "ReconstructionDiameter", "mfr", "ConvolutionKernel"]
]
z_raw = pd.to_numeric(train["ZSpacing"], errors="coerce")
z_low_raw = train.assign(zval=z_raw)[z_raw < 0.1][
    ["VolumeName", "zval", "nslices", "is_no_chest", "is_unencodable", "mfr"]
]
spacing["z_low_investigation"] = {
    "clean_min_z": float(z_min_clean),
    "clean_n_below_0.1mm": int(len(z_low_clean)),
    "raw_min_z": float(z_raw.min()),
    "raw_n_below_0.1mm": int(len(z_low_raw)),
    "raw_below_0.1mm_examples": z_low_raw.astype(str).to_dict(orient="records"),
    "note": "0.035/0.075mm z from patient train_9792 (both no_chest, physically "
    "implausible: 0.035mm x 331 slices = 11.6mm coverage -> metadata error); "
    "clean-population Z min is 0.3mm.",
}
# >1000-slice cases list (train).
big = tc[tc["nslices"] > 1000][["VolumeName", "nslices", "z", "mfr"]].sort_values(
    "nslices", ascending=False
)
spacing["train_gt1000_slice_cases"] = big.head(30).astype(str).to_dict(orient="records")
# matrix x kernel cross (train).
mk = s_tr.copy()
mk["matrix"] = (
    mk["rows"].astype("Int64").astype(str)
    + "x"
    + mk["cols"].astype("Int64").astype(str)
)
spacing["train_matrix_x_kernel"] = {
    str(m): {
        str(k): int(v)
        for k, v in grp["ConvolutionKernel"].value_counts().head(5).to_dict().items()
    }
    for m, grp in mk.groupby("matrix")
}

raw["spacing_shape"] = spacing
pd.DataFrame(ss_rows).to_csv(OUT / "tables/spacing_shape.csv", index=False)
print(
    "XY median train:",
    spacing["train"]["xy_median"],
    "valid:",
    spacing["valid"]["xy_median"],
)
print(
    "Z median train:", spacing["train"]["z_median"], "min:", spacing["train"]["z_min"]
)
print(
    "Slices median train:",
    spacing["train"]["nslices_median"],
    "max:",
    spacing["train"]["nslices_max"],
)
print("Rescale intercept train:", spacing["train"]["rescale_intercept_counts"])


# ---------------------------------------------------------------- MISSINGNESS (volume-level)
META_COLS = [
    c
    for c in train.columns
    if c
    in [
        "Manufacturer",
        "SeriesDescription",
        "ManufacturerModelName",
        "PatientSex",
        "PatientAge",
        "ReconstructionDiameter",
        "DistanceSourceToDetector",
        "DistanceSourceToPatient",
        "GantryDetectorTilt",
        "TableHeight",
        "RotationDirection",
        "ExposureTime",
        "XRayTubeCurrent",
        "Exposure",
        "FilterType",
        "GeneratorPower",
        "FocalSpots",
        "ConvolutionKernel",
        "PatientPosition",
        "RevolutionTime",
        "SingleCollimationWidth",
        "TotalCollimationWidth",
        "TableSpeed",
        "TableFeedPerRotation",
        "SpiralPitchFactor",
        "DataCollectionCenterPatient",
        "ReconstructionTargetCenterPatient",
        "ExposureModulationType",
        "CTDIvol",
        "ImagePositionPatient",
        "ImageOrientationPatient",
        "SliceLocation",
        "SamplesPerPixel",
        "PhotometricInterpretation",
        "Rows",
        "Columns",
        "XYSpacing",
        "RescaleIntercept",
        "RescaleSlope",
        "RescaleType",
        "NumberofSlices",
        "ZSpacing",
        "StudyDate",
    ]
]


def missing_frac(df, col):
    s = df[col]
    m = s.isna() | (s.astype(str).str.strip() == "")
    return float(m.mean())


miss = {}
miss_rows = []
for name, df in [("train", tc), ("valid", vc)]:
    d = {c: round(100.0 * missing_frac(df, c), 2) for c in META_COLS}
    miss[name] = d
for c in META_COLS:
    miss_rows.append(
        {
            "column": c,
            "train_missing_pct": miss["train"][c],
            "valid_missing_pct": miss["valid"][c],
        }
    )
miss_df = pd.DataFrame(miss_rows).sort_values("train_missing_pct", ascending=False)
miss_df.to_csv(OUT / "tables/metadata_missingness.csv", index=False)

# Vendor-conditioned missingness (train, canonical mfr) — fingerprint detection.
vendor_miss = {}
for mfr, grp in s_tr.groupby("mfr"):
    vendor_miss[mfr] = {c: round(100.0 * missing_frac(grp, c), 1) for c in META_COLS}
# Flag columns whose missingness is a scanner fingerprint (variance across vendors high).
fingerprint = {}
for c in META_COLS:
    vals = [vendor_miss[m][c] for m in vendor_miss]
    if max(vals) - min(vals) > 50:  # >50pp swing across vendors
        fingerprint[c] = {m: vendor_miss[m][c] for m in vendor_miss}
miss["vendor_conditioned_train"] = vendor_miss
miss["scanner_fingerprint_columns"] = fingerprint
raw["missingness"] = miss
print("Top missing (train):")
print(miss_df.head(10).to_string(index=False))
print("Fingerprint cols:", list(fingerprint.keys()))

# Recommendation: safe-as-condition columns (low missing, not fingerprint).
safe_cols = [c for c in META_COLS if miss["train"][c] < 1.0 and c not in fingerprint]
raw["condition_safe_columns"] = safe_cols
print("Safe-as-condition columns:", safe_cols)


# ---------------------------------------------------------------- FIGURE
fig, axes = plt.subplots(3, 3, figsize=(18, 15))

# (0,0) Age hist (scan-level).
ax = axes[0, 0]
ax.hist(
    scan_level(tc)["age"].dropna(),
    bins=40,
    color=TRAIN_BLUE,
    alpha=0.6,
    density=True,
    label="train",
)
ax.hist(
    scan_level(vc)["age"].dropna(),
    bins=40,
    color=VALID_ORANGE,
    alpha=0.6,
    density=True,
    label="valid",
)
ax.set_title("PatientAge (scan-level)")
ax.set_xlabel("age (years)")
ax.set_ylabel("density")
ax.legend()

# (0,1) Sex (scan-level).
ax = axes[0, 1]
cats = ["M", "F", "missing"]
tv = [demo["train"]["sex_counts"].get(c, 0) for c in cats]
vv = [demo["valid"]["sex_counts"].get(c, 0) for c in cats]
x = np.arange(len(cats))
w = 0.35
ax.bar(x - w / 2, tv, w, color=TRAIN_BLUE, label="train")
ax.bar(x + w / 2, vv, w, color=VALID_ORANGE, label="valid")
ax.set_xticks(x)
ax.set_xticklabels(cats)
ax.set_title("PatientSex (scan-level)")
ax.set_ylabel("scans")
ax.legend()
ax.set_yscale("log")

# (0,2) Manufacturer canonical (scan-level).
ax = axes[0, 2]
mfrs = ["Philips", "Siemens", "PNMS", "missing"]
tv = [scanner["train"]["manufacturer_canon"].get(m, 0) for m in mfrs]
vv = [scanner["valid"]["manufacturer_canon"].get(m, 0) for m in mfrs]
x = np.arange(len(mfrs))
ax.bar(x - w / 2, tv, w, color=TRAIN_BLUE, label="train")
ax.bar(x + w / 2, vv, w, color=VALID_ORANGE, label="valid")
ax.set_xticks(x)
ax.set_xticklabels(mfrs, rotation=20)
ax.set_title("Manufacturer canonical (scan-level)")
ax.set_ylabel("scans")
ax.legend()
ax.set_yscale("log")

# (1,0) Kernel top (train scan-level).
ax = axes[1, 0]
kt_top = scan_level(tc)["ConvolutionKernel"].fillna("missing").value_counts().head(10)
ax.barh(range(len(kt_top)), kt_top.values[::-1], color=TRAIN_BLUE)
ax.set_yticks(range(len(kt_top)))
ax.set_yticklabels([str(k) for k in kt_top.index[::-1]], fontsize=7)
ax.set_title("Top ConvolutionKernel (train, scan-level)")
ax.set_xlabel("scans")

# (1,1) XY spacing (volume-level).
ax = axes[1, 1]
ax.hist(
    tc["xy"].dropna(), bins=50, color=TRAIN_BLUE, alpha=0.6, density=True, label="train"
)
ax.hist(
    vc["xy"].dropna(),
    bins=50,
    color=VALID_ORANGE,
    alpha=0.6,
    density=True,
    label="valid",
)
ax.set_title("XY spacing (volume-level)")
ax.set_xlabel("mm")
ax.set_ylabel("density")
ax.legend()

# (1,2) Z spacing (volume-level).
ax = axes[1, 2]
ax.hist(
    tc["z"].dropna().clip(upper=5),
    bins=50,
    color=TRAIN_BLUE,
    alpha=0.6,
    density=True,
    label="train",
)
ax.hist(
    vc["z"].dropna().clip(upper=5),
    bins=50,
    color=VALID_ORANGE,
    alpha=0.6,
    density=True,
    label="valid",
)
ax.set_title("Z spacing (volume-level, clip<=5mm)")
ax.set_xlabel("mm")
ax.set_ylabel("density")
ax.legend()

# (2,0) NumberofSlices (volume-level).
ax = axes[2, 0]
ax.hist(
    tc["nslices"].dropna().clip(upper=1200),
    bins=50,
    color=TRAIN_BLUE,
    alpha=0.6,
    density=True,
    label="train",
)
ax.hist(
    vc["nslices"].dropna().clip(upper=1200),
    bins=50,
    color=VALID_ORANGE,
    alpha=0.6,
    density=True,
    label="valid",
)
ax.set_title("NumberofSlices (volume-level, clip<=1200)")
ax.set_xlabel("slices")
ax.set_ylabel("density")
ax.legend()

# (2,1) Matrix (Rows x Columns, volume-level).
ax = axes[2, 1]
mc_tr = (
    (
        tc["rows"].astype("Int64").astype(str)
        + "x"
        + tc["cols"].astype("Int64").astype(str)
    )
    .value_counts()
    .head(6)
)
ax.bar(range(len(mc_tr)), mc_tr.values, color=TRAIN_BLUE)
ax.set_xticks(range(len(mc_tr)))
ax.set_xticklabels([str(k) for k in mc_tr.index], rotation=30, fontsize=8)
ax.set_title("Matrix RowsxCols (train, volume-level)")
ax.set_ylabel("volumes")
ax.set_yscale("log")

# (2,2) Missingness top-15 (volume-level).
ax = axes[2, 2]
top_miss = miss_df.head(15)
ax.barh(
    range(len(top_miss)), top_miss["train_missing_pct"].values[::-1], color=TRAIN_BLUE
)
ax.set_yticks(range(len(top_miss)))
ax.set_yticklabels(top_miss["column"].values[::-1], fontsize=7)
ax.set_title("Top-15 missing columns (train, volume-level)")
ax.set_xlabel("% missing")

fig.suptitle(
    "CT-RATE metadata overview (CLEAN population: no_chest + unencodable excluded)\n"
    f"train {len(tc)} vol / {tc['scan'].nunique()} scan / {tc['patient'].nunique()} pt   |   "
    f"valid {len(vc)} vol / {vc['scan'].nunique()} scan / {vc['patient'].nunique()} pt",
    fontsize=13,
)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT / "figures/metadata_overview.png", dpi=200)
plt.close(fig)


# ---------------------------------------------------------------- save raw json
def _sanitize(o):
    if isinstance(o, dict):
        return {str(k): _sanitize(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_sanitize(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


(OUT / "tables/metadata_raw.json").write_text(json.dumps(_sanitize(raw), indent=2))
print("\nDONE. Outputs written under", OUT)
