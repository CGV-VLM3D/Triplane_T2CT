"""Independent adversarial recompute of module 04_report headline numbers.
Different code path: direct CSV reads, own dedup. Read-only."""

import sys, json, re

sys.path.insert(0, "/workspace")
import pandas as pd, numpy as np

ROOT = "/workspace/datasets/datasets/CT-RATE/dataset"
FULL = "/workspace/data/ctrate_full"


def clean_scans(split):
    ids = json.load(open(f"{FULL}/{split}/ids.json"))["ids"]
    # scan_key = first 3 underscore tokens
    return set("_".join(v.split("_")[:3]) for v in ids)


def load(split):
    rep = "train_reports.csv" if split == "train" else "validation_reports.csv"
    df = pd.read_csv(f"{ROOT}/radiology_text_reports/{rep}")
    df["scan_key"] = (
        df["VolumeName"]
        .str.replace(".nii.gz", "", regex=False)
        .map(lambda v: "_".join(v.split("_")[:3]))
    )
    cs = clean_scans(split)
    df = df[df.scan_key.isin(cs)].copy()
    # dedup to scan: keep first by sorted VolumeName (match their keep=first after sort)
    df = (
        df.sort_values("VolumeName")
        .drop_duplicates("scan_key", keep="first")
        .reset_index(drop=True)
    )
    return df, len(cs)


NOT_GIVEN = re.compile(r"^\s*(not given|none|n/?a|-)\.?\s*$", re.IGNORECASE)


def missing(t):
    t = (str(t) if t == t else "").strip()
    return t == "" or bool(NOT_GIVEN.match(t))


def wc(t):
    return len((str(t) if t == t else "").split())


res = {}
for split in ["train", "valid"]:
    df, ncs = load(split)
    res[f"{split}_n_clean_scans_set"] = ncs
    res[f"{split}_n_deduped_rows"] = len(df)
    # clinical missing
    clin = df["ClinicalInformation_EN"]
    res[f"{split}_clinical_missing_pct"] = round(
        sum(missing(x) for x in clin) / len(df) * 100, 2
    )
    # findings words (non-missing)
    fnm = [wc(t) for t in df["Findings_EN"] if not missing(t)]
    res[f"{split}_findings_word_median"] = float(np.median(fnm))
    res[f"{split}_findings_iqr"] = [
        float(np.percentile(fnm, 25)),
        float(np.percentile(fnm, 75)),
    ]
    # impression
    inm = [wc(t) for t in df["Impressions_EN"] if not missing(t)]
    res[f"{split}_impression_word_median"] = float(np.median(inm))
    res[f"{split}_impression_missing_pct"] = round(
        sum(missing(x) for x in df["Impressions_EN"]) / len(df) * 100, 2
    )
    # exact-unique findings ratio (non-missing, normalized ws+lower)
    norm = [
        re.sub(r"\s+", " ", str(t).strip().lower())
        for t in df["Findings_EN"]
        if not missing(t)
    ]
    res[f"{split}_findings_exact_unique_pct"] = round(
        len(set(norm)) / len(norm) * 100, 2
    )

# boilerplate: count exact Findings sentence 'Pericardial effusion-thickening was not observed.' in train
dtr, _ = load("train")
SENT = re.compile(r"(?<=[.!?;])\s+")
cnt_peri = 0
labels = pd.read_csv(f"{ROOT}/multi_abnormality_labels/train_predicted_labels.csv")
labels["scan_key"] = (
    labels["VolumeName"]
    .str.replace(".nii.gz", "", regex=False)
    .map(lambda v: "_".join(v.split("_")[:3]))
)
for t in dtr["Findings_EN"]:
    if missing(t):
        continue
    for s in SENT.split(str(t).strip()):
        if s.strip() == "Pericardial effusion-thickening was not observed.":
            cnt_peri += 1
            break  # count reports containing it (matches 'count' = sentence occurrences? check both)
# also total occurrences
occ_peri = 0
for t in dtr["Findings_EN"]:
    if missing(t):
        continue
    for s in SENT.split(str(t).strip()):
        if s.strip() == "Pericardial effusion-thickening was not observed.":
            occ_peri += 1
res["peri_sentence_reports_with"] = cnt_peri
res["peri_sentence_total_occ"] = occ_peri

# burden band medians (train), merge labels
LAB = [c for c in labels.columns if c not in ("VolumeName", "scan_key")]
labdedup = labels.sort_values("VolumeName").drop_duplicates("scan_key", keep="first")
m = dtr.merge(labdedup[["scan_key"] + LAB], on="scan_key", how="inner")
m["nl"] = m[LAB].sum(axis=1)
m["fw"] = m["Findings_EN"].map(wc)
m["iw"] = m["Impressions_EN"].map(wc)
from scipy.stats import spearmanr

res["spearman_nl_findings"] = round(float(spearmanr(m.nl, m.fw)[0]), 3)
bands = {"all_zero": (0, 0), "1-3": (1, 3), "4-7": (4, 7), "8+": (8, 18)}
res["bands"] = {}
for b, (lo, hi) in bands.items():
    sub = m[(m.nl >= lo) & (m.nl <= hi)]
    res["bands"][b] = [int(np.median(sub.fw)), int(np.median(sub.iw)), len(sub)]

print(json.dumps(res, indent=2))
json.dump(res, open("/workspace/tests/ctrate_eda/tables/_verify_04_indep.json", "w"), indent=2)
