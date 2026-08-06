import sys, os, re, json

sys.path.insert(0, "/workspace")
import numpy as np, pandas as pd

ROOT = "/workspace/datasets/datasets/CT-RATE/dataset"
META = f"{ROOT}/metadata"


def load_nochest(p):
    ids = set()
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(os.path.basename(line).replace(".nii.gz", ""))
    return ids


nc_train = load_nochest(f"{META}/no_chest_train.txt")
nc_valid = load_nochest(f"{META}/no_chest_valid.txt")
unenc = {
    "train_14384_a_2",
    "valid_251_a_2",
    "train_1267_a_4",
    "train_11755_a_3",
    "train_11755_a_4",
}
print("no_chest_train unique basenames:", len(nc_train), "valid:", len(nc_valid))


def vol_id(v):
    return v.replace(".nii.gz", "")


def scan_key(v):  # patient_scan
    m = re.match(r"((?:train|valid)_\d+_[a-z])_\d+", vol_id(v))
    return m.group(1)


def patient_key(v):
    m = re.match(r"((?:train|valid)_\d+)_[a-z]_\d+", vol_id(v))
    return m.group(1)


out = {}
for split, mfile in [
    ("train", "train_metadata.csv"),
    ("valid", "validation_metadata.csv"),
]:
    df = pd.read_csv(f"{META}/{mfile}")
    df["vid"] = df["VolumeName"].map(vol_id)
    nc = nc_train if split == "train" else nc_valid
    excl = nc | unenc
    n_raw = len(df)
    clean = df[~df["vid"].isin(excl)].copy()
    clean["scan"] = clean["VolumeName"].map(scan_key)
    clean["patient"] = clean["VolumeName"].map(patient_key)
    nvol, nscan, npat = len(clean), clean["scan"].nunique(), clean["patient"].nunique()
    # scan-level dedup
    scandf = clean.drop_duplicates("scan")
    # manufacturer raw scan-level
    man = scandf["Manufacturer"].astype(str).str.strip().value_counts().to_dict()
    # sex scan-level
    sex = scandf["PatientSex"].astype(str).str.strip()
    sexcnt = sex.value_counts(dropna=False).to_dict()
    # spacing volume-level
    xy = pd.to_numeric(
        clean["XYSpacing"].astype(str).str.split(",").str[0].str.strip("[] "),
        errors="coerce",
    )
    # XYSpacing may be like "[0.68, 0.68]"
    xy = xy.dropna()
    z = pd.to_numeric(clean["ZSpacing"], errors="coerce").dropna()
    zvc = z.round(3).value_counts().head(5).to_dict()
    ri = (
        pd.to_numeric(clean["RescaleIntercept"], errors="coerce")
        .value_counts()
        .to_dict()
    )
    out[split] = dict(
        n_raw=n_raw,
        nvol=nvol,
        nscan=nscan,
        npat=npat,
        excl_nochest=len(nc),
        excl_total_in_df=int(df["vid"].isin(excl).sum()),
        man=man,
        sex=sexcnt,
        xy_median=float(xy.median()),
        z_median=float(z.median()),
        zvc={str(k): int(v) for k, v in zvc.items()},
        ri={str(k): int(v) for k, v in ri.items()},
    )

print(json.dumps(out, indent=1, default=str))
