import sys, os, json, re

sys.path.insert(0, "/workspace")
import pandas as pd
import numpy as np

DS = "/workspace/datasets/datasets/CT-RATE/dataset"
LAB = f"{DS}/multi_abnormality_labels"
META = f"{DS}/metadata"


def load_nochest(split):
    fn = f"{META}/no_chest_{'train' if split == 'train' else 'valid'}.txt"
    with open(fn) as f:
        lines = [l.strip() for l in f if l.strip()]
    return set(os.path.basename(l) for l in lines)  # e.g. train_10100_a_1.nii.gz


UNENCODABLE = {
    "train_14384_a_2.nii.gz",
    "valid_251_a_2.nii.gz",
    "train_1267_a_4.nii.gz",
    "train_11755_a_3.nii.gz",
    "train_11755_a_4.nii.gz",
}

LABELS = [
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


def scan_key(vn):
    # train_1_a_1.nii.gz -> train_1_a
    b = vn.replace(".nii.gz", "")
    parts = b.split("_")
    return "_".join(parts[:3])


def analyze(split):
    csv = f"{LAB}/{'train_predicted_labels' if split == 'train' else 'valid_predicted_labels'}.csv"
    df = pd.read_csv(csv)
    n_vol_manifest = len(df)
    nochest = load_nochest(split)
    # clean population volume-level
    df["clean"] = ~df["VolumeName"].isin(nochest | UNENCODABLE)
    dfc = df[df["clean"]].copy()
    n_vol_clean = len(dfc)
    # scan-level: dedup, labels constant per scan -> take first
    dfc["scan"] = dfc["VolumeName"].map(scan_key)
    scan = dfc.groupby("scan")[LABELS].first()
    n_scan_clean = len(scan)
    prev = (scan[LABELS].mean() * 100).sort_values(ascending=False)
    allzero = (scan[LABELS].sum(axis=1) == 0).mean() * 100
    meanpos = scan[LABELS].sum(axis=1).mean()
    return dict(
        n_vol_manifest=n_vol_manifest,
        n_vol_clean=n_vol_clean,
        n_scan_clean=n_scan_clean,
        prev=prev,
        allzero=allzero,
        meanpos=meanpos,
        scan=scan,
    )


tr = analyze("train")
va = analyze("valid")

print("=== COUNTS ===")
print(
    f"train vol manifest={tr['n_vol_manifest']} vol_clean={tr['n_vol_clean']} scan_clean={tr['n_scan_clean']}"
)
print(
    f"valid vol manifest={va['n_vol_manifest']} vol_clean={va['n_vol_clean']} scan_clean={va['n_scan_clean']}"
)
print(f"\n=== all_zero_pct: train={tr['allzero']:.4f} valid={va['allzero']:.4f}")
print(f"=== mean_pos: train={tr['meanpos']:.4f} valid={va['meanpos']:.4f}")
print("\n=== TRAIN scan-clean prevalence (top) ===")
for k, v in tr["prev"].items():
    print(f"  {k}: {v:.4f}")

# scanner all_zero (train, clean, scan-level) — need Manufacturer from metadata
md = pd.read_csv(f"{META}/train_metadata.csv")
# map VolumeName->Manufacturer
man = dict(zip(md["VolumeName"], md["Manufacturer"]))
csv = f"{LAB}/train_predicted_labels.csv"
df = pd.read_csv(csv)
nochest = load_nochest("train")
df = df[~df["VolumeName"].isin(nochest | UNENCODABLE)].copy()
df["scan"] = df["VolumeName"].map(scan_key)
# per scan take first vol; use that vol's manufacturer
first = df.groupby("scan").first()
first["Manufacturer"] = df.groupby("scan")["VolumeName"].first().map(man)
first["az"] = first[LABELS].sum(axis=1) == 0
print("\n=== scanner all_zero_pct (train scan-clean) ===")
g = first.groupby("Manufacturer").agg(
    n=("az", "size"),
    az_pct=("az", lambda s: s.mean() * 100),
    lung_nodule=("Lung nodule", lambda s: s.mean() * 100),
)
print(g.to_string())

# co-occurrence phi: Arterial wall calc vs Coronary artery wall calc (train scan clean)
sc = tr["scan"]
a = sc["Arterial wall calcification"].astype(int).values
c = sc["Coronary artery wall calcification"].astype(int).values
n11 = int(((a == 1) & (c == 1)).sum())


def phi(x, y):
    n = len(x)
    n1_ = x.sum()
    n_1 = y.sum()
    n11 = ((x == 1) & (y == 1)).sum()
    num = n * n11 - n1_ * n_1
    den = np.sqrt(n1_ * n_1 * (n - n1_) * (n - n_1))
    return num / den


phi_ac = phi(a, c)
lo = sc["Lung opacity"].astype(int).values
cons = sc["Consolidation"].astype(int).values
phi_loc = phi(lo, cons)
print(f"\n=== phi Arterial+Coronary={phi_ac:.4f} joint_count={n11}")
print(f"=== phi LungOpacity+Consolidation={phi_loc:.4f}")

out = dict(
    train_scan_clean_n=int(tr["n_scan_clean"]),
    valid_scan_clean_n=int(va["n_scan_clean"]),
    train_vol_clean_n=int(tr["n_vol_clean"]),
    valid_vol_clean_n=int(va["n_vol_clean"]),
    allzero_train=float(tr["allzero"]),
    allzero_valid=float(va["allzero"]),
    meanpos_train=float(tr["meanpos"]),
    meanpos_valid=float(va["meanpos"]),
    phi_ac=float(phi_ac),
    phi_ac_joint=n11,
    phi_loc=float(phi_loc),
    prev_train={k: float(v) for k, v in tr["prev"].items()},
)
with open("/workspace/tests/ctrate_eda/tables/_verify_03_label.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nwrote _verify_03_label.json")
