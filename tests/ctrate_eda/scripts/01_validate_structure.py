"""EDA v2 — Module 01: structure / integrity / quarantine verification.

Independently RE-COMPUTES the prior (GPT-5, CSV-only) EDA §2 numbers from the
original CT-RATE files and assigns a verdict (Verified / Minor / Major / Unable)
to each. GPT saw only CSVs; this also reconciles CSV manifest vs on-disk vs the
clean census (`ctrate_full`).

CT-RATE naming (HF card): VolumeName = split_patientID_scanID_reconstructionID,
e.g. valid_1_a_1 → patient 'valid_1', scan 'valid_1_a', reconstruction 1.
Read-only. No NIfTI voxel load (header/existence only).

Outputs (under /workspace/tests/ctrate_eda/):
  tables/dataset_counts.csv        — counts per split (manifest / on-disk / census)
  tables/dataset_counts_raw.json   — full machine-readable digest (pre-rounding)
  tables/discrepancy_seed.csv      — GPT value vs recomputed vs verdict
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path("/workspace/datasets/datasets/CT-RATE/dataset")
CTRATE_FULL = Path("/workspace/data/ctrate_full")
VALID_V2_IDS = Path("/workspace/data/ctrate_toy_v2/valid_v2/ids.json")
OUT = Path("/workspace/tests/ctrate_eda/tables")

LABELS = None  # filled from CSV columns


def patient_of(v: str) -> str:
    return "_".join(v.replace(".nii.gz", "").split("_")[:2])


def scan_of(v: str) -> str:
    return "_".join(v.replace(".nii.gz", "").split("_")[:3])


def recon_of(v: str) -> str:
    return v.replace(".nii.gz", "").split("_")[3]


def count_niftis(d: Path) -> set[str]:
    """basenames of all *.nii.gz under d (on-disk truth)."""
    out = set()
    if not d.exists():
        return out
    for _, _, files in os.walk(d):
        for f in files:
            if f.endswith(".nii.gz"):
                out.add(f)
    return out


def parse_no_chest(txt: Path) -> set[str]:
    """no_chest_*.txt stores v1 full paths (valid/valid_109/.../valid_109_a_1.nii.gz);
    reduce to bare VolumeName basename to match CSV VolumeName."""
    if not txt.is_file():
        return set()
    return {
        line.strip().split("/")[-1]
        for line in txt.read_text().splitlines()
        if line.strip()
    }


def verdict(prev, now, *, tol=0):
    if prev is None:
        return "Unable"
    if isinstance(prev, (int, float)) and isinstance(now, (int, float)):
        if prev == now:
            return "Verified"
        return "Minor" if abs(prev - now) <= tol else "Major"
    return "Verified" if prev == now else "Major"


def load_split(split: str) -> dict:
    if split == "train":
        rep, lab, meta = (
            "train_reports.csv",
            "train_predicted_labels.csv",
            "train_metadata.csv",
        )
        fixed_dir = ROOT / "train_fixed"
        nc = ROOT / "metadata" / "no_chest_train.txt"
    else:
        rep, lab, meta = (
            "validation_reports.csv",
            "valid_predicted_labels.csv",
            "validation_metadata.csv",
        )
        fixed_dir = ROOT / "valid_fixed"
        nc = ROOT / "metadata" / "no_chest_valid.txt"
    R = pd.read_csv(ROOT / "radiology_text_reports" / rep)
    L = pd.read_csv(ROOT / "multi_abnormality_labels" / lab)
    M = pd.read_csv(ROOT / "metadata" / meta)
    global LABELS
    LABELS = [c for c in L.columns if c != "VolumeName"]

    vols = pd.Index(R.VolumeName.unique())
    pats = vols.map(patient_of)
    scans = vols.map(scan_of)
    d: dict = {"split": split}

    # --- counts (manifest) ---
    d["manifest"] = {
        "volumes": int(vols.nunique()),
        "scans": int(pd.Index(scans).nunique()),
        "patients": int(pd.Index(pats).nunique()),
    }

    # --- matching across the 3 CSVs ---
    sr, sl, sm = set(R.VolumeName), set(L.VolumeName), set(M.VolumeName)
    d["matching"] = {
        "reports_rows": int(len(R)),
        "reports_unique": int(vols.nunique()),
        "labels_rows": int(len(L)),
        "meta_rows": int(len(M)),
        "reports_not_in_labels": len(sr - sl),
        "reports_not_in_meta": len(sr - sm),
        "labels_not_in_reports": len(sl - sr),
        "meta_not_in_reports": len(sm - sr),
    }

    # --- reconstruction-per-scan distribution ---
    recons_per_scan = pd.Series(scans).value_counts()
    d["recon_dist"] = {
        int(k): int(v) for k, v in sorted(Counter(recons_per_scan.values).items())
    }
    d["pct_scan_with_2_recon"] = round(100 * (recons_per_scan == 2).mean(), 2)

    # --- scan-internal consistency: report & label identical across recons ---
    rep_key = R[["VolumeName", "Findings_EN", "Impressions_EN"]].copy()
    rep_key["scan"] = rep_key.VolumeName.map(scan_of)
    rep_nuniq = rep_key.groupby("scan")[["Findings_EN", "Impressions_EN"]].nunique()
    d["scan_report_inconsistent"] = int(((rep_nuniq > 1).any(axis=1)).sum())
    Lk = L.copy()
    Lk["scan"] = Lk.VolumeName.map(scan_of)
    lab_nuniq = Lk.groupby("scan")[LABELS].apply(lambda g: g.drop_duplicates().shape[0])
    d["scan_label_inconsistent"] = int((lab_nuniq > 1).sum())

    # --- no_chest ---
    ncs = parse_no_chest(nc)
    nc_in_manifest = ncs & sr
    nc_scans = {scan_of(v) for v in nc_in_manifest}
    nc_pats = {patient_of(v) for v in nc_in_manifest}
    d["no_chest"] = {
        "listed": len(ncs),
        "in_manifest": len(nc_in_manifest),
        "scans": len(nc_scans),
        "patients": len(nc_pats),
        "pct_of_volumes": round(100 * len(nc_in_manifest) / vols.nunique(), 3),
    }

    # --- on-disk vs manifest ---
    disk = count_niftis(fixed_dir)
    nc_disk = count_niftis(ROOT / "no_chest_data" / fixed_dir.name)
    err_disk = count_niftis(ROOT / "error_ctrate_data" / fixed_dir.name)
    moved = sr - disk
    d["on_disk"] = {
        "fixed_dir_volumes": len(disk),
        "manifest_minus_disk": len(moved),
        "in_no_chest_data": len(nc_disk),
        "in_error_ctrate_data": len(err_disk),
        "moved_not_in_quarantine": sorted(moved - nc_disk - err_disk),
    }
    d["_sets"] = {"manifest": sr, "disk": disk}  # for leakage; stripped before json
    return d


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tr = load_split("train")
    va = load_split("valid")

    # --- leakage: patient overlap train vs valid ---
    tr_pats = {patient_of(v) for v in tr["_sets"]["manifest"]}
    va_pats = {patient_of(v) for v in va["_sets"]["manifest"]}
    leakage = {
        "train_patients": len(tr_pats),
        "valid_patients": len(va_pats),
        "overlap": len(tr_pats & va_pats),
    }

    # --- ctrate_full clean census ---
    census = {}
    cf = sorted(CTRATE_FULL.glob("*.json"))
    if cf:
        raw = json.loads(cf[0].read_text())
        census = raw.get("splits", raw)

    # --- valid_v2 ---
    v2 = {}
    if VALID_V2_IDS.is_file():
        raw = json.loads(VALID_V2_IDS.read_text())
        ids = raw.get("ids") if isinstance(raw, dict) else raw
        if ids:
            norm = {i if i.endswith(".nii.gz") else f"{i}.nii.gz" for i in ids}
            v2 = {
                "n": len(norm),
                "patients": len({patient_of(i) for i in norm}),
                "one_per_patient": len(norm) == len({patient_of(i) for i in norm}),
                "subset_of_disk": len(norm - va["_sets"]["disk"]) == 0,
            }

    for d in (tr, va):
        d.pop("_sets", None)
    digest = {
        "train": tr,
        "valid": va,
        "leakage": leakage,
        "ctrate_full_census": census,
        "valid_v2": v2,
    }

    # --- discrepancy table vs GPT §2 claims ---
    GPT = {
        "train.patients": 20000,
        "train.scans": 24128,
        "train.volumes": 47149,
        "valid.patients": 1304,
        "valid.scans": 1564,
        "valid.volumes": 3039,
        "total.patients": 21304,
        "total.scans": 25692,
        "total.volumes": 50188,
        "orphans": 0,
        "recon_label_mismatch": 0,
        "no_chest.train": 752,
        "no_chest.valid": 37,
        "no_chest.total": 789,
        "no_chest.scans": 260,
        "no_chest.patients": 258,
    }
    now = {
        "train.patients": tr["manifest"]["patients"],
        "train.scans": tr["manifest"]["scans"],
        "train.volumes": tr["manifest"]["volumes"],
        "valid.patients": va["manifest"]["patients"],
        "valid.scans": va["manifest"]["scans"],
        "valid.volumes": va["manifest"]["volumes"],
        "total.patients": tr["manifest"]["patients"] + va["manifest"]["patients"],
        "total.scans": tr["manifest"]["scans"] + va["manifest"]["scans"],
        "total.volumes": tr["manifest"]["volumes"] + va["manifest"]["volumes"],
        "orphans": tr["matching"]["reports_not_in_labels"]
        + va["matching"]["reports_not_in_labels"],
        "recon_label_mismatch": tr["scan_label_inconsistent"]
        + va["scan_label_inconsistent"],
        "no_chest.train": tr["no_chest"]["in_manifest"],
        "no_chest.valid": va["no_chest"]["in_manifest"],
        "no_chest.total": tr["no_chest"]["in_manifest"] + va["no_chest"]["in_manifest"],
        "no_chest.scans": tr["no_chest"]["scans"] + va["no_chest"]["scans"],
        "no_chest.patients": tr["no_chest"]["patients"] + va["no_chest"]["patients"],
    }
    disc = pd.DataFrame(
        [
            {
                "metric": k,
                "gpt_value": GPT[k],
                "recomputed": now[k],
                "verdict": verdict(GPT[k], now[k]),
            }
            for k in GPT
        ]
    )
    disc.to_csv(OUT / "discrepancy_seed.csv", index=False)

    # --- counts table (manifest vs on-disk vs census) ---
    rows = []
    for name, d in (("train", tr), ("valid", va)):
        rows.append(
            {"split": name, "level": "manifest_volumes", "n": d["manifest"]["volumes"]}
        )
        rows.append(
            {"split": name, "level": "manifest_scans", "n": d["manifest"]["scans"]}
        )
        rows.append(
            {
                "split": name,
                "level": "manifest_patients",
                "n": d["manifest"]["patients"],
            }
        )
        rows.append(
            {
                "split": name,
                "level": "on_disk_volumes",
                "n": d["on_disk"]["fixed_dir_volumes"],
            }
        )
        rows.append(
            {
                "split": name,
                "level": "quarantined_no_chest",
                "n": d["on_disk"]["in_no_chest_data"],
            }
        )
    pd.DataFrame(rows).to_csv(OUT / "dataset_counts.csv", index=False)
    (OUT / "dataset_counts_raw.json").write_text(json.dumps(digest, indent=2))

    print(json.dumps(digest, indent=2, ensure_ascii=False))
    print("\n=== DISCREPANCY vs GPT §2 ===")
    print(disc.to_string(index=False))
    print(
        f"\nWrote {OUT}/dataset_counts.csv, discrepancy_seed.csv, dataset_counts_raw.json"
    )


if __name__ == "__main__":
    main()
