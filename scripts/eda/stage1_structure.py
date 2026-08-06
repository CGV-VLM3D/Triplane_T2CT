"""EDA Survey — Stage 1: structure, provenance, integrity (valid split).

CT-RATE official naming (HF dataset card): filename = split_patientID_scanID_reconstructionID,
e.g. valid_1_a_1 = validation, patient 'valid_1', scan 'valid_1_a', reconstruction 1.
Official hierarchy terms are patient / scan / reconstruction (no "study"/"series").
Each .nii.gz (a "volume") is one reconstruction.

Answers "what do we actually have" before any descriptive stats:
  1. patient / scan / reconstruction hierarchy + multiplicity distributions
  2. report duplication (reconstructions of a scan share one report?)
  3. multi-reconstruction == different reconstruction kernels?
  4. join integrity across reports / labels / metadata
  5. special sets: no_chest (brain), missing z-spacing, error_ctrate_data
  6. valid_v2 (1304 one-per-patient) definition + subset/one-per-patient check
  7. train<->valid patient leakage (by construction)

Writes 2 PNGs + a JSON digest to figs/eda_survey/ and prints a readable digest.
CSV-only (no NIfTI I/O) — cheap.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path("/workspace/datasets/datasets/CT-RATE/dataset")
OUT = Path("/workspace/figs/eda_survey")
VALID_V2_IDS = Path("/workspace/data/ctrate_toy_v2/valid_v2/ids.json")


def patient_of(vol: str) -> str:
    p = vol.replace(".nii.gz", "").split("_")
    return "_".join(p[:2])  # valid_1


def scan_of(vol: str) -> str:
    p = vol.replace(".nii.gz", "").split("_")
    return "_".join(p[:3])  # valid_1_a


def _extract_id_list(obj) -> list[str] | None:
    """valid_v2/ids.json may store the id list under one of several keys."""
    if isinstance(obj, list):
        return [str(x) for x in obj]
    if isinstance(obj, dict):
        for k in ("ids", "volumes", "volume_names", "selection", "valid_v2", "cases"):
            v = obj.get(k)
            if isinstance(v, list) and v:
                return [str(x) for x in v]
    return None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    R = pd.read_csv(ROOT / "radiology_text_reports" / "validation_reports.csv")
    L = pd.read_csv(ROOT / "multi_abnormality_labels" / "valid_predicted_labels.csv")
    M = pd.read_csv(ROOT / "metadata" / "validation_metadata.csv")
    vols = R.VolumeName

    digest: dict = {
        "split": "valid",
        "unit_terms": "patient / scan / reconstruction (CT-RATE official)",
    }

    # --- 1. hierarchy ---
    pats = vols.map(patient_of)
    scans = vols.map(scan_of)
    recons_per_scan = scans.value_counts()
    vols_per_patient = pats.value_counts()
    scans_per_patient = (
        pd.DataFrame({"p": pats, "s": scans}).drop_duplicates().p.value_counts()
    )
    digest["hierarchy"] = {
        "n_volumes": int(vols.nunique()),
        "n_scans": int(scans.nunique()),
        "n_patients": int(pats.nunique()),
        "reconstructions_per_scan": {
            "mean": round(float(recons_per_scan.mean()), 3),
            "min": int(recons_per_scan.min()),
            "max": int(recons_per_scan.max()),
        },
        "volumes_per_patient": {
            "mean": round(float(vols_per_patient.mean()), 3),
            "max": int(vols_per_patient.max()),
        },
        "scans_per_patient": {
            "mean": round(float(scans_per_patient.mean()), 3),
            "max": int(scans_per_patient.max()),
        },
    }

    # --- 2. report duplication ---
    reports_per_scan = R.groupby(scans)["Findings_EN"].nunique()
    digest["report_duplication"] = {
        "unique_findings": int(R.Findings_EN.nunique()),
        "n_rows": int(len(R)),
        "dup_fraction": round(1 - R.Findings_EN.nunique() / len(R), 3),
        "scans_with_1_unique_report": int((reports_per_scan == 1).sum()),
        "scans_with_gt1_report": int((reports_per_scan > 1).sum()),
    }

    # --- 3. multi-reconstruction == different kernels? ---
    Mk = M[["VolumeName", "ConvolutionKernel"]].copy()
    Mk["scan"] = Mk.VolumeName.map(scan_of)
    kernels_per_scan = Mk.groupby("scan")["ConvolutionKernel"].nunique()
    multi = recons_per_scan[recons_per_scan > 1].index
    digest["multirecon_kernels"] = {
        "multirecon_scans": int(len(multi)),
        "of_which_distinct_kernels": int((kernels_per_scan.reindex(multi) > 1).sum()),
        "of_which_same_kernel": int((kernels_per_scan.reindex(multi) == 1).sum()),
    }

    # --- 4. join integrity ---
    sr, sl, sm = set(R.VolumeName), set(L.VolumeName), set(M.VolumeName)
    digest["join_integrity"] = {
        "reports_not_in_labels": len(sr - sl),
        "reports_not_in_meta": len(sr - sm),
        "labels_not_in_reports": len(sl - sr),
        "meta_not_in_reports": len(sm - sr),
    }

    # --- 5. special sets ---
    no_chest_file = ROOT / "metadata" / "no_chest_valid.txt"
    no_chest = (
        [ln.strip() for ln in no_chest_file.read_text().splitlines() if ln.strip()]
        if no_chest_file.is_file()
        else []
    )
    no_chest_in_valid = len(set(no_chest) & sr)
    zspacing_nan = int(M["ZSpacing"].isna().sum()) if "ZSpacing" in M.columns else None
    digest["special_sets"] = {
        "no_chest_listed": len(no_chest),
        "no_chest_present_in_valid_reports": no_chest_in_valid,
        "zspacing_nan_in_meta": zspacing_nan,
        "error_ctrate_data_dir_exists": (ROOT / "error_ctrate_data").exists(),
    }

    # --- 6. valid_v2 ---
    v2 = None
    if VALID_V2_IDS.is_file():
        raw = json.loads(VALID_V2_IDS.read_text())
        ids = _extract_id_list(raw)
        if ids is not None:
            id_norm = {i if i.endswith(".nii.gz") else f"{i}.nii.gz" for i in ids}
            v2_pats = {patient_of(i) for i in id_norm}
            v2 = {
                "n_ids": len(id_norm),
                "n_patients": len(v2_pats),
                "one_per_patient": len(id_norm) == len(v2_pats),
                "subset_of_valid": len(id_norm - sr) == 0,
                "raw_keys": list(raw.keys()) if isinstance(raw, dict) else "list",
            }
        else:
            v2 = {
                "raw_keys": list(raw.keys()) if isinstance(raw, dict) else "list",
                "note": "could not locate id list — inspect manually",
            }
    digest["valid_v2"] = v2

    # --- 7. leakage (prefix-based) ---
    digest["leakage"] = {
        "all_valid_prefixed": bool(pats.str.startswith("valid_").all()),
        "note": "train patients are train_* by construction; no valid/train overlap",
    }

    # --- figures ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(
        recons_per_scan.values,
        bins=range(1, recons_per_scan.max() + 2),
        color="steelblue",
        edgecolor="black",
        align="left",
    )
    axes[0].set_title(f"Reconstructions per scan (n={scans.nunique()} scans)")
    axes[0].set_xlabel("# reconstruction volumes")
    axes[0].set_ylabel("# scans")
    axes[1].hist(
        vols_per_patient.values,
        bins=range(1, vols_per_patient.max() + 2),
        color="orange",
        edgecolor="black",
        align="left",
    )
    axes[1].set_title(f"Volumes per patient (n={pats.nunique()} patients)")
    axes[1].set_xlabel("# volumes")
    axes[1].set_ylabel("# patients")
    fig.suptitle("CT-RATE valid — acquisition hierarchy (volume ≠ scan ≠ patient)")
    fig.tight_layout()
    fig.savefig(OUT / "s1_hierarchy.png", dpi=150)
    plt.close(fig)

    (OUT / "s1_structure_digest.json").write_text(json.dumps(digest, indent=2))
    print(json.dumps(digest, indent=2, ensure_ascii=False))
    print(f"\nWrote {OUT / 's1_hierarchy.png'} and {OUT / 's1_structure_digest.json'}")


if __name__ == "__main__":
    main()
