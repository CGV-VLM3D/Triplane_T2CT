"""Load EvalCase list from CT-RATE validation set for proxy evaluation.

Joins validation_reports.csv + validation_metadata.csv to build a list of
EvalCase objects and, optionally, converts the corresponding CT volumes to .mha
ground-truth files.

CT-RATE path convention:
  VolumeName = "valid_1_a_1.nii.gz"
  NIfTI path = valid_fixed / "valid_1" / "valid_1_a" / "valid_1_a_1.nii.gz"
               ┗── patient dir ──┘  ┗─── series dir ───┘
"""

from __future__ import annotations

import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk

from src.eval.samplers.base import EvalCase

log = logging.getLogger(__name__)

_CT_RATE_ROOT = Path("/workspace/datasets/datasets/CT-RATE/dataset")
_REPORTS_CSV = _CT_RATE_ROOT / "radiology_text_reports" / "validation_reports.csv"
_METADATA_CSV = _CT_RATE_ROOT / "metadata" / "validation_metadata.csv"
_VALID_FIXED = _CT_RATE_ROOT / "valid_fixed"


def _volume_name_to_nifti(volume_name: str) -> Path:
    """Convert VolumeName 'valid_1_a_1.nii.gz' → absolute NIfTI path."""
    stem = volume_name.replace(".nii.gz", "")
    parts = stem.rsplit("_", 1)  # ["valid_1_a", "1"]
    series_dir = parts[0]  # "valid_1_a"
    patient_dir = "_".join(series_dir.rsplit("_", 1)[:-1])  # "valid_1"
    return _VALID_FIXED / patient_dir / series_dir / volume_name


def _volume_name_to_id(volume_name: str) -> str:
    """Strip .nii.gz suffix to get the scan_id used for .mha filenames."""
    return volume_name.replace(".nii.gz", "")


def load_eval_cases(n_samples: int | None = None) -> list[EvalCase]:
    """Load CT-RATE validation cases as EvalCase objects.

    Args:
        n_samples: if given, cap at this many cases (deterministic head selection).

    Returns:
        list of EvalCase. NOTE: EvalCase.spacing_mm is NOT used for the saved affine — each
        sampler stamps its own true physical spacing (text2ct 0.75/0.75/3.0, generatect
        0.75/0.75/1.5, report2ct conditions+stamps its configured (1.0,1.0,1.5)). The field
        is kept only as a placeholder for a possible per-scan-conditioning contingency
        (plan silent-bug-floating-sunset.md, WI-3).
    """
    reports = pd.read_csv(_REPORTS_CSV)
    meta = pd.read_csv(_METADATA_CSV)

    # Merge on VolumeName; keep only rows where NIfTI actually exists
    df = reports.merge(meta[["VolumeName"]], on="VolumeName", how="left")
    df = df[df["VolumeName"].apply(lambda v: _volume_name_to_nifti(v).is_file())]

    if n_samples is not None:
        df = df.head(n_samples)

    cases = []
    for _, row in df.iterrows():
        cases.append(
            EvalCase(
                scan_id=_volume_name_to_id(row["VolumeName"]),
                findings=str(row.get("Findings_EN", "") or ""),
                impression=str(row.get("Impressions_EN", "") or ""),
                spacing_mm=[1.0, 1.0, 1.0],
            )
        )
    log.info("Loaded %d eval cases from CT-RATE validation.", len(cases))
    return cases


def prepare_proxy_gt(cases: list[EvalCase], gt_dir: Path) -> list[Path]:
    """Convert CT-RATE NIfTI volumes to .mha for use as proxy ground-truth.

    For each EvalCase, reads the corresponding NIfTI, clips HU to [-1000, 1000],
    casts to int16, and writes ``gt_dir/{case.scan_id}.mha``.

    Already-existing files are skipped (idempotent).

    Returns list of written/existing paths.
    """
    gt_dir = Path(gt_dir)
    gt_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for case in cases:
        out_path = gt_dir / f"{case.scan_id}.mha"
        if out_path.exists():
            written.append(out_path)
            continue

        nifti_path = _volume_name_to_nifti(f"{case.scan_id}.nii.gz")
        if not nifti_path.is_file():
            log.warning("NIfTI not found for %s — skipping GT prep.", case.scan_id)
            continue

        img_nib = nib.load(str(nifti_path))
        vol = img_nib.get_fdata(dtype=np.float32)
        hu = np.clip(vol, -1000.0, 1000.0).astype(np.int16)

        # NIfTI voxel order is (X, Y, Z); SimpleITK GetImageFromArray expects (Z, Y, X)
        arr_zyx = hu.transpose(2, 1, 0)
        sitk_img = sitk.GetImageFromArray(arr_zyx)

        zooms = img_nib.header.get_zooms()  # (sx, sy, sz) in mm
        sitk_img.SetSpacing([float(z) for z in zooms[:3]])

        sitk.WriteImage(sitk_img, str(out_path))
        written.append(out_path)
        log.debug("GT written: %s", out_path.name)

    log.info("Proxy GT: %d files ready in %s", len(written), gt_dir)
    return written


def write_prompt_xlsx(cases: list[EvalCase], out_path: Path) -> Path:
    """Write XLSX with (Names, Text_prompts) for evaluate_clip.py.

    'Names' = "<scan_id>.mha" (matches generated/.mha filenames).
    'Text_prompts' = findings + impression joined by single space.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "Names": f"{c.scan_id}.mha",
            "Text_prompts": (c.findings + " " + c.impression).strip(),
        }
        for c in cases
    ]
    pd.DataFrame(rows).to_excel(out_path, engine="openpyxl", index=False)
    log.info("Prompt XLSX written: %s (%d rows)", out_path, len(rows))
    return out_path


__all__ = ["load_eval_cases", "prepare_proxy_gt", "write_prompt_xlsx"]
