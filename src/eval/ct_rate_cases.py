"""Load EvalCase list from CT-RATE validation set for evaluation.

Joins validation_reports.csv + validation_metadata.csv to build a list of
EvalCase objects and, optionally, converts the corresponding CT volumes to .mha
ground-truth files.

CT-RATE path convention:
  VolumeName = "valid_1_a_1.nii.gz"
  NIfTI path = valid_fixed / "valid_1" / "valid_1_a" / "valid_1_a_1.nii.gz"
               ┗── patient dir ──┘  ┗─── series dir ───┘
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import replace
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

# Frozen representative valid_v2 set (valid_fixed, one scan per patient = 1304),
# built by scripts/make_toy_dataset.py. When present, load_eval_cases restricts to these
# scan_ids instead of the legacy CSV-order head selection (which covered only ~425 patients).
# See plan: /root/.claude/plans/inherited-seeking-pike.md
_DEFAULT_VALID_V2_IDS = Path("/workspace/data/ctrate_toy_v2/valid_v2/ids.json")


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


def _clean_meta(value) -> str:
    """CT-RATE metadata cell → str, mapping NaN/None to '' (upstream uses 'None')."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value)


def _case_table() -> pd.DataFrame:
    """reports ⨝ metadata on VolumeName, keeping only rows whose NIfTI exists on disk.

    Shared by :func:`load_eval_cases` (plain runs) and :func:`load_manifest_cases`
    (mask-intervention runs) so both read the reports through exactly one code path.
    """
    reports = pd.read_csv(_REPORTS_CSV)
    meta = pd.read_csv(_METADATA_CSV)

    # Merge on VolumeName; carry PatientAge/PatientSex through for the GenerateCT prompt
    # template. Keep only rows where the NIfTI actually exists.
    meta_cols = ["VolumeName"] + [
        c for c in ("PatientAge", "PatientSex") if c in meta.columns
    ]
    df = reports.merge(meta[meta_cols], on="VolumeName", how="left")
    return df[df["VolumeName"].apply(lambda v: _volume_name_to_nifti(v).is_file())]


def _case_from_row(row) -> EvalCase:
    """One merged table row -> EvalCase (manifest fields left at their plain-run defaults)."""
    return EvalCase(
        scan_id=_volume_name_to_id(row["VolumeName"]),
        findings=str(row.get("Findings_EN", "") or ""),
        impression=str(row.get("Impressions_EN", "") or ""),
        spacing_mm=[
            1.0,
            1.0,
            1.0,
        ],  #  unused placeholder — samplers stamp their own spacing (see docstring/WI-3)
        age=_clean_meta(row.get("PatientAge")),
        sex=_clean_meta(row.get("PatientSex")),
    )


def load_eval_cases(
    n_samples: int | None = None,
    ids_json: str | Path | None = None,
) -> list[EvalCase]:
    """Load CT-RATE validation cases as EvalCase objects.

    Selection (representative valid_v2; replaces the old CSV-order head):
      * If ``ids_json`` resolves to an existing file (default: the frozen
        ``ctrate_toy_v2/valid_v2/ids.json`` = valid_fixed one-scan-per-patient = 1304),
        cases are restricted to those scan_ids. Otherwise it falls back to CSV order
        (legacy behaviour) so the function still works before the toy set is built.
      * ``n_samples`` caps the set via a SEEDED shuffle + head: the subset is representative
        (uniform over the frozen set), nested across sizes, and identical for every model
        (fair comparison). ``None`` = full set.

    Args:
        n_samples: cap on number of cases (None = all ~1304).
        ids_json: path to a frozen scan-id list (a dict with key "ids", or a bare list).
            None → the default frozen valid_v2 ids.

    Returns:
        list of EvalCase. NOTE: EvalCase.spacing_mm is NOT used for the saved affine — each
        sampler stamps its own true physical spacing (text2ct 0.75/0.75/3.0, generatect
        0.75/0.75/1.5, report2ct conditions+stamps its configured (1.0,1.0,1.5)). The field
        is kept only as a placeholder for a possible per-scan-conditioning contingency
        (plan silent-bug-floating-sunset.md, WI-3).
    """
    df = _case_table()

    # Restrict to the frozen representative valid_v2 set when available.
    ids_path = Path(ids_json) if ids_json is not None else _DEFAULT_VALID_V2_IDS
    if ids_path.is_file():
        frozen = json.loads(ids_path.read_text())
        frozen_ids = set(frozen["ids"] if isinstance(frozen, dict) else frozen)
        df = df[df["VolumeName"].apply(lambda v: _volume_name_to_id(v) in frozen_ids)]
        log.info(
            "Eval cases restricted to %d frozen valid_v2 ids (%s).",
            len(frozen_ids),
            ids_path,
        )
    else:
        log.warning(
            "Frozen valid_v2 ids not found at %s — falling back to CSV-order selection.",
            ids_path,
        )

    # Deterministic, representative subsample (seeded shuffle + head; nested across sizes).
    if n_samples is not None and n_samples < len(df):
        order = df["VolumeName"].tolist()
        random.Random(42).shuffle(order)
        keep = set(order[:n_samples])
        df = df[df["VolumeName"].isin(keep)]

    cases = [_case_from_row(row) for _, row in df.iterrows()]
    log.info("Loaded %d eval cases from CT-RATE validation.", len(cases))
    return cases


def load_manifest_cases(manifest_path: str | Path) -> list[EvalCase]:
    """Load one EvalCase per mask-intervention manifest ROW (not per scan).

    The same target appears in several rows, so the returned list holds several cases with the
    same ``scan_id`` — they differ in ``sample_id`` (output filename), ``cond_mask_source_id``
    (which scan's mask conditions the model) and ``condition``. Text conditioning always comes
    from the TARGET's report: a swapped mask is the only thing the intervention changes.

    Args:
        manifest_path: JSONL from ``scripts/build_mask_intervention_manifest.py``.

    Returns:
        list of EvalCase in manifest order, one per row.
    """
    from src.eval.manifest import read_manifest_rows

    rows = read_manifest_rows(manifest_path)
    wanted = {row["target_id"] for row in rows}
    df = _case_table()
    df = df[df["VolumeName"].apply(lambda v: _volume_name_to_id(v) in wanted)]
    by_id = {}
    for _, table_row in df.iterrows():
        case = _case_from_row(table_row)
        by_id[case.scan_id] = case

    unknown = sorted(wanted - set(by_id))
    if unknown:
        raise ValueError(
            f"{len(unknown)} manifest target(s) have no CT-RATE report/volume "
            f"(e.g. {unknown[:3]})"
        )

    cases = [
        replace(
            by_id[row["target_id"]],
            sample_id=row["sample_id"],
            condition=row["condition"],
            cond_mask_source_id=row["cond_mask_source_id"],
            seed=row["seed"],
        )
        for row in rows
    ]
    log.info("Loaded %d manifest cases over %d target scans.", len(cases), len(by_id))
    return cases


def prepare_valid_gt(cases: list[EvalCase], gt_dir: Path) -> list[Path]:
    """Convert CT-RATE NIfTI volumes to .mha for use as valid-set ground-truth.

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

    log.info("valid_v2 GT: %d files ready in %s", len(written), gt_dir)
    return written


def write_prompt_xlsx(cases: list[EvalCase], out_path: Path) -> Path:
    """Write XLSX with (Names, Text_prompts) for evaluate_clip.py.

    'Names' = "<out_stem>.mha" — the GENERATED file's stem, which upstream keys its prompt
    lookup on (``evaluate_clip.py::VolTextDS``). In a plain run that is the scan_id; in a
    mask-intervention run it is the sample_id, while the prompt text stays the TARGET's report
    (every condition of one target is scored against the same text).
    'Text_prompts' = findings + impression joined by single space.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "Names": f"{c.out_stem}.mha",
            "Text_prompts": (c.findings + " " + c.impression).strip(),
        }
        for c in cases
    ]
    pd.DataFrame(rows).to_excel(out_path, engine="openpyxl", index=False)
    log.info("Prompt XLSX written: %s (%d rows)", out_path, len(rows))
    return out_path


__all__ = [
    "load_eval_cases",
    "load_manifest_cases",
    "prepare_valid_gt",
    "write_prompt_xlsx",
]
