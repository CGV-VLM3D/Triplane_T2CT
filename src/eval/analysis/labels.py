"""18-abnormality label loading and per-sample derived fields (burden, normal/disease, cluster membership).

Foundational module for the per-sample / subgroup evaluation extension (plan:
/root/.claude/plans/repository-run-eval-jazzy-twilight.md). Every downstream module
(persample.py, subgroup.py, subgroup_setlevel.py) derives its label-related columns from
the functions here, computed once per sample and never re-derived.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.tasks._fvd_ctclip import PATHOLOGIES

# Re-exported so callers don't need to know the constant's true home (_fvd_ctclip.py, where
# it is load-bearing for the 18-d FVD profile axes). Same 18 names, same order, single source
# of truth — DO NOT redefine this list elsewhere.
LABEL_NAMES: tuple[str, ...] = PATHOLOGIES

_CT_RATE_ROOT = Path("/workspace/datasets/datasets/CT-RATE/dataset")
_LABEL_CSV = {
    "train": _CT_RATE_ROOT / "multi_abnormality_labels" / "train_predicted_labels.csv",
    "valid": _CT_RATE_ROOT / "multi_abnormality_labels" / "valid_predicted_labels.csv",
}


def load_label_csv(split: str = "valid") -> dict[str, np.ndarray]:
    """Load the 18-label CSV for one CT-RATE split, keyed by scan_id.

    Args:
        split: ``"train"`` or ``"valid"``.

    Returns:
        dict mapping scan_id (``VolumeName`` minus ``.nii.gz``, e.g. ``"valid_1_a_1"``) to
        an int array ``(18,)`` in :data:`LABEL_NAMES` order (values are ``0``/``1`` — the
        CT-RATE label CSVs carry no uncertain/missing values, verified 2026-07-26).
    """
    csv_path = _LABEL_CSV[split]
    df = pd.read_csv(csv_path)
    scan_ids = df["VolumeName"].str.replace(".nii.gz", "", regex=False)
    vecs = df[list(LABEL_NAMES)].to_numpy(dtype=np.int64)
    return dict(zip(scan_ids, vecs))


def label_burden(vec: np.ndarray) -> int:
    """Count of positive labels in an 18-vector."""
    return int(vec.sum())


def label_class(vec: np.ndarray) -> str:
    """``"normal"`` if all 18 labels are zero, else ``"disease"``."""
    return "normal" if label_burden(vec) == 0 else "disease"


def burden_band(burden: int, bands: dict[str, tuple[int, int]]) -> str | None:
    """Map a label-burden count to its configured band name.

    Args:
        burden: positive-label count (0-18).
        bands: band name -> ``(lo, hi)`` inclusive range, e.g. ``{"all_zero": (0, 0), ...}``
            (see ``configs/eval/subgroup/default.yaml``).

    Returns:
        The matching band name, or ``None`` if no band covers ``burden`` (should not happen
        for a correctly-specified config spanning 0-18; callers should treat ``None`` as a
        config-validation failure, not a silent skip).
    """
    for name, (lo, hi) in bands.items():
        if lo <= burden <= hi:
            return name
    return None


def cluster_membership(
    vec: np.ndarray, clusters: dict[str, list[str]]
) -> dict[str, bool]:
    """Overlapping organ-cluster membership for one sample's label vector.

    A sample belongs to a cluster if ANY of that cluster's labels is positive
    (multi-membership — a sample may belong to several clusters at once, per the
    user-confirmed subgroup policy in the plan).

    Args:
        vec: label vector ``(18,)`` in :data:`LABEL_NAMES` order.
        clusters: cluster name -> list of label names (see
            ``configs/eval/subgroup/default.yaml``).

    Returns:
        dict mapping cluster name -> bool (True if the sample has >=1 positive label in
        that cluster).
    """
    name_to_idx = {name: i for i, name in enumerate(LABEL_NAMES)}
    membership = {}
    for cluster_name, label_list in clusters.items():
        idxs = [name_to_idx[label] for label in label_list]
        membership[cluster_name] = bool(vec[idxs].any())
    return membership


def patient_id_of(scan_id: str) -> str:
    """Derive the patient id from a scan id (``"valid_1_a_1"`` -> ``"valid_1"``).

    Mirrors the series/patient-dir parsing in
    ``src/eval/ct_rate_cases.py::_volume_name_to_nifti`` (two trailing ``_<token>`` segments
    are the series letter and reconstruction index; what remains is the patient id) — kept
    as a small, separate copy here since importing a private (``_``-prefixed) helper from
    that module would be non-idiomatic.

    Args:
        scan_id: e.g. ``"valid_1000_a_1"``.

    Returns:
        Patient id, e.g. ``"valid_1000"``.
    """
    series_id = scan_id.rsplit("_", 1)[0]  # drop reconstruction index -> "valid_1000_a"
    return series_id.rsplit("_", 1)[0]  # drop series letter -> "valid_1000"


__all__ = [
    "LABEL_NAMES",
    "load_label_csv",
    "label_burden",
    "label_class",
    "burden_band",
    "cluster_membership",
    "patient_id_of",
]
