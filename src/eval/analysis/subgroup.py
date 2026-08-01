"""Subgroup summaries over ``per_sample.csv`` — per-abnormality (A) / label-burden (B) /
organ-cluster (C) axes.

Pure pandas filter+aggregate over the already-computed per-sample metrics; nothing here
re-runs CLIP, segmentation, or any other metric. Mask-explainability (axis D) is deferred (see
plan §3.4/§3.6 — the user wants to inspect segmentation maps before defining it), so it has no
function here. No confidence intervals (user decision 2026-07-27): bootstrap CIs are unsuited to
FID-family metrics and add runtime for no benefit here; analytic CIs were judged unnecessary too.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.eval.analysis.labels import LABEL_NAMES
from src.eval.analysis.seg_metrics import SURFACE_METRICS
from src.eval.analysis.subgroup_config import SubgroupConfig

# The per-sample metric columns every subgroup aggregate is computed over. Derived from the metric
# families in seg_metrics rather than hand-listed, so a new family propagates here AND to
# scripts/compare_runs.py (which imports this list) — a hand-maintained copy there silently
# dropped hd_mm/assd_mm when they were added, 2026-08-01.
METRIC_COLS = (
    ["clip_t2i", "clip_i2i"]
    + [
        f"{family}_to_{ref}"
        for family in ("dice", *SURFACE_METRICS)
        for ref in ("input_mask", "gt_mask")
    ]
    # Not a quality metric — the surface-distance means' own denominator. A subgroup whose mean
    # here is below 4 scored fewer organs than one at 4, which is exactly when those means stop
    # being comparable (seg_metrics._surface_metrics_per_organ). Which organs dropped out is in
    # per_sample.csv's `organs_missing_to_*` (a string, so it cannot be averaged here).
    + [f"n_organs_scored_to_{ref}" for ref in ("input_mask", "gt_mask")]
)


def _summarize(sub_df: pd.DataFrame) -> dict:
    """mean/std/median/n_valid/n_missing per metric column, over one subgroup's rows."""
    out: dict = {"n": len(sub_df)}
    for col in METRIC_COLS:
        vals = (
            pd.to_numeric(sub_df[col], errors="coerce").dropna()
            if col in sub_df
            else pd.Series(dtype=float)
        )
        n_valid = len(vals)
        n_missing = len(sub_df) - n_valid
        out[f"{col}_mean"] = float(vals.mean()) if n_valid else float("nan")
        out[f"{col}_std"] = float(vals.std()) if n_valid > 1 else float("nan")
        out[f"{col}_median"] = float(vals.median()) if n_valid else float("nan")
        out[f"{col}_n_valid"] = n_valid
        out[f"{col}_n_missing"] = n_missing
    return out


def per_abnormality(df: pd.DataFrame) -> pd.DataFrame:
    """Axis (A): one row per of the 18 labels, over its positive samples. Always all 18 rows,
    n small or not (requirement C.A — n is reported, not hidden)."""
    rows = []
    for name in LABEL_NAMES:
        col = f"label_{name}"
        sub = df[df[col] == 1] if col in df else df.iloc[0:0]
        rows.append({"label": name, **_summarize(sub)})
    return pd.DataFrame(rows)


def label_burden(df: pd.DataFrame, bands: dict[str, tuple[int, int]]) -> pd.DataFrame:
    """Axis (B): one row per configured burden band."""
    rows = []
    for band in bands:
        sub = df[df["burden_band"] == band]
        rows.append({"band": band, **_summarize(sub)})
    return pd.DataFrame(rows)


def organ_cluster(df: pd.DataFrame, clusters: dict[str, list[str]]) -> pd.DataFrame:
    """Axis (C): one row per organ cluster (overlapping membership — a sample may appear in
    more than one cluster's row, per the user-confirmed subgroup policy)."""
    rows = []
    for name in clusters:
        col = f"cluster_{name}"
        sub = df[df[col] == True] if col in df else df.iloc[0:0]  # noqa: E712
        rows.append({"cluster": name, **_summarize(sub)})
    return pd.DataFrame(rows)


def run(
    per_sample_csv: str | Path, subgroup_cfg: SubgroupConfig, out_dir: str | Path
) -> dict[str, pd.DataFrame]:
    """Compute all three subgroup axes and write ``analysis/subgroup/{name}.csv`` each.

    Args:
        per_sample_csv: path to ``analysis/per_sample.csv``.
        subgroup_cfg: loaded subgroup definitions.
        out_dir: ``analysis/subgroup/`` directory.

    Returns:
        ``{"per_abnormality": df, "label_burden": df, "organ_cluster": df}``.
    """
    df = pd.read_csv(per_sample_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "per_abnormality": per_abnormality(df),
        "label_burden": label_burden(df, subgroup_cfg.label_burden_bands),
        "organ_cluster": organ_cluster(df, subgroup_cfg.organ_clusters),
    }
    for name, table in results.items():
        table.to_csv(out_dir / f"{name}.csv", index=False)
    return results


__all__ = ["METRIC_COLS", "per_abnormality", "label_burden", "organ_cluster", "run"]
