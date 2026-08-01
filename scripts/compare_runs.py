"""Paired comparison between two eval runs' ``analysis/per_sample.csv`` (e.g. a baseline vs a
mask-conditioned run) — per-sample deltas + subgroup delta means. A standalone analysis step,
deliberately outside ``run_eval``/``orchestrate.py`` (plan §2/§5 Phase 5): it only ever reads two
already-produced ``per_sample.csv`` files, never regenerates or re-scores anything.

No confidence intervals (matches the ``subgroup.py`` decision — bootstrap/analytic CIs were
judged unnecessary and out of scope here); reports mean delta + n only.

Usage:
    python scripts/compare_runs.py \\
        --baseline outputs/report2ct_wan/eval_.../analysis/per_sample.csv \\
        --treatment outputs/report2ct_wan_mask/eval_.../analysis/per_sample.csv \\
        --out outputs/compare_wan_vs_wan_mask
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.eval.analysis.subgroup import METRIC_COLS

log = logging.getLogger(__name__)

# Same columns the subgroup tables aggregate, imported rather than re-listed: this file used to
# keep its own copy and silently produced no delta for hd_mm/assd_mm when those were added
# (2026-08-01). Columns absent from either run's CSV are skipped per-column below, so an old
# per_sample.csv still compares fine.
_DELTA_COLS = METRIC_COLS


def pair_runs(
    baseline_df: pd.DataFrame, treatment_df: pd.DataFrame, key: str = "target_id"
) -> pd.DataFrame:
    """Inner-join two per_sample tables on ``key``, logging any missing/mismatched samples.

    Args:
        baseline_df: baseline run's per_sample table.
        treatment_df: treatment (e.g. mask-conditioned) run's per_sample table.
        key: pairing key — ``target_id`` (not ``sample_id``, which diverges under mask
            intervention; plan §2 explicitly pairs on ``target_id`` [+ ``seed`` if needed]).

    Returns:
        Inner-joined DataFrame with ``_baseline``/``_treatment`` suffixed columns.
    """
    missing_in_treatment = set(baseline_df[key]) - set(treatment_df[key])
    missing_in_baseline = set(treatment_df[key]) - set(baseline_df[key])
    if missing_in_treatment:
        log.warning(
            "%d %s(s) in baseline missing from treatment (dropped): %s",
            len(missing_in_treatment),
            key,
            sorted(missing_in_treatment)[:5],
        )
    if missing_in_baseline:
        log.warning(
            "%d %s(s) in treatment missing from baseline (dropped): %s",
            len(missing_in_baseline),
            key,
            sorted(missing_in_baseline)[:5],
        )
    return baseline_df.merge(
        treatment_df, on=key, suffixes=("_baseline", "_treatment"), how="inner"
    )


def compute_deltas(merged: pd.DataFrame) -> pd.DataFrame:
    """Add ``delta_<metric> = treatment - baseline`` columns for every metric present on both sides."""
    for col in _DELTA_COLS:
        b, t = f"{col}_baseline", f"{col}_treatment"
        if b in merged.columns and t in merged.columns:
            merged[f"delta_{col}"] = merged[t] - merged[b]
    return merged


def _summarize_delta(sub: pd.DataFrame) -> dict:
    out: dict = {"n": len(sub)}
    for col in _DELTA_COLS:
        dcol = f"delta_{col}"
        if dcol not in sub.columns:
            continue
        vals = pd.to_numeric(sub[dcol], errors="coerce").dropna()
        out[f"{dcol}_mean"] = float(vals.mean()) if len(vals) else float("nan")
        out[f"{dcol}_n_valid"] = int(len(vals))
        out[f"{dcol}_n_missing"] = int(len(sub) - len(vals))
    return out


def _resolve_col(merged: pd.DataFrame, base_name: str) -> str | None:
    """Resolve a baseline-side column name after ``pair_runs``'s merge: pandas ONLY suffixes
    columns that collide between both frames — a column present only in ``baseline_df`` (no
    same-named column in ``treatment_df``) keeps its plain name, unsuffixed."""
    suffixed = f"{base_name}_baseline"
    if suffixed in merged.columns:
        return suffixed
    return base_name if base_name in merged.columns else None


def subgroup_deltas(merged: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Delta means over the label/burden/cluster columns carried on the baseline side (dataset
    properties — identical between runs for the same ``target_id``)."""
    results: dict[str, pd.DataFrame] = {}

    from src.eval.analysis.labels import LABEL_NAMES

    rows = []
    for name in LABEL_NAMES:
        col = _resolve_col(merged, f"label_{name}")
        if col is None:
            continue
        sub = merged[merged[col] == 1]
        rows.append({"label": name, **_summarize_delta(sub)})
    if rows:
        results["per_abnormality"] = pd.DataFrame(rows)

    burden_col = _resolve_col(merged, "burden_band")
    if burden_col is not None:
        rows = []
        for band in sorted(merged[burden_col].dropna().unique()):
            sub = merged[merged[burden_col] == band]
            rows.append({"band": band, **_summarize_delta(sub)})
        results["label_burden"] = pd.DataFrame(rows)

    # cluster names aren't a fixed constant (they come from subgroup config) — recover the
    # base names from whichever columns are present, suffixed or not.
    cluster_base_names = set()
    for c in merged.columns:
        if not c.startswith("cluster_"):
            continue
        base = c[: -len("_baseline")] if c.endswith("_baseline") else c
        base = base[: -len("_treatment")] if base.endswith("_treatment") else base
        cluster_base_names.add(base)

    rows = []
    for base in sorted(cluster_base_names):
        col = _resolve_col(merged, base)
        if col is None:
            continue
        name = base[len("cluster_") :]
        sub = merged[merged[col] == True]  # noqa: E712
        rows.append({"cluster": name, **_summarize_delta(sub)})
    if rows:
        results["organ_cluster"] = pd.DataFrame(rows)

    return results


def run(
    baseline_csv: str | Path, treatment_csv: str | Path, out_dir: str | Path
) -> dict[str, pd.DataFrame]:
    """Full comparison: pair, delta, subgroup-delta, write CSVs."""
    baseline_df = pd.read_csv(baseline_csv)
    treatment_df = pd.read_csv(treatment_csv)
    merged = pair_runs(baseline_df, treatment_df)
    merged = compute_deltas(merged)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_dir / "per_sample_delta.csv", index=False)

    tables = subgroup_deltas(merged)
    for name, df in tables.items():
        df.to_csv(out_dir / f"subgroup_delta_{name}.csv", index=False)

    overall = _summarize_delta(merged)
    log.info("Overall deltas (n=%d paired samples): %s", overall["n"], overall)
    return {"per_sample_delta": merged, **tables}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--baseline", required=True, help="baseline run's analysis/per_sample.csv"
    )
    ap.add_argument(
        "--treatment", required=True, help="treatment run's analysis/per_sample.csv"
    )
    ap.add_argument("--out", required=True, help="output directory for comparison CSVs")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)
    run(args.baseline, args.treatment, args.out)


if __name__ == "__main__":
    main()
