"""Write ``analysis/SUMMARY.md`` — a one-glance rollup of everything this extension produces.

Pulls from whatever artifacts exist (headline metrics.json, per_sample.csv, subgroup/*.csv,
setlevel/*.csv, figures/) — sections for missing artifacts are simply omitted, so this can be
called after any subset of the new flags ran.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _fmt(value, digits=4) -> str:
    if value is None:
        return "n/a"
    try:
        if pd.isna(value):
            return "NaN"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _headline_section(metrics_json: Path) -> list[str]:
    if not metrics_json.is_file():
        return []
    metrics = json.loads(metrics_json.read_text())
    lines = ["## Headline VLM3D metrics (`metrics.json`)", ""]
    for key, value in metrics.items():
        # `_history` (and any future `_`-prefixed bookkeeping) records WHEN each metric was
        # scored, not a metric — rendering it here would dump a nested list into the summary.
        if key.startswith("_"):
            continue
        lines.append(f"- **{key}**: {_fmt(value)}")
    lines.append("")
    return lines


def _setlevel_section(analysis_dir: Path) -> list[str]:
    path = analysis_dir / "setlevel" / "setlevel_fid_fvd.csv"
    if not path.is_file():
        return []
    df = pd.read_csv(path)
    lines = ["## Set-level FID/FVD by subgroup (`setlevel/setlevel_fid_fvd.csv`)", ""]
    core = df[df["axis"].isin(["overall", "normal", "disease"])]
    lines.append(
        "| axis | FID_2p5D_Avg | real_n | gen_n | real_patients | below_threshold |"
    )
    lines.append("|---|---|---|---|---|---|")
    for _, row in core.iterrows():
        lines.append(
            f"| {row['axis']} | {_fmt(row.get('FID_2p5D_Avg'))} | {int(row.get('real_n', 0))} "
            f"| {int(row.get('gen_n', 0))} | {int(row.get('real_patients', 0))} "
            f"| {row.get('below_threshold')} |"
        )
    n_other = len(df) - len(core)
    if n_other:
        lines.append(
            f"\n(+{n_other} more rows — per-label/cluster/burden — see the CSV directly.)"
        )
    lines.append("")
    return lines


def _subgroup_section(
    title: str, csv_path: Path, key_col: str, top_n: int = 6
) -> list[str]:
    if not csv_path.is_file():
        return []
    df = pd.read_csv(csv_path)
    lines = [f"## {title} (`{csv_path.name}`)", ""]
    cols = [
        key_col,
        "n",
        "clip_t2i_mean",
        "dice_to_gt_mask_mean",
        "dice_to_input_mask_mean",
    ]
    cols = [c for c in cols if c in df.columns]
    shown = (
        df.sort_values("n", ascending=False).head(top_n)
        if "n" in df.columns
        else df.head(top_n)
    )
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * len(cols))
    for _, row in shown.iterrows():
        lines.append(
            "| "
            + " | ".join(_fmt(row[c]) if c != key_col else str(row[c]) for c in cols)
            + " |"
        )
    if len(df) > top_n:
        lines.append(f"\n(showing {top_n}/{len(df)} rows by n — see the CSV for all.)")
    lines.append("")
    return lines


def _mask_section(per_sample_csv: Path) -> list[str]:
    if not per_sample_csv.is_file():
        return []
    df = pd.read_csv(per_sample_csv)
    if (
        "dice_to_input_mask" not in df.columns
        or df["dice_to_input_mask"].dropna().empty
    ):
        return []
    organs = ["lung", "heart", "aorta", "esophagus"]
    lines = ["## Mask-following Dice (input mask vs target GT mask)", ""]
    lines.append("| organ | dice_to_input_mask | dice_to_gt_mask |")
    lines.append("|---|---|---|")
    for organ in organs:
        col_in = f"dice_to_input_mask_{organ}"
        col_gt = f"dice_to_gt_mask_{organ}"
        if col_in not in df.columns:
            continue
        lines.append(
            f"| {organ} | {_fmt(df[col_in].mean())} | {_fmt(df[col_gt].mean())} |"
        )
    lines.append("")
    return lines


def _figures_section(figures_dir: Path) -> list[str]:
    cases_json = figures_dir / "cases.json"
    if not cases_json.is_file():
        return []
    cases = json.loads(cases_json.read_text())
    lines = ["## QC figures (`figures/`)", ""]
    for c in cases:
        lines.append(f"- `{c['case']}.png` — {c['label'] or 'normal (all-zero)'}")
    lines.append("")
    return lines


def write_summary(out_dir: str | Path, metrics_path: str | Path | None = None) -> Path:
    """Assemble ``analysis/SUMMARY.md`` from whatever artifacts exist under ``out_dir``.

    Args:
        out_dir: the eval run's top-level output directory; everything except the headline
            metrics is read from ``out_dir/analysis/``.
        metrics_path: this run's ``metrics.json``. Since 2026-07-31 it lives in the profile
            folder (``<out_dir>/fid_<profile>/metrics.json``) because ``docker`` and
            ``research`` write the same keys on incomparable scales. Left as None, the old
            top-level location is used, so pre-2026-07-31 runs still render.

    Returns:
        Path to the written ``SUMMARY.md``.
    """
    out_dir = Path(out_dir)
    analysis_dir = out_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    lines = [f"# Eval summary — `{out_dir.name}`", ""]
    lines += _headline_section(
        Path(metrics_path) if metrics_path else out_dir / "metrics.json"
    )
    lines += _setlevel_section(analysis_dir)
    lines += _subgroup_section(
        "Per-abnormality (18 labels)",
        analysis_dir / "subgroup" / "per_abnormality.csv",
        "label",
    )
    lines += _subgroup_section(
        "Label burden bands", analysis_dir / "subgroup" / "label_burden.csv", "band"
    )
    lines += _subgroup_section(
        "Organ clusters", analysis_dir / "subgroup" / "organ_cluster.csv", "cluster"
    )
    lines += _mask_section(analysis_dir / "per_sample.csv")
    lines += _figures_section(analysis_dir / "figures")

    summary_path = analysis_dir / "SUMMARY.md"
    summary_path.write_text("\n".join(lines))
    return summary_path


__all__ = ["write_summary"]
