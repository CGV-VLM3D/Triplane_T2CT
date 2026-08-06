"""Write ``analysis/SUMMARY.md`` — a one-glance rollup of everything this extension produces.

Pulls from whatever artifacts exist (headline metrics.json, per_sample.csv, subgroup/*.csv,
setlevel/*.csv, figures/) — sections for missing artifacts are simply omitted, so this can be
called after any subset of the new flags ran.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.eval.analysis.seg_metrics import SURFACE_METRICS


def _fmt(value, digits=4, ints: bool = False) -> str:
    """Table cell: fixed-point float, or a plain integer when ``ints`` (sample counts)."""
    if value is None:
        return "n/a"
    try:
        if pd.isna(value):
            return "NaN"
        return str(int(value)) if ints else f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


# FID lives on a different scale per profile, so it is tabulated per profile rather than
# listed once; everything else (CLIP / FVD) is profile-independent and listed once.
_FID_COLS = ("FID_2p5D_XY", "FID_2p5D_YZ", "FID_2p5D_XZ", "FID_2p5D_Avg")
_FID_META = ("fid_num_images", "fid_profile", "fid_model_name")
# Display order; any profile not listed here is appended alphabetically.
_PROFILE_ORDER = ("docker", "docker_n300", "research")


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _collect_metrics(out_dir: Path, metrics_json: Path | None) -> tuple[dict, dict]:
    """Gather every profile's metrics.json under ``out_dir``.

    Returns:
        ``(by_profile, shared)`` — ``by_profile`` maps profile name to its metrics dict;
        ``shared`` holds the profile-independent metrics (CLIP / FVD), taken from
        ``metrics_json`` when it carries any, else from the first source that does.
    """
    by_profile: dict[str, dict] = {}
    # Pre-2026-07-31 runs kept one metrics.json at the top level with no `fid_profile` key;
    # by convention that file is the `research` family (docs/ctgen_local_eval.md). Read it
    # first so a real fid_<profile>/ folder overwrites it on collision.
    legacy = out_dir / "metrics.json"
    if legacy.is_file():
        metrics = _read_json(legacy)
        by_profile[metrics.get("fid_profile") or "research"] = metrics
    for path in sorted(out_dir.glob("fid_*/metrics.json")):
        metrics = _read_json(path)
        # dir name is the authority — a merged file could carry a stale `fid_profile`.
        by_profile[path.parent.name[len("fid_") :]] = metrics

    def _shared_of(metrics: dict) -> dict:
        # `_history` (and any future `_`-prefixed bookkeeping) records WHEN each metric was
        # scored, not a metric — rendering it here would dump a nested list into the summary.
        return {
            k: v
            for k, v in metrics.items()
            if not k.startswith("_") and k not in _FID_COLS and k not in _FID_META
        }

    preferred = _shared_of(_read_json(metrics_json)) if metrics_json else {}
    if preferred:
        return by_profile, preferred
    for metrics in by_profile.values():
        shared = _shared_of(metrics)
        if shared:
            return by_profile, shared
    return by_profile, {}


def _headline_section(out_dir: Path, metrics_json: Path | None = None) -> list[str]:
    by_profile, shared = _collect_metrics(out_dir, metrics_json)
    if not by_profile:
        return []
    lines = ["## Headline VLM3D metrics", ""]
    for key, value in shared.items():
        lines.append(f"- **{key}**: {_fmt(value)}")
    lines.append("")

    ordered = [p for p in _PROFILE_ORDER if p in by_profile]
    ordered += sorted(p for p in by_profile if p not in _PROFILE_ORDER)
    rows = [
        (p, by_profile[p])
        for p in ordered
        if any(c in by_profile[p] for c in _FID_COLS)
    ]
    if rows:
        lines += [
            "### 2.5D-FID by profile",
            "",
            "| profile | n | "
            + " | ".join(c.replace("FID_2p5D_", "") for c in _FID_COLS)
            + " |",
            "|---|---|" + "---|" * len(_FID_COLS),
        ]
        for profile, metrics in rows:
            cells = " | ".join(_fmt(metrics.get(c)) for c in _FID_COLS)
            lines.append(
                f"| {profile} | {_fmt(metrics.get('fid_num_images'), ints=True)} | {cells} |"
            )
        lines += [
            "",
            "⚠ Profiles are **not** comparable to each other (different feature network and/or "
            "volume count); compare a model against another model within one profile only.",
            "",
        ]
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
            f"\n(+{n_other} more rows — per-label/cluster/burden — full table in the appendix.)"
        )
    lines.append("")
    return lines


# Preferred column order for the set-level appendix; whatever is absent is skipped, so a run
# scored without FVD (the common case — it is opt-in) simply drops that column.
_SETLEVEL_APPENDIX_COLS = (
    "axis",
    "FID_2p5D_Avg",
    "FID_2p5D_XY",
    "FID_2p5D_YZ",
    "FID_2p5D_XZ",
    "FVD_CTCLIP",
    "real_n",
    "gen_n",
    "real_patients",
    "below_threshold",
)


def _setlevel_appendix(analysis_dir: Path) -> list[str]:
    """Every set-level axis, in CSV order (overall/normal/disease, then per-label, burden,
    cluster).

    The section near the top stays a 3-row headline; this is the full breakdown, kept at the end
    because it is a lookup table rather than something to read top-to-bottom (user request,
    2026-08-02). Row order is the CSV's, which groups the axis families — not sorted by FID,
    since that would scatter related axes.
    """
    path = analysis_dir / "setlevel" / "setlevel_fid_fvd.csv"
    if not path.is_file():
        return []
    df = pd.read_csv(path)
    cols = [c for c in _SETLEVEL_APPENDIX_COLS if c in df.columns]
    lines = [
        f"## Appendix — set-level FID/FVD, all {len(df)} axes "
        f"(`setlevel/setlevel_fid_fvd.csv`)",
        "",
        "`below_threshold` marks a small generated set (a sample-size flag, not a validity "
        "claim). ⚠ On a mask-intervention run these numbers are diagnostic only — one target "
        "contributes several volumes.",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "---|" * len(cols),
    ]
    counts = ("real_n", "gen_n", "real_patients")
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            if c in ("axis", "below_threshold"):
                cells.append(str(row[c]))
            elif c in counts:
                cells.append(str(int(row[c])) if pd.notna(row[c]) else "-")
            else:
                cells.append(_fmt(row[c]))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


# Every quality metric a subgroup/condition row can carry, in reading order: text alignment, then
# each surface/overlap family against both references. Derived from SURFACE_METRICS (not
# hand-listed) so a new family shows up here automatically — the same drift guard subgroup.py's
# METRIC_COLS uses. Only the MEAN is rendered; std/median/n_valid stay in the CSV, which is the
# one thing SUMMARY.md deliberately does not reproduce (5 stats x 20 metrics = 100+ columns).
_SUMMARY_METRICS: tuple[str, ...] = ("clip_t2i", "clip_i2i") + tuple(
    f"{family}_to_{ref}"
    for family in ("dice", *SURFACE_METRICS)
    for ref in ("input_mask", "gt_mask")
)


def _metric_header(col: str) -> str:
    """``dice_to_input_mask_mean`` -> ``dice_to_input_mask`` (the ``_mean`` is implied by the
    section note, and dropping it keeps a 12-column table narrow enough to read)."""
    return col[: -len("_mean")] if col.endswith("_mean") else col


def _subgroup_section(
    title: str, csv_path: Path, key_col: str, top_n: int | None = None
) -> list[str]:
    """One subgroup table, rows ordered by n (descending).

    Args:
        top_n: how many rows to render; ``None`` (the default since 2026-08-02) renders ALL of
            them. Every table used to be cut at 6 rows, which hid two thirds of the 18 labels —
            and the CSV these sections summarise is too wide to read directly, so anything cut
            here was effectively lost (user request).
    """
    if not csv_path.is_file():
        return []
    df = pd.read_csv(csv_path)
    lines = [
        f"## {title} (`{csv_path.name}`)",
        "",
        "Values are means; std / median / n_valid per metric are in the CSV.",
        "",
    ]
    cols = [key_col, "n"] + [f"{m}_mean" for m in _SUMMARY_METRICS]
    cols = [c for c in cols if c in df.columns]
    shown = df.sort_values("n", ascending=False) if "n" in df.columns else df
    if top_n is not None:
        shown = shown.head(top_n)
    lines.append("| " + " | ".join(_metric_header(c) for c in cols) + " |")
    lines.append("|" + "---|" * len(cols))
    for _, row in shown.iterrows():
        lines.append(
            "| "
            + " | ".join(
                str(row[c]) if c == key_col else _fmt(row[c], ints=(c == "n"))
                for c in cols
            )
            + " |"
        )
    if top_n is not None and len(df) > top_n:
        lines.append(f"\n(showing {top_n}/{len(df)} rows by n — see the CSV for all.)")
    lines.append("")
    return lines


# Reading order for the intervention arms; anything unexpected is appended after these.
_CONDITION_ORDER = ("gt", "label_matched_swap", "label_mismatched_swap", "null", "none")


def _condition_section(per_sample_csv: Path) -> list[str]:
    """Mask-intervention headline: one row per conditioning-mask condition.

    This is the table the experiment exists to produce — same target, same report, same initial
    noise, different input mask — so it sits above the label/burden/cluster breakdowns. It renders
    only when a run actually has more than one condition (a plain run has just ``gt`` or ``none``,
    where the row would repeat the overall means).

    ``n`` differs across conditions on purpose: ``label_matched_swap`` only covers targets that
    have an exact label twin (docs/mask_intervention_manifest.md), so comparisons involving it
    are over a subset of the targets the other arms cover.
    """
    if not per_sample_csv.is_file():
        return []
    df = pd.read_csv(per_sample_csv)
    if "condition" not in df.columns or df["condition"].dropna().nunique() <= 1:
        return []

    present = [c for c in _CONDITION_ORDER if c in set(df["condition"])]
    present += sorted(set(df["condition"].dropna()) - set(present))
    metrics = [m for m in _SUMMARY_METRICS if m in df.columns]

    lines = [
        "## Mask intervention by condition (`per_sample.csv`)",
        "",
        f"{df['sample_id'].nunique()} generated volumes over {df['target_id'].nunique()} "
        "target scans. Values are means over each condition's samples; `dice_to_input_mask` is "
        "NA for `null` (no designated input mask).",
        "",
        "| condition | n | " + " | ".join(metrics) + " |",
        "|" + "---|" * (len(metrics) + 2),
    ]
    for condition in present:
        sub = df[df["condition"] == condition]
        cells = [_fmt(pd.to_numeric(sub[m], errors="coerce").mean()) for m in metrics]
        lines.append(f"| {condition} | {len(sub)} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _condition_fid_section(out_dir: Path) -> list[str]:
    """FID/FVD by condition (``condition_fid/condition_fid_fvd.csv``, opt-in).

    A run's default scoring REFUSES pooled FID/FVD for a manifest run (one target repeated
    across conditions breaks "1 volume = 1 independent sample" — ``run_eval._refuse_setlevel_
    metrics``), so this table only exists after ``scripts/score_condition_fid.py`` has scored
    each condition's SUBSET separately, which has no such problem. Renders nothing until then.
    """
    path = out_dir / "condition_fid" / "condition_fid_fvd.csv"
    if not path.is_file():
        return []
    df = pd.read_csv(path)
    if df.empty:
        return []
    cols = ["condition", "n", "FID_2p5D_Avg", "FVD_CTCLIP"]
    cols = [c for c in cols if c in df.columns]
    lines = [
        "## FID/FVD by condition (`condition_fid/condition_fid_fvd.csv`)",
        "",
        "Each condition scored as its OWN distribution (no pooling across conditions — see "
        "module docstring in `condition_setlevel.py`), so these ARE valid FID/FVD, unlike the "
        "pooled headline metric this run's `metrics.json` refuses to report.",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "---|" * len(cols),
    ]
    for _, row in df.iterrows():
        lines.append(
            "| "
            + " | ".join(
                str(row[c]) if c == "condition" else _fmt(row[c], ints=(c == "n"))
                for c in cols
            )
            + " |"
        )
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
    lines = ["## Mask-following Dice + HD95 (input mask vs target GT mask)", ""]
    lines.append(
        "| organ | dice_to_input_mask | dice_to_gt_mask "
        "| hd95_mm_to_input_mask | hd95_mm_to_gt_mask |"
    )
    lines.append("|---|---|---|---|---|")
    for organ in organs:
        col_in = f"dice_to_input_mask_{organ}"
        col_gt = f"dice_to_gt_mask_{organ}"
        if col_in not in df.columns:
            continue
        hd_in = f"hd95_mm_to_input_mask_{organ}"
        hd_gt = f"hd95_mm_to_gt_mask_{organ}"
        lines.append(
            f"| {organ} | {_fmt(df[col_in].mean())} | {_fmt(df[col_gt].mean())} "
            f"| {_fmt(df[hd_in].mean()) if hd_in in df.columns else 'n/a'} "
            f"| {_fmt(df[hd_gt].mean()) if hd_gt in df.columns else 'n/a'} |"
        )
    lines.append("")
    return lines


def _organ_generation_section(per_sample_csv: Path) -> list[str]:
    """Reference-independent: how often the model produces each organ at all — always available
    (mask AND non-mask models), unlike ``_mask_section`` which only renders for mask-conditioned
    runs. This is the rate the surface-distance sentinel (``seg_metrics.
    _surface_metrics_per_organ``) does not itself expose: a penalized organ still counts toward
    the surface mean, but doesn't say HOW OFTEN it was penalized.
    """
    if not per_sample_csv.is_file():
        return []
    df = pd.read_csv(per_sample_csv)
    organs = ["lung", "heart", "aorta", "esophagus"]
    cols = {o: f"organ_generated_{o}" for o in organs}
    if not any(c in df.columns for c in cols.values()):
        return []
    lines = [
        "## Organ generation rate (fraction of samples where the model produced that organ)",
        "",
        "| organ | generation_rate |",
        "|---|---|",
    ]
    for organ, col in cols.items():
        if col not in df.columns:
            continue
        lines.append(f"| {organ} | {_fmt(df[col].mean())} |")
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
        metrics_path: preferred source for the profile-independent metrics (CLIP / FVD) —
            normally this run's ``<out_dir>/fid_<profile>/metrics.json``. **Optional**: the
            headline section discovers every ``fid_*/metrics.json`` under ``out_dir`` (plus a
            pre-2026-07-31 top-level ``metrics.json``, which is the ``research`` family) and
            tabulates their FIDs per profile regardless. Callers that omit it — e.g.
            ``scripts/score_subgroups.py``, which re-scores an existing run — still get a full
            headline instead of none.

    Returns:
        Path to the written ``SUMMARY.md``.
    """
    out_dir = Path(out_dir)
    analysis_dir = out_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    lines = [f"# Eval summary — `{out_dir.name}`", ""]
    lines += _headline_section(out_dir, Path(metrics_path) if metrics_path else None)
    lines += _setlevel_section(analysis_dir)
    lines += _condition_section(analysis_dir / "per_sample.csv")
    lines += _condition_fid_section(out_dir)
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
    lines += _organ_generation_section(analysis_dir / "per_sample.csv")
    lines += _figures_section(analysis_dir / "figures")
    lines += _setlevel_appendix(analysis_dir)

    summary_path = analysis_dir / "SUMMARY.md"
    summary_path.write_text("\n".join(lines))
    return summary_path


__all__ = ["write_summary"]
