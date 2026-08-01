"""Shared dispatch for the per-sample/subgroup/QC-figure analysis extension.

Called from ``scripts/run_eval.py`` (after ``CTGenEvaluator.evaluate``) and
``scripts/score_subgroups.py`` (post-hoc, on an existing predictions dir). Defines the single
source of truth for the ``analysis/`` output layout (plan §3.13) so both entrypoints and the
test suite agree on paths.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.eval.analysis import (
    persample,
    qc_figures,
    subgroup,
    subgroup_setlevel,
    summary,
)
from src.eval.analysis import subgroup_config as subgroup_config_mod
from src.eval.analysis.labels import load_label_csv

log = logging.getLogger(__name__)


def analysis_paths(out_dir: str | Path) -> dict[str, Path]:
    """Single source of truth for the ``analysis/`` sub-layout under an eval run's ``out_dir``."""
    analysis_dir = Path(out_dir) / "analysis"
    return {
        "analysis_dir": analysis_dir,
        "per_sample_csv": analysis_dir / "per_sample.csv",
        "subgroup_dir": analysis_dir / "subgroup",
        "setlevel_dir": analysis_dir / "setlevel",
        "figures_dir": analysis_dir / "figures",
    }


def run_qc_figures(
    pred_dir: str | Path,
    gt_dir: str | Path,
    out_dir: str | Path,
    model_name: str,
    n: int = 5,
    label_split: str = "valid",
) -> list[Path]:
    """Independent trigger: GT-vs-model QC figures (no metric flags required — see plan §3.10)."""
    paths = analysis_paths(out_dir)
    label_map = load_label_csv(label_split)
    return qc_figures.run(
        pred_dir=pred_dir,
        gt_dir=gt_dir,
        label_map=label_map,
        out_dir=paths["figures_dir"],
        model_name=model_name,
        n=n,
    )


def run_metrics_analysis(
    scan_ids: list[str],
    pred_dir: str | Path,
    gt_dir: str | Path,
    out_dir: str | Path,
    subgroup_config_path: str | Path,
    per_sample: bool = False,
    dice: bool = False,
    hd95: bool = False,
    subgroup_flag: bool = False,
    subgroup_setlevel_flag: bool = False,
    is_mask_model: bool = False,
    manifest_path: str | Path | None = None,
    seg_backend: str = "totalseg",
    seg_max_n: int | None = None,
    clip_scores: dict[str, tuple[float, float]] | None = None,
    subgroup_setlevel_compute_fvd: bool = False,
    label_split: str = "valid",
    subgroup_fid_profile: str = "research",
    metrics_path: str | Path | None = None,
) -> None:
    """Dispatch the metric-analysis flags: build per_sample.csv (if any flag needs it), then
    subgroup / subgroup_setlevel / SUMMARY.md.

    Args mirror the ``task.metrics.*`` config flags 1:1 (see plan §3.3); ``subgroup_flag`` /
    ``subgroup_setlevel_flag`` are named with a suffix to avoid shadowing the ``subgroup`` module.
    ``metrics_path`` points SUMMARY.md's headline section at this run's metrics.json, which since
    2026-07-31 lives in the profile folder ``<out_dir>/fid_<profile>/`` rather than at ``out_dir``.
    """
    if not any([per_sample, dice, hd95, subgroup_flag, subgroup_setlevel_flag]):
        return

    paths = analysis_paths(out_dir)
    paths["analysis_dir"].mkdir(parents=True, exist_ok=True)
    cfg = subgroup_config_mod.load(subgroup_config_path)

    # dice/hd95/subgroup/subgroup_setlevel all structurally need per_sample.csv to exist, so it
    # is always (re)built whenever any flag is set — not just when `per_sample` itself is true.
    df = persample.build_per_sample(
        scan_ids=scan_ids,
        pred_dir=pred_dir,
        gt_dir=gt_dir,
        subgroup_cfg=cfg,
        label_split=label_split,
        clip_scores=clip_scores,
        is_mask_model=is_mask_model,
        manifest_path=manifest_path,
        seg_backend=seg_backend if (dice or hd95) else None,
        compute_hd95=hd95,
        seg_max_n=seg_max_n,
    )
    df.to_csv(paths["per_sample_csv"], index=False)
    log.info("per_sample.csv: %d rows -> %s", len(df), paths["per_sample_csv"])

    if subgroup_flag:
        subgroup.run(paths["per_sample_csv"], cfg, paths["subgroup_dir"])

    if subgroup_setlevel_flag:
        subgroup_setlevel.run(
            run_out_dir=out_dir,
            per_sample_csv=paths["per_sample_csv"],
            subgroup_cfg=cfg,
            analysis_out_dir=paths["setlevel_dir"],
            gt_dir=gt_dir,
            compute_fvd=subgroup_setlevel_compute_fvd,
            label_split=label_split,
            fid_profile=subgroup_fid_profile,
        )

    summary.write_summary(out_dir, metrics_path=metrics_path)


__all__ = ["analysis_paths", "run_qc_figures", "run_metrics_analysis"]
