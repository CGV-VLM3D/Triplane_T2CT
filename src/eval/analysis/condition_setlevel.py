"""FID / FVD by mask-intervention condition (``gt`` / ``label_matched_swap`` /
``label_mismatched_swap``) — requirement raised after the fact: the intervention experiment
wants these compared per condition, the same way it already compares CLIP-T2I and Dice.

``run_eval.py`` REFUSES the pooled/default ``fid_2p5d``/``fvd``/``fvd_ctclip`` for a manifest
run (``_refuse_setlevel_metrics``), and that refusal stays correct: pooling every condition's
volumes into one distribution repeats each target up to 3x, breaking "1 volume = 1 independent
sample". But a SINGLE condition's subset does not have that problem — a manifest generates one
target once per condition, so within ``label_mismatched_swap`` alone (say) no target repeats.
This module scores each condition's subset on its own, via the same ``CTGenEvaluator`` code path
the headline metrics use, so the numbers are directly comparable to any other single-condition
run (e.g. the model's own plain-run FID).

GT pairing gotcha: the docker-family FID profiles and FVD_CTCLIP both look up GT by the
PREDICTION'S OWN FILENAME STEM (``gt_dir/<stem>.mha``) — for a manifest run that stem is
``sample_id``, not ``target_id``, so the run's ``gt_view/`` (built once at generation time by
``src.eval.manifest.build_gt_view``) must be passed as ``gt_dir`` here, not the raw census.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

# Below this, CTGenEvaluator's own FID loop already logs "may be unreliable" (ctgen.py); no
# point scoring a condition subset smaller than that threshold.
_MIN_N = 10


def run(
    pred_dir: str | Path,
    gt_view_dir: str | Path,
    per_sample_csv: str | Path,
    out_dir: str | Path,
    fid_profile: str = "docker_n300",
    ctclip_ckpt: str | Path | None = None,
) -> pd.DataFrame:
    """Score FID_2p5D + FVD_CTCLIP once per condition present in ``per_sample.csv``.

    Args:
        pred_dir: the run's ``predictions/`` dir (filenames = ``sample_id.mha``).
        gt_view_dir: the run's ``gt_view/`` dir — required for GT-by-stem pairing (see module
            docstring). Built automatically by a manifest ``run_eval.py`` run.
        per_sample_csv: ``analysis/per_sample.csv`` (needs ``sample_id`` + ``condition``).
        out_dir: per-condition ``CTGenEvaluator`` output lands in ``out_dir/<condition>/``
            (same layout as a top-level ``fid_<profile>/`` folder); the summary CSV goes to
            ``out_dir/condition_fid_fvd.csv``.
        fid_profile: which FID feature network/truncation to use (``_FID_PROFILES``). Defaults
            to ``docker_n300`` — the profile this project's other subgroup breakdowns use, so
            these numbers sit on the same scale.
        ctclip_ckpt: ``CT-CLIP_v2.pt`` path (default: ``CTGenEvaluator``'s own default).

    Returns:
        One row per scored condition: ``condition``, ``n``, ``FID_2p5D_{Avg,XY,YZ,XZ}``,
        ``FVD_CTCLIP``, ``fid_num_images``. A condition with fewer than :data:`_MIN_N` samples
        or no matching prediction files is skipped (logged), not silently dropped from the CSV
        as a blank row.
    """
    from src.eval.tasks.ctgen import CTGenEvaluator

    pred_dir = Path(pred_dir)
    gt_view_dir = Path(gt_view_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(per_sample_csv)
    if "condition" not in df.columns:
        raise ValueError(
            f"{per_sample_csv} has no `condition` column — this is not a manifest run."
        )

    rows: list[dict] = []
    for condition, sub in df.groupby("condition"):
        sample_ids = sub["sample_id"].astype(str).tolist()
        if len(sample_ids) < _MIN_N:
            log.warning(
                "condition=%r has only %d samples (<%d) — skipping FID/FVD as unreliable.",
                condition,
                len(sample_ids),
                _MIN_N,
            )
            continue

        subset_dir = out_dir / "_pred_subset" / str(condition)
        if subset_dir.exists():
            shutil.rmtree(subset_dir)
        subset_dir.mkdir(parents=True)
        linked = 0
        for sid in sample_ids:
            src = pred_dir / f"{sid}.mha"
            if src.is_file():
                (subset_dir / src.name).symlink_to(src.resolve())
                linked += 1
        if linked == 0:
            log.warning(
                "condition=%r: none of its %d sample_id(s) have a prediction file in %s.",
                condition,
                len(sample_ids),
                pred_dir,
            )
            shutil.rmtree(subset_dir, ignore_errors=True)
            continue

        score_dir = out_dir / str(condition)
        evaluator = CTGenEvaluator(
            gt_dir=gt_view_dir,
            metrics={
                "fid_2p5d": True,
                "fvd_ctclip": True,
                "fvd": False,
                "clip_score": False,
            },
            ctclip_ckpt=ctclip_ckpt,
            fid_profile=fid_profile,
        )
        results = evaluator.evaluate(subset_dir, score_dir)
        shutil.rmtree(subset_dir, ignore_errors=True)

        rows.append(
            {
                "condition": condition,
                "n": linked,
                "FID_2p5D_Avg": results.get("FID_2p5D_Avg"),
                "FID_2p5D_XY": results.get("FID_2p5D_XY"),
                "FID_2p5D_YZ": results.get("FID_2p5D_YZ"),
                "FID_2p5D_XZ": results.get("FID_2p5D_XZ"),
                "FVD_CTCLIP": results.get("FVD_CTCLIP"),
                "fid_num_images": results.get("fid_num_images"),
            }
        )

    shutil.rmtree(
        out_dir / "_pred_subset", ignore_errors=True
    )  # drop the now-empty parent too

    result_df = pd.DataFrame(rows)
    result_df.to_csv(out_dir / "condition_fid_fvd.csv", index=False)
    log.info("condition FID/FVD: %d condition(s) -> %s", len(result_df), out_dir)
    return result_df


__all__ = ["run"]
