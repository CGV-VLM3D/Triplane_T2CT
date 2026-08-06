#!/usr/bin/env python3
"""Score FID_2p5D + FVD_CTCLIP per condition on an existing mask-intervention run — no
regeneration (reuses ``predictions/`` and ``gt_view/`` already on disk).

``run_eval.py`` refuses the pooled FID/FVD for a manifest run because pooling every condition
together repeats each target up to 3x. This script scores each condition's subset SEPARATELY,
which has no such problem (docs/mask_intervention_manifest.md;
src/eval/analysis/condition_setlevel.py), so it is the follow-up step for a run that was scored
with ``task.metrics.fid_2p5d=false task.metrics.fvd_ctclip=false`` (the manifest-run default).

  CUDA_VISIBLE_DEVICES=3 python scripts/score_condition_fid.py \\
      --run-dir /workspace/outputs/report2ct_wan_mask_v2/eval_intervention_n300_sp0.75_1.3_cfgt5_cfgm1.0

Writes ``<run-dir>/condition_fid/condition_fid_fvd.csv`` (+ each condition's own
``fid_<profile>/`` folder under ``condition_fid/<condition>/``) and refreshes ``SUMMARY.md``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.eval.analysis import condition_setlevel  # noqa: E402
from src.eval.analysis.summary import write_summary  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-dir",
        required=True,
        help="the run's out_dir (has predictions/, gt_view/, analysis/per_sample.csv)",
    )
    ap.add_argument(
        "--fid-profile",
        default="docker_n300",
        choices=["docker", "docker_n300", "research"],
    )
    ap.add_argument("--ctclip-ckpt", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    per_sample_csv = run_dir / "analysis" / "per_sample.csv"
    gt_view_dir = run_dir / "gt_view"
    if not per_sample_csv.is_file():
        raise SystemExit(
            f"{per_sample_csv} not found — run scoring with task.metrics.per_sample=true first."
        )
    if not gt_view_dir.is_dir():
        raise SystemExit(
            f"{gt_view_dir} not found — this does not look like a manifest run's out_dir."
        )

    df = condition_setlevel.run(
        pred_dir=run_dir / "predictions",
        gt_view_dir=gt_view_dir,
        per_sample_csv=per_sample_csv,
        out_dir=run_dir / "condition_fid",
        fid_profile=args.fid_profile,
        ctclip_ckpt=args.ctclip_ckpt,
    )
    print(df.to_string(index=False))

    # Point SUMMARY.md's headline at whichever fid_<profile>/metrics.json this run already has
    # (unaffected by this script — the pooled top-level metrics stay whatever they were, usually
    # absent for a manifest run).
    metrics_path = next(run_dir.glob("fid_*/metrics.json"), None)
    write_summary(run_dir, metrics_path=metrics_path)
    print(f"\nSUMMARY.md refreshed: {run_dir / 'analysis' / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
