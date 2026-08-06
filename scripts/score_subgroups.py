"""Run the per-sample/subgroup/QC analysis extension on an EXISTING predictions dir, without
regenerating anything — the post-hoc counterpart to ``run_eval.py``'s in-line triggers (mirrors
[rescore_predictions.py](rescore_predictions.py)'s pattern for the existing VLM3D metrics).

  CUDA_VISIBLE_DEVICES=0 python scripts/score_subgroups.py \\
      --pred-dir outputs/report2ct_wan_mask/eval_.../predictions \\
      --out outputs/report2ct_wan_mask/eval_.../ \\
      --n 300 --is-mask-model --dice --hd95 --per-sample --subgroup --subgroup-setlevel

  # subgroup_setlevel FID against a different profile's cached features (must match what
  # this run's task.fid_profile actually scored with; see --subgroup-fid-profile choices):
  CUDA_VISIBLE_DEVICES=0 python scripts/score_subgroups.py \\
      --pred-dir outputs/.../predictions --out outputs/... --n 300 \\
      --subgroup-setlevel --subgroup-fid-profile docker
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.eval.analysis import clip_persample
from src.eval.analysis.orchestrate import run_metrics_analysis, run_qc_figures
from src.eval.ct_rate_cases import load_eval_cases, write_prompt_xlsx
from src.eval.tasks.ctgen import _DEFAULT_SUBGROUP_FID_PROFILE, _FID_PROFILES

log = logging.getLogger(__name__)

GT_3001 = "/workspace/data/vlm3d_eval/_valid_full_3001"
CTCLIP = "/workspace/data/vlm3d_eval/models/CT-CLIP_v2.pt"
SUBGROUP_CONFIG = "/workspace/configs/eval/subgroup/default.yaml"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument(
        "--out", required=True, help="run's top-level out_dir (analysis/ goes here)"
    )
    ap.add_argument(
        "--n", type=int, required=True, help="head cases (same as generation)"
    )
    ap.add_argument("--gt-dir", default=GT_3001)
    ap.add_argument(
        "--model-name",
        default=None,
        help="for QC figure labeling (default: pred-dir parent name)",
    )

    ap.add_argument("--qc-figures", action="store_true")
    ap.add_argument("--qc-n", type=int, default=5)
    ap.add_argument("--per-sample", action="store_true")
    ap.add_argument("--dice", action="store_true")
    ap.add_argument("--hd95", action="store_true")
    ap.add_argument("--subgroup", action="store_true")
    ap.add_argument("--subgroup-setlevel", action="store_true")
    ap.add_argument("--subgroup-setlevel-fvd", action="store_true")
    ap.add_argument("--is-mask-model", action="store_true")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--seg-backend", default="totalseg")
    ap.add_argument("--seg-max-n", type=int, default=None)
    ap.add_argument("--subgroup-config", default=SUBGROUP_CONFIG)
    ap.add_argument(
        "--subgroup-fid-profile",
        default=_DEFAULT_SUBGROUP_FID_PROFILE,
        choices=sorted(_FID_PROFILES),
        help="which FID profile's cached features subgroup_setlevel reuses (default: "
        "docker_n300, changed from research on 2026-08-01 — the n=100 'docker' profile left "
        "most per-label axes with single-digit or zero GT matches). Must match a profile "
        "this run's task.fid_profile actually scored with, and that profile's 29-axis "
        "ref-stats must exist (scripts/precompute_subgroup_fid_refstats.py --fid-profile "
        "<name>) for the fast path — a missing cache still works, just slower.",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)

    out_dir = Path(args.out)
    pred_dir = Path(args.pred_dir)
    model_name = args.model_name or pred_dir.parent.parent.name

    cases = load_eval_cases(n_samples=args.n)
    scan_ids = [c.scan_id for c in cases]

    if args.qc_figures:
        run_qc_figures(
            pred_dir, args.gt_dir, out_dir, model_name=model_name, n=args.qc_n
        )

    if not any(
        [args.per_sample, args.dice, args.hd95, args.subgroup, args.subgroup_setlevel]
    ):
        return

    clip_scores = None
    if args.per_sample:
        prompt_xlsx = out_dir / "prompts.xlsx"
        write_prompt_xlsx(cases, prompt_xlsx)
        clip_scores, _ = clip_persample.run(
            pred_dir=pred_dir,
            gt_dir=args.gt_dir,
            prompt_xlsx=prompt_xlsx,
            out_dir=out_dir,
            ctclip_ckpt=CTCLIP,
        )

    run_metrics_analysis(
        scan_ids=scan_ids,
        pred_dir=pred_dir,
        gt_dir=args.gt_dir,
        out_dir=out_dir,
        subgroup_config_path=args.subgroup_config,
        per_sample=args.per_sample,
        dice=args.dice,
        hd95=args.hd95,
        subgroup_flag=args.subgroup,
        subgroup_setlevel_flag=args.subgroup_setlevel,
        is_mask_model=args.is_mask_model,
        manifest_path=args.manifest,
        seg_backend=args.seg_backend,
        seg_max_n=args.seg_max_n,
        clip_scores=clip_scores,
        subgroup_setlevel_compute_fvd=args.subgroup_setlevel_fvd,
        subgroup_fid_profile=args.subgroup_fid_profile,
    )
    log.info("Done — see %s/analysis/", out_dir)


if __name__ == "__main__":
    main()
