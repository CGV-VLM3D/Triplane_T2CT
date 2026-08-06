"""Score an existing ctgen predictions dir (e.g. Wan decode output) — no regeneration.

Reuses the production ``CTGenEvaluator``, so the numbers are produced by exactly the same code
path as ``run_eval.py``. The FID profile selects the metric family:

  docker   (default) squeezenet1_1 over 100 volumes — the official container's setting
  research           radimagenet_resnet50 over every prediction vs the 3001-GT (mu, Sigma)

  # docker-profile scoring of an existing run -> lands in <dir>/fid_docker/
  CUDA_VISIBLE_DEVICES=1 python scripts/rescore_predictions.py --metrics fid_2p5d \
      --pred-dir <dir>/predictions --out <dir>

  # reproduce a previously recorded (research) number -> lands in <dir>/fid_research/
  CUDA_VISIBLE_DEVICES=1 python scripts/rescore_predictions.py --metrics fid_2p5d \
      --fid-profile research --pred-dir <dir>/predictions --out <dir>

**Results always land in a profile-named subdirectory of ``--out``** (``fid_docker`` /
``fid_research``), so two profiles' results coexist instead of overwriting each other. Passing
``--out <dir>/fid_docker`` is accepted too and is not nested twice. Since 2026-07-31
``run_eval.py`` writes to the same place, so both scorers share one layout and there is no
top-level ``metrics.json`` for either of them to clobber (docs/ctgen_local_eval.md).

One invocation scores ONE profile — deliberately, since this is the "add/redo a metric on an
existing run" tool. (``run_eval.py`` is the one that defaults to scoring both families, because
there the alternative is regenerating the predictions.) It does not update the run's combined
top-level ``summary.json``; that file is written by the run that produced the predictions.

Within a profile folder ``metrics.json`` is **merged**, so a partial re-score (say FVD only)
adds to what is already recorded rather than replacing it, and ``_history`` keeps a dated row
per pass. That is why FVD-only results no longer need their own ``fvd/`` subdirectory.

When to use this instead of ``run_eval.py``: to add or recompute a metric on predictions that
already exist. ``run_eval.py`` is the full run (generate → score → analysis → QC figures →
``.hydra/`` provenance) and is what the epoch sweep calls.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.eval.ct_rate_cases import load_eval_cases, write_prompt_xlsx
from src.eval.tasks.ctgen import (
    _DEFAULT_FID_PROFILE,
    _FID_PROFILES,
    CTGenEvaluator,
)

GT_3001 = "/workspace/data/vlm3d_eval/_valid_full_3001"
CTCLIP = "/workspace/data/vlm3d_eval/models/CT-CLIP_v2.pt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument(
        "--out",
        required=True,
        help="the run's eval dir; results go to <out>/fid_<profile>/ so the original "
        "metrics.json is never overwritten",
    )
    ap.add_argument(
        "--n",
        type=int,
        help="head cases (same as generation) — only needed to build the CLIP prompt xlsx",
    )
    ap.add_argument("--gt-dir", default=GT_3001)
    ap.add_argument(
        "--metrics",
        nargs="+",
        default=["clip_score", "fid_2p5d"],
        choices=["fvd", "fvd_ctclip", "clip_score", "fid_2p5d"],
        help="which metrics to run (default: the original clip+fid pair)",
    )
    ap.add_argument(
        "--fid-profile",
        default=_DEFAULT_FID_PROFILE,
        choices=sorted(_FID_PROFILES),
        help="FID metric family (affects FID only; CLIP/FVD are identical either way)",
    )
    args = ap.parse_args()

    # Always write into a profile-named subdirectory. Re-scoring writes metrics.json/fid.json/
    # summary.json, so pointing --out straight at an existing eval dir would replace that run's
    # recorded numbers with a different metric family — which is exactly how the wan_mask_v2
    # research FID was lost on 2026-07-29. Idempotent when --out already names the subdir.
    subdir = f"fid_{args.fid_profile}"
    out = Path(args.out)
    if out.name != subdir:
        out = out / subdir
    out.mkdir(parents=True, exist_ok=True)
    print(f"results -> {out}  (profile={args.fid_profile})")

    # The prompt xlsx is CLIP-only; skip it (and the CSV join it needs) for a FID-only re-score.
    prompt_xlsx = None
    if "clip_score" in args.metrics:
        if args.n is None:
            ap.error("--n is required when clip_score is among --metrics")
        cases = load_eval_cases(n_samples=args.n)  # same seeded head as generation
        prompt_xlsx = out / "prompts.xlsx"
        write_prompt_xlsx(cases, prompt_xlsx)

    ev = CTGenEvaluator(
        gt_dir=args.gt_dir,
        metrics={
            m: m in args.metrics
            for m in ("fvd", "fvd_ctclip", "clip_score", "fid_2p5d")
        },
        ctclip_ckpt=CTCLIP,
        prompt_xlsx=prompt_xlsx,
        fid_profile=args.fid_profile,
    )
    res = ev.evaluate(Path(args.pred_dir), out)
    print("\n=== metrics ===")
    for k, v in res.items():
        print(f"  {k:<22s} {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
