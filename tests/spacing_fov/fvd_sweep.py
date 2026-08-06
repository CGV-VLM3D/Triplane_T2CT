#!/usr/bin/env python3
"""Does FVD_CTCLIP respond to declared spacing, the way 2.5D-FID does?

The FID/CLIP spacing sweeps (`sweep_fid__*.json`, `zsweep_clip__*.json`) never computed FVD —
both scripts pass ``"fvd_ctclip": False``. The only FVD-vs-spacing evidence so far is the n=100
2×2 in `results/decompose.json`, where FVD stayed at 0.284–0.294 across three of four cells while
FID moved 1.64↔7.35. This sweeps the same declared-spacing grid the CLIP sweep used, FVD only.

Same relabel trick as `z_sweep_clip.py`: only the header changes, the voxels never do. So this
isolates the *declared*-spacing axis; the conditioning axis would need regeneration.

Each cell is deleted after scoring — a 300-volume cell is ~35 GB and the grid would otherwise
need ~300 GB of scratch.

Usage:
    CUDA_VISIBLE_DEVICES=0 python tests/spacing_fov/fvd_sweep.py \
        --tag wan_ep299_300 \
        --src-dir outputs/report2ct_wan/eval_ep299_n300_sp0.73_1.34_cfg5/predictions \
        --config-pairs 0.68,1.34 0.715,1.34 0.75,1.34 0.78,1.34 0.715,1.5 0.75,1.3
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys

sys.path.insert(0, "/workspace")

from src.eval.tasks.ctgen import CTGenEvaluator  # noqa: E402
from tests.spacing_fov.z_sweep_clip import _materialise  # noqa: E402

HERE = pathlib.Path(__file__).parent
RESULTS = HERE / "results"
CELLS = HERE / "_fvdcells"

GT_3001 = "/workspace/data/vlm3d_eval/_valid_full_3001"
# FVD pairs each generated volume to its own GT by stem and ignores extra GT, so the canonical
# 3001 set works for any n. No prompts needed — FVD never touches the text side.
METRICS = {"fvd": False, "fvd_ctclip": True, "clip_score": False, "fid_2p5d": False}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="names the output json")
    ap.add_argument("--src-dir", required=True, help="predictions dir to re-declare")
    ap.add_argument("--gt-dir", default=GT_3001)
    ap.add_argument(
        "--config-pairs",
        nargs="+",
        required=True,
        metavar="SX,SZ",
        help="declared 'inplane,z' spacings",
    )
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    # .resolve() matters: the evaluator shells out with a different cwd (see z_sweep_clip.py).
    src_dir = pathlib.Path(args.src_dir).resolve()
    gt_dir = pathlib.Path(args.gt_dir).resolve()
    configs = [tuple(float(x) for x in p.split(",")) for p in args.config_pairs]

    out_path = RESULTS / f"fvd_sweep__{args.tag}.json"
    out = json.loads(out_path.read_text()) if out_path.is_file() else {}

    for sx, sz in configs:
        key = f"{sx}_{sz}"
        if out.get(key, {}).get("FVD_CTCLIP") is not None:
            print(
                f"[fvd-sweep] {key} already computed ({out[key]['FVD_CTCLIP']:.4f}) — skip"
            )
            continue
        cell_dir = CELLS / f"{args.tag}__{key}"
        _materialise(args.tag, sx, sz, cell_dir, src_dir=src_dir)
        print(f"\n===== {args.tag} declared ({sx}, {sx}, {sz}) =====", flush=True)
        ev = CTGenEvaluator(gt_dir=gt_dir, metrics=METRICS, prompt_xlsx=None)
        out[key] = ev.evaluate(cell_dir, HERE / "runs" / f"fvd_{args.tag}_{key}")
        out_path.write_text(json.dumps(out, indent=2))
        shutil.rmtree(cell_dir, ignore_errors=True)  # ~35 GB per cell

    print(f"\n{'declared':18s}{'FVD_CTCLIP':>12s}")
    for sx, sz in configs:
        v = out[f"{sx}_{sz}"].get("FVD_CTCLIP", float("nan"))
        print(f"({sx},{sx},{sz})".ljust(18) + f"{v:12.4f}")
    print(f"\n[fvd-sweep] → {out_path}")


if __name__ == "__main__":
    main()
