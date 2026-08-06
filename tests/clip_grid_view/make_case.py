#!/usr/bin/env python3
"""Batch-build a viewer comparison set for one (or more) case IDs.

Given just a case id (e.g. ``valid_1000_a_1``) this resolves the GT and every
model's prediction for that case, resamples them all onto the CLIPScore eval
grid, and writes them into ``clip_grid_view/<case>/`` so they overlap in a
medical viewer.

    python make_case.py valid_1000_a_1
    python make_case.py valid_1000_a_1 valid_500_a_1        # several at once
    python make_case.py valid_1_a_1 --model text2ct=/path/to/eval_dir   # add a model

GT is flipped in-plane (LPS→RAS) so it aligns with the RAS-decoded predictions;
predictions are written as-is. Edit MODELS below to change the default line-up
(tag → eval dir that contains a ``predictions/`` folder).
"""

from __future__ import annotations

import argparse
import pathlib

from resample_to_clip_grid import resample_and_save

# ── config ──────────────────────────────────────────────────────────────────
GT_ROOT = pathlib.Path("/workspace/data/vlm3d_eval/_valid_full_3001")
OUT_ROOT = pathlib.Path(__file__).parent

# tag → eval dir (each holds predictions/<case>.mha)
MODELS: dict[str, pathlib.Path] = {
    "report2ct_toy_v2_cfg5_sp0.8": pathlib.Path(
        "/workspace/outputs/report2ct/eval_cfg5_spacing0.8_toy_v2"
    ),
    "report2ct_full_sp0.8": pathlib.Path(
        "/workspace/outputs/report2ct/eval_report2ct_spacing0.8_full"
    ),
    "wan": pathlib.Path("/workspace/outputs/report2ct_wan/eval_ep299_sp0.73_1.34_cfg5"),
    "wan_mask": pathlib.Path(
        "/workspace/outputs/report2ct_wan_mask/eval_ep299_sp0.73_1.34_cfg5"
    ),
}


def build_case(case_id: str, models: dict[str, pathlib.Path]) -> None:
    """Resample GT + every model's prediction for `case_id` onto the CLIP grid."""
    out_dir = OUT_ROOT / case_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # GT (LPS on disk → flip to RAS so it aligns with predictions)
    gt = GT_ROOT / f"{case_id}.mha"
    if gt.is_file():
        resample_and_save(gt, out_dir / f"{case_id}__gt_clipgrid.mha", flip_xy=True)
    else:
        print(f"  [skip] GT not found: {gt}")

    # predictions (already RAS)
    for tag, eval_dir in models.items():
        pred = eval_dir / "predictions" / f"{case_id}.mha"
        if pred.is_file():
            resample_and_save(
                pred, out_dir / f"{case_id}__{tag}_clipgrid.mha", flip_xy=False
            )
        else:
            print(f"  [skip] {tag}: no prediction at {pred}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build a CLIP-grid viewer set for given case id(s)."
    )
    ap.add_argument("cases", nargs="+", help="case id(s), e.g. valid_1000_a_1")
    ap.add_argument(
        "--model",
        action="append",
        default=[],
        help="extra model as tag=/path/to/eval_dir (repeatable)",
    )
    args = ap.parse_args()

    models = dict(MODELS)
    for spec in args.model:
        tag, _, path = spec.partition("=")
        models[tag] = pathlib.Path(path)

    for case_id in args.cases:
        print(f"=== {case_id} ===")
        build_case(case_id, models)


if __name__ == "__main__":
    main()
