"""One-time precompute of per-axis GT ref-stats for subgroup FID (Dice/HD95 unaffected).

Axis membership (per-label/burden/cluster/normal/disease) depends only on the CT-RATE label
CSV + subgroup config — never on which model is being evaluated — so this only needs to run
ONCE per (GT set, subgroup config, FID profile) triple, and every future
``task.metrics.subgroup_setlevel=true`` run against that same GT set + profile reuses it
(mirrors how the main "overall" 2.5D-FID ref-stats.npz is built once and shared across every
model eval).

Prerequisite: at least one eval run with ``task.metrics.fid_2p5d=true`` (under the SAME
``--fid-profile``) must have completed against this GT set, so its GT feature cache
(``_shared_gt_fidfeat/<key>/``) exists — this script only re-groups already-cached per-volume
features, it does not extract new ones. A ``docker``/``docker_n300`` GT cache only ever holds as
many volumes as some prior run's ``num_images`` scored (100 / 300) — never the full 3001 the
``research`` profile has — so per-label axes are far smaller and less reliable under those
profiles; this script does not judge that, it just re-groups whatever features exist.

  python scripts/precompute_subgroup_fid_refstats.py \\
      --gt-dir /workspace/data/vlm3d_eval/_valid_full_3001 \\
      --subgroup-config /workspace/configs/eval/subgroup/default.yaml \\
      --fid-profile research   # or docker | docker_n300
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.eval.analysis import subgroup_config as subgroup_config_mod
from src.eval.analysis.labels import load_label_csv
from src.eval.analysis.subgroup_refstats import (
    precompute_axis_refstats,
    subgroup_refstats_dir,
)
from src.eval.analysis.subgroup_setlevel import _build_feature_index, _build_gt_axes
from src.eval.tasks.ctgen import (
    _DEFAULT_SUBGROUP_FID_PROFILE,
    _FID_PROFILES,
    _shared_gt_feat_dir,
)

log = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt-dir", required=True, help="GT .mha directory (task.gt_dir)")
    ap.add_argument(
        "--subgroup-config", default="/workspace/configs/eval/subgroup/default.yaml"
    )
    ap.add_argument("--label-split", default="valid")
    ap.add_argument(
        "--fid-profile",
        default=_DEFAULT_SUBGROUP_FID_PROFILE,
        choices=sorted(_FID_PROFILES),
        help="which FID profile's cached GT features to re-group (default: docker_n300, "
        "changed from research on 2026-08-01 to match subgroup_setlevel's new default).",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)

    gt_dir = Path(args.gt_dir)
    fid_model = _FID_PROFILES[args.fid_profile]["model"]
    shared_gt_feat_dir = _shared_gt_feat_dir(gt_dir, fid_model)
    if not shared_gt_feat_dir.is_dir():
        raise FileNotFoundError(
            f"{shared_gt_feat_dir} not found — run an eval with task.metrics.fid_2p5d=true "
            f"task.fid_profile={args.fid_profile} against this GT set first (this script only "
            "re-groups already-cached features)."
        )

    cfg = subgroup_config_mod.load(args.subgroup_config)
    gt_label_map = load_label_csv(args.label_split)
    gt_index = _build_feature_index(shared_gt_feat_dir)
    log.info(
        "GT feature cache (%s profile): %d cached volumes at %s",
        args.fid_profile,
        len(gt_index),
        shared_gt_feat_dir,
    )

    axes = _build_gt_axes(gt_label_map, cfg, gt_index)
    cache_dir = subgroup_refstats_dir(gt_dir, model=fid_model)
    log.info("Writing per-axis ref-stats to %s", cache_dir)

    for axis_name, gt_ids in axes:
        precompute_axis_refstats(gt_index, cache_dir, axis_name, gt_ids)

    log.info("Done — %d axes processed.", len(axes))


if __name__ == "__main__":
    main()
