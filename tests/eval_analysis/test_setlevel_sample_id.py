"""Set-level subgroup axes must key the PRED side on ``sample_id``, not ``target_id``.

A cached pred feature file is named after the generated volume's stem = ``sample_id``. In a plain
run ``sample_id == target_id``, so the old ``target_id`` keying worked by coincidence; in a
mask-intervention run every lookup missed, so every axis reported ``gen_n=0`` and every subgroup
FID came out NaN. These tests pin both halves: the manifest run now finds its predictions, and the
plain run's axes are byte-for-byte what they were.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.eval.analysis import subgroup_config
from src.eval.analysis.labels import LABEL_NAMES, load_label_csv
from src.eval.analysis.persample import build_per_sample
from src.eval.analysis.subgroup_setlevel import _build_axes, _build_feature_index, run
from tests.eval_analysis._helpers import write_mha

_CFG = subgroup_config.load("/workspace/configs/eval/subgroup/default.yaml")
_LABELS = load_label_csv("valid")
# Two real valid scans with different label vectors (one normal, one diseased) so the
# normal/disease and per-label axes are both exercised.
_NORMAL = next(sid for sid, vec in sorted(_LABELS.items()) if vec.sum() == 0)
_DISEASE = next(sid for sid, vec in sorted(_LABELS.items()) if vec.sum() >= 2)


def _per_sample_df(rows: list[dict]) -> pd.DataFrame:
    """Build a per_sample-shaped frame through the real builder (label columns included)."""
    manifest_rows = [
        {
            "sample_id": r["sample_id"],
            "target_id": r["target_id"],
            "condition": r.get("condition", "gt"),
            "cond_mask_source_id": r.get("cond_mask_source_id"),
            "seed": 0,
            "cfg_scale_text": 5.0,
            "cfg_scale_mask": 1.0,
            "run_id": "test",
            "source_labels": None,
            "label_overlap": None,
        }
        for r in rows
    ]
    return manifest_rows


@pytest.fixture
def plain_df(tmp_path) -> pd.DataFrame:
    """Plain run: one row per scan, sample_id == target_id."""
    manifest = tmp_path / "plain.jsonl"
    manifest.write_text(
        "".join(
            json.dumps(r) + "\n"
            for r in _per_sample_df(
                [{"sample_id": sid, "target_id": sid} for sid in (_NORMAL, _DISEASE)]
            )
        )
    )
    return build_per_sample(
        scan_ids=[],
        pred_dir=tmp_path,
        gt_dir=tmp_path,
        subgroup_cfg=_CFG,
        manifest_path=manifest,
        seg_backend=None,
    )


@pytest.fixture
def manifest_df(tmp_path) -> pd.DataFrame:
    """Mask-intervention run: 3 conditions for each of the same two targets."""
    rows = []
    for target in (_NORMAL, _DISEASE):
        for condition in ("gt", "label_mismatched_swap", "null"):
            rows.append(
                {
                    "sample_id": f"{target}__{condition}__sm-1.0__seed-0",
                    "target_id": target,
                    "condition": condition,
                    "cond_mask_source_id": target if condition == "gt" else None,
                }
            )
    manifest = tmp_path / "intervention.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in _per_sample_df(rows)))
    return build_per_sample(
        scan_ids=[],
        pred_dir=tmp_path,
        gt_dir=tmp_path,
        subgroup_cfg=_CFG,
        manifest_path=manifest,
        seg_backend=None,
    )


def _axes(df: pd.DataFrame) -> dict[str, set[str]]:
    gt_index = {sid: Path(f"/nonexistent/{sid}.pt") for sid in _LABELS}
    return {name: pred for name, _gt, pred in _build_axes(df, _LABELS, _CFG, gt_index)}


def test_plain_run_axes_are_unchanged(plain_df):
    """(11) With sample_id == target_id the fix is a no-op — axes still hold the scan ids."""
    axes = _axes(plain_df)
    assert axes["overall"] == {_NORMAL, _DISEASE}
    assert axes["normal"] == {_NORMAL}
    assert axes["disease"] == {_DISEASE}
    for name, idx in zip(LABEL_NAMES, range(len(LABEL_NAMES))):
        expected = {sid for sid in (_NORMAL, _DISEASE) if _LABELS[sid][idx] == 1}
        assert axes[f"label:{name}"] == expected


def test_manifest_run_axes_use_sample_ids(manifest_df):
    """(10) Every generated sample lands in its target's axes, keyed by the file's own stem."""
    axes = _axes(manifest_df)
    all_samples = set(manifest_df["sample_id"])
    assert len(all_samples) == 6
    assert axes["overall"] == all_samples
    # the pre-fix keying would have produced target ids, which no feature file is named after
    assert not (axes["overall"] & set(manifest_df["target_id"]))
    assert len(axes["normal"]) == 3 and len(axes["disease"]) == 3
    assert axes["normal"] | axes["disease"] == all_samples

    disease_samples = set(
        manifest_df.loc[manifest_df["target_id"] == _DISEASE, "sample_id"]
    )
    for idx, name in enumerate(LABEL_NAMES):
        if _LABELS[_DISEASE][idx] == 1:
            assert axes[f"label:{name}"] == disease_samples


def _write_fake_feature(path: Path, n_slices: int = 4, dim: int = 8) -> None:
    """Cached 2.5D feature triplet, the shape ``_fid_subset`` restacks: 3 x (n_slices, dim)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tuple(torch.randn(n_slices, dim) for _ in range(3)), path)


def test_manifest_run_gen_n_is_positive_end_to_end(manifest_df, tmp_path):
    """(10) Through ``run()``: every axis the targets belong to gets gen_n > 0, and the overall
    axis counts every generated sample — the failure mode this fix addresses was gen_n=0 and an
    all-NaN CSV."""
    run_dir = tmp_path / "run"
    features = run_dir / "fid_research" / "fid_features"
    for sid in manifest_df["sample_id"]:
        _write_fake_feature(features / "pred" / f"{sid}.pt")
    for sid in (_NORMAL, _DISEASE):
        _write_fake_feature(features / "gt" / f"{sid}.pt")

    csv_path = tmp_path / "per_sample.csv"
    manifest_df.to_csv(csv_path, index=False)
    result = run(
        run_out_dir=run_dir,
        per_sample_csv=csv_path,
        subgroup_cfg=_CFG,
        analysis_out_dir=tmp_path / "setlevel",
        fid_profile="research",
    )

    by_axis = result.set_index("axis")
    assert by_axis.loc["overall", "gen_n"] == len(manifest_df)
    assert by_axis.loc["normal", "gen_n"] + by_axis.loc["disease", "gen_n"] == len(
        manifest_df
    )
    assert np.isfinite(by_axis.loc["overall", "FID_2p5D_Avg"])
    # axes whose GT side is non-empty must also see predictions (the pre-fix bug made these 0)
    populated = by_axis[by_axis["real_n"] > 0]
    assert (populated["gen_n"] > 0).all(), populated[["real_n", "gen_n"]]


def test_clip_scores_join_on_sample_id(tmp_path):
    """clip_persample keys its scores by the generated file's stem; per_sample must join on the
    same key, or a manifest run's CLIP column would be silently empty."""
    manifest = tmp_path / "m.jsonl"
    rows = _per_sample_df(
        [
            {
                "sample_id": f"{_DISEASE}__gt__sm-1.0__seed-0",
                "target_id": _DISEASE,
                "condition": "gt",
                "cond_mask_source_id": _DISEASE,
            }
        ]
    )
    manifest.write_text(json.dumps(rows[0]) + "\n")
    sample_id = rows[0]["sample_id"]
    write_mha(tmp_path / f"{sample_id}.mha", np.zeros((2, 2, 2), np.int16))
    write_mha(tmp_path / f"{_DISEASE}.mha", np.zeros((2, 2, 2), np.int16))

    df = build_per_sample(
        scan_ids=[],
        pred_dir=tmp_path,
        gt_dir=tmp_path,
        subgroup_cfg=_CFG,
        manifest_path=manifest,
        clip_scores={
            sample_id: (61.5, 24.0)
        },  # keyed by stem, as clip_persample returns
        seg_backend=None,
    )
    assert df.iloc[0]["clip_t2i"] == 61.5
    assert df.iloc[0]["clip_i2i"] == 24.0
    assert "clip_missing" not in df.iloc[0]["failure_reason"]
