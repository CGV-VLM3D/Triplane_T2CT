"""Manifest consumption: condition=null -> no designated input mask -> dice_to_input_mask stays
NA (only dice_to_gt_mask is computed), while a designated condition (gt/swap) computes both.
Segmentation itself is monkeypatched (no GPU/TotalSegmentator needed) — this test is about the
ROUTING logic in seg_metrics.compute_seg_metrics /persample.build_per_sample, not segmentation
accuracy (already verified live against real data — see plan §7)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.analysis import seg_metrics
from src.eval.analysis import subgroup_config
from src.eval.analysis.persample import build_per_sample
from tests.eval_analysis._helpers import write_mha

_CFG = subgroup_config.load("/workspace/configs/eval/subgroup/default.yaml")


@pytest.fixture(autouse=True)
def _fake_segment_volume(monkeypatch):
    """Stub out TotalSegmentator with a deterministic fake label map (values 0/10/51/52/15,
    matching real TS-v2 ids for background/lung/heart/aorta/esophagus) so seg_metrics'
    apply_grouping + Dice logic runs for real without needing a GPU."""

    def _fake(mha_path, backend="totalseg"):
        arr = np.zeros((6, 6, 6), dtype=np.int64)
        arr[0:2, 0:2, 0:2] = 10  # lung
        return arr

    monkeypatch.setattr(seg_metrics, "segment_volume", _fake)


def test_condition_null_leaves_dice_to_input_na(tmp_path: Path, monkeypatch):
    """cond_mask_source_id=None (condition=null) -> dice_to_input_mask stays NaN; dice_to_gt_mask
    is still computed from the target's own ts_seg."""

    # stub the reference-mask loader too (avoids needing a real ts_seg file on disk)
    def _fake_ref_mask(scan_id, target_shape_xyz, split="valid_fixed"):
        arr = np.zeros((6, 6, 6), dtype=np.int64)
        arr[0:2, 0:2, 0:2] = 1  # lung group id
        return arr

    monkeypatch.setattr(seg_metrics, "load_reference_organ_mask", _fake_ref_mask)

    gen_path = write_mha(tmp_path / "gen.mha", np.zeros((6, 6, 6), dtype=np.int16))
    result_null = seg_metrics.compute_seg_metrics(
        gen_mha_path=gen_path, target_id="target1", cond_mask_source_id=None
    )
    assert np.isnan(result_null["dice_to_input_mask"])
    assert not np.isnan(result_null["dice_to_gt_mask"])
    assert "no_input_mask_designated" in result_null["failure_reason"]

    result_gt = seg_metrics.compute_seg_metrics(
        gen_mha_path=gen_path, target_id="target1", cond_mask_source_id="target1"
    )
    assert not np.isnan(result_gt["dice_to_input_mask"])
    assert (
        result_gt["dice_to_input_mask"] == result_gt["dice_to_gt_mask"]
    )  # same source scan


def test_organ_generated_is_reference_independent(tmp_path: Path, monkeypatch):
    """``organ_generated_<organ>`` reflects only the model's own segmentation (lung present,
    the other three absent, per the autouse fake), regardless of whether any reference mask is
    available at all — unlike dice/surface metrics, it needs no gt_mask/input_mask."""
    gen_path = write_mha(tmp_path / "gen.mha", np.zeros((6, 6, 6), dtype=np.int16))

    def _no_ref_mask(scan_id, target_shape_xyz, split="valid_fixed"):
        return None

    monkeypatch.setattr(seg_metrics, "load_reference_organ_mask", _no_ref_mask)

    result = seg_metrics.compute_seg_metrics(
        gen_mha_path=gen_path, target_id="target1", cond_mask_source_id=None
    )
    assert result["organ_generated_lung"] == 1
    assert result["organ_generated_heart"] == 0
    assert result["organ_generated_aorta"] == 0
    assert result["organ_generated_esophagus"] == 0
    # no reference was available at all, yet organ_generated_* is still populated
    assert np.isnan(result["dice_to_gt_mask"])


def test_manifest_condition_null_propagates_to_persample(tmp_path: Path):
    pred_dir = tmp_path / "pred"
    gt_dir = tmp_path / "gt"
    target_id = "valid_1_a_1"
    sample_id = f"{target_id}__null__src-none__sm-0.0__seed-0"
    write_mha(pred_dir / f"{sample_id}.mha", np.zeros((6, 6, 6), dtype=np.int16))
    write_mha(gt_dir / f"{target_id}.mha", np.zeros((6, 6, 6), dtype=np.int16))

    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "sample_id": sample_id,
                "target_id": target_id,
                "condition": "null",
                "cond_mask_source_id": None,
                "seed": 0,
                "cfg_scale_text": 5.0,
                "cfg_scale_mask": 0.0,
                "run_id": "test",
                "source_labels": None,
                "label_overlap": None,
            }
        )
    )

    df = build_per_sample(
        scan_ids=[],
        pred_dir=pred_dir,
        gt_dir=gt_dir,
        subgroup_cfg=_CFG,
        manifest_path=manifest_path,
        seg_backend="totalseg",  # stubbed via the autouse fixture above
    )
    row = df.iloc[0]
    assert row["condition"] == "null"
    assert row["cond_mask_source_id"] is None or (
        isinstance(row["cond_mask_source_id"], float)
        and np.isnan(row["cond_mask_source_id"])
    )
