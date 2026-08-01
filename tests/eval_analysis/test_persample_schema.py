"""persample.py: plain-run PK identity (sample_id==scan_id==target_id), and manifest-driven
multi-row-per-target rows with explicit label_<name>/source_label_<name> columns.

Uses real scan ids from the on-disk CT-RATE valid label CSV (no monkeypatching needed) with
tiny fake ``.mha`` files so gen_path/gt_path existence checks are exercised for real.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.analysis import subgroup_config
from src.eval.analysis.persample import build_per_sample
from tests.eval_analysis._helpers import write_mha

_SUBGROUP_CFG = subgroup_config.load("/workspace/configs/eval/subgroup/default.yaml")
_SCAN_IDS = ["valid_1_a_1", "valid_1004_a_2"]


def _make_dirs(tmp_path: Path) -> tuple[Path, Path]:
    pred_dir = tmp_path / "predictions"
    gt_dir = tmp_path / "gt"
    tiny = np.zeros((4, 4, 4), dtype=np.int16)
    for sid in _SCAN_IDS:
        write_mha(pred_dir / f"{sid}.mha", tiny)
        write_mha(gt_dir / f"{sid}.mha", tiny)
    return pred_dir, gt_dir


def test_plain_run_pk_identity(tmp_path: Path):
    pred_dir, gt_dir = _make_dirs(tmp_path)
    df = build_per_sample(
        scan_ids=_SCAN_IDS,
        pred_dir=pred_dir,
        gt_dir=gt_dir,
        subgroup_cfg=_SUBGROUP_CFG,
        is_mask_model=False,
        seg_backend=None,  # no GPU/TotalSegmentator needed for this schema test
    )
    assert len(df) == 2
    assert (df["sample_id"] == df["scan_id"]).all()
    assert (df["sample_id"] == df["target_id"]).all()
    assert set(df["sample_id"]) == set(_SCAN_IDS)
    # dice disabled -> always "partial" with dice_disabled reason (not "ok", not silently blank)
    assert (df["status"] == "partial").all()
    assert df["failure_reason"].str.contains("dice_disabled").all()
    # real label data actually joined (not placeholder)
    assert df.loc[df["target_id"] == "valid_1_a_1", "label_burden"].iloc[0] == 1


def test_is_mask_model_populates_cond_mask_source_id(tmp_path: Path):
    pred_dir, gt_dir = _make_dirs(tmp_path)
    df = build_per_sample(
        scan_ids=_SCAN_IDS,
        pred_dir=pred_dir,
        gt_dir=gt_dir,
        subgroup_cfg=_SUBGROUP_CFG,
        is_mask_model=True,
        seg_backend=None,
    )
    assert (df["condition"] == "gt").all()
    assert (df["cond_mask_source_id"] == df["target_id"]).all()


def test_non_mask_model_has_no_cond_source(tmp_path: Path):
    pred_dir, gt_dir = _make_dirs(tmp_path)
    df = build_per_sample(
        scan_ids=_SCAN_IDS,
        pred_dir=pred_dir,
        gt_dir=gt_dir,
        subgroup_cfg=_SUBGROUP_CFG,
        is_mask_model=False,
        seg_backend=None,
    )
    assert (df["condition"] == "none").all()
    assert df["cond_mask_source_id"].isna().all()


def test_manifest_drives_multirow_per_target(tmp_path: Path):
    pred_dir, gt_dir = _make_dirs(tmp_path)
    target_id = "valid_1_a_1"
    manifest_lines = [
        {
            "sample_id": f"{target_id}__gt__src-{target_id}__sm-1.0__seed-0",
            "target_id": target_id,
            "condition": "gt",
            "cond_mask_source_id": target_id,
            "seed": 0,
            "cfg_scale_text": 5.0,
            "cfg_scale_mask": 1.0,
            "run_id": "test_run",
            "source_labels": None,
            "label_overlap": None,
        },
        {
            "sample_id": f"{target_id}__label_matched_swap__src-valid_1004_a_2__sm-1.0__seed-0",
            "target_id": target_id,
            "condition": "label_matched_swap",
            "cond_mask_source_id": "valid_1004_a_2",
            "seed": 0,
            "cfg_scale_text": 5.0,
            "cfg_scale_mask": 1.0,
            "run_id": "test_run",
            "source_labels": {"Cardiomegaly": 1},
            "label_overlap": 1,
        },
    ]
    # both sample_ids must resolve to a real .mha for the gen_path check — reuse the tiny fixtures
    for rec in manifest_lines:
        write_mha(
            pred_dir / f"{rec['sample_id']}.mha", np.zeros((4, 4, 4), dtype=np.int16)
        )
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text("\n".join(json.dumps(r) for r in manifest_lines))

    df = build_per_sample(
        scan_ids=[],  # ignored when manifest is given
        pred_dir=pred_dir,
        gt_dir=gt_dir,
        subgroup_cfg=_SUBGROUP_CFG,
        manifest_path=manifest_path,
        seg_backend=None,
    )
    assert (
        len(df) == 2
    )  # two rows for the SAME target_id — "1 scan = 1 row" correctly broken
    assert (df["target_id"] == target_id).all()
    assert set(df["sample_id"]) == {r["sample_id"] for r in manifest_lines}
    matched_row = df[df["condition"] == "label_matched_swap"].iloc[0]
    assert matched_row["cond_mask_source_id"] == "valid_1004_a_2"
    assert matched_row["label_overlap"] == 1
    assert matched_row["source_label_Cardiomegaly"] == 1
    gt_row = df[df["condition"] == "gt"].iloc[0]
    assert pd.isna(
        gt_row["source_label_Cardiomegaly"]
    )  # gt condition has no source labels


def test_manifest_skips_malformed_lines_with_warning(tmp_path: Path, caplog):
    """A hand-edited manifest is untrusted input: bad JSON, missing required fields, and
    path-traversal id values must be skipped (skip+warn), not crash the whole build."""
    pred_dir, gt_dir = _make_dirs(tmp_path)
    good_id = "valid_1_a_1"
    write_mha(pred_dir / f"{good_id}.mha", np.zeros((4, 4, 4), dtype=np.int16))

    lines = [
        "{not valid json",  # invalid JSON
        json.dumps({"target_id": good_id}),  # missing sample_id
        json.dumps(
            {
                "sample_id": "../../etc/passwd",
                "target_id": good_id,
                "cond_mask_source_id": None,
            }
        ),  # path traversal in sample_id
        json.dumps(
            {"sample_id": good_id, "target_id": good_id, "cond_mask_source_id": None}
        ),  # the one well-formed record
    ]
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text("\n".join(lines))

    df = build_per_sample(
        scan_ids=[],
        pred_dir=pred_dir,
        gt_dir=gt_dir,
        subgroup_cfg=_SUBGROUP_CFG,
        manifest_path=manifest_path,
        seg_backend=None,
    )
    assert len(df) == 1  # only the well-formed record survives
    assert df.iloc[0]["sample_id"] == good_id


def test_manifest_rejects_traversal_in_cond_mask_source_id(tmp_path: Path):
    pred_dir, gt_dir = _make_dirs(tmp_path)
    target_id = "valid_1_a_1"
    write_mha(pred_dir / f"{target_id}.mha", np.zeros((4, 4, 4), dtype=np.int16))
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "sample_id": target_id,
                "target_id": target_id,
                "cond_mask_source_id": "../../secret",
            }
        )
    )
    df = build_per_sample(
        scan_ids=[],
        pred_dir=pred_dir,
        gt_dir=gt_dir,
        subgroup_cfg=_SUBGROUP_CFG,
        manifest_path=manifest_path,
        seg_backend=None,
    )
    assert len(df) == 0  # the unsafe record was skipped entirely
