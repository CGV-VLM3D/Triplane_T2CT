"""subgroup_setlevel.py: below_threshold marker (not a validity claim — no "reliable" field),
real_n/gen_n always reported, computed even for tiny subgroups."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.eval.analysis import subgroup_config
from src.eval.analysis.persample import build_per_sample
from src.eval.analysis.subgroup_setlevel import _build_feature_index, _fid_subset, run
from tests.eval_analysis._helpers import write_mha

_CFG = subgroup_config.load("/workspace/configs/eval/subgroup/default.yaml")
_SCAN_IDS = ["valid_1_a_1", "valid_1004_a_2"]


def _write_fake_feature(path: Path, n_slices=4, dim=8):
    path.parent.mkdir(parents=True, exist_ok=True)
    feats = tuple(torch.randn(n_slices, dim) for _ in range(3))  # (xy, yz, zx)
    torch.save(feats, path)


def test_fid_subset_reports_n_and_finite_fid(tmp_path: Path):
    features_dir = tmp_path / "fid_features"
    for sid in _SCAN_IDS:
        _write_fake_feature(features_dir / "gt" / f"{sid}.pt")
        _write_fake_feature(features_dir / "pred" / f"{sid}.pt")

    gt_index = _build_feature_index(features_dir / "gt")
    pred_index = _build_feature_index(features_dir / "pred")
    result = _fid_subset(gt_index, pred_index, set(_SCAN_IDS), set(_SCAN_IDS))
    assert result["real_n"] == 2
    assert result["gen_n"] == 2
    assert np.isfinite(result["FID_2p5D_Avg"])


def test_fid_subset_missing_files_gives_nan_but_reports_n():
    # empty indices — simulates ids with no cached feature file (nothing found for them)
    result = _fid_subset({}, {}, {"a", "b"}, {"c"})
    assert result["real_n"] == 0
    assert result["gen_n"] == 0
    assert np.isnan(result["FID_2p5D_Avg"])


def test_below_threshold_no_reliable_field(tmp_path: Path):
    pred_dir = tmp_path / "predictions"
    gt_dir = tmp_path / "gt"
    tiny = np.zeros((4, 4, 4), dtype=np.int16)
    for sid in _SCAN_IDS:
        write_mha(pred_dir / f"{sid}.mha", tiny)
        write_mha(gt_dir / f"{sid}.mha", tiny)

    per_sample_df = build_per_sample(
        scan_ids=_SCAN_IDS,
        pred_dir=pred_dir,
        gt_dir=gt_dir,
        subgroup_cfg=_CFG,
        seg_backend=None,
    )
    per_sample_csv = tmp_path / "per_sample.csv"
    per_sample_df.to_csv(per_sample_csv, index=False)

    run_out_dir = tmp_path / "run"
    for sid in _SCAN_IDS:
        _write_fake_feature(run_out_dir / "fid_features" / "gt" / f"{sid}.pt")
        _write_fake_feature(run_out_dir / "fid_features" / "pred" / f"{sid}.pt")

    small_n_cfg = replace(
        _CFG, subgroup_fid_small_n=100
    )  # 2 << 100 -> below_threshold=True
    result_df = run(
        run_out_dir=run_out_dir,
        per_sample_csv=per_sample_csv,
        subgroup_cfg=small_n_cfg,
        analysis_out_dir=tmp_path / "setlevel",
    )

    assert "reliable" not in result_df.columns  # no validity-implying field
    assert "below_threshold" in result_df.columns
    overall_row = result_df[result_df["axis"] == "overall"].iloc[0]
    assert overall_row["gen_n"] == 2
    assert overall_row["below_threshold"] == True  # noqa: E712

    large_n_cfg = replace(
        _CFG, subgroup_fid_small_n=1
    )  # 2 >= 1 -> below_threshold=False
    result_df2 = run(
        run_out_dir=run_out_dir,
        per_sample_csv=per_sample_csv,
        subgroup_cfg=large_n_cfg,
        analysis_out_dir=tmp_path / "setlevel2",
    )
    overall_row2 = result_df2[result_df2["axis"] == "overall"].iloc[0]
    assert overall_row2["below_threshold"] == False  # noqa: E712


def test_build_axes_overall_restricted_to_gt_index(tmp_path: Path):
    """Perf/correctness regression (live n=50 test, 2026-07-27): the label CSV (3039 rows) is a
    superset of the clean GT census with cached features (~3001) — 'overall' gt_ids must be
    restricted to gt_index's keys, or the ref-stats fast path's exact-count check never matches."""
    from src.eval.analysis.subgroup_setlevel import _build_axes

    gt_label_map = {f"scan{i}": np.zeros(18, dtype=np.int64) for i in range(5)}
    gt_index = {
        f"scan{i}": Path(f"/fake/scan{i}.pt") for i in range(3)
    }  # only 3 of 5 cached
    df = pd.DataFrame(
        {
            "target_id": ["scan0", "scan1"],
            "label_class": ["normal", "normal"],
            "burden_band": ["all_zero", "all_zero"],
        }
    )
    axes = _build_axes(df, gt_label_map, _CFG, gt_index)
    overall_gt_ids = dict((name, gt_ids) for name, gt_ids, _ in axes)["overall"]
    assert overall_gt_ids == set(gt_index.keys())  # NOT the full 5-entry label map
    assert len(overall_gt_ids) == 3


def test_fid_subset_uses_ref_stats_fast_path_when_counts_match(
    monkeypatch, tmp_path: Path
):
    """When gt_ids exactly matches the precomputed ref-stats' volume count, _fid_subset must
    take the fid_from_ref_stats_files_fast fast path (not restack GT files) — the actual perf
    fix from the live n=50 test."""
    import src.eval.analysis.subgroup_setlevel as sl

    called = {"fast_path": False}

    def _fake_fid_from_ref_stats_files_fast(pred_files, ref_stats):
        called["fast_path"] = True
        return {
            "FID_2p5D_XY": 1.0,
            "FID_2p5D_YZ": 1.0,
            "FID_2p5D_XZ": 1.0,
            "FID_2p5D_Avg": 1.0,
        }

    monkeypatch.setattr(
        "src.eval.tasks._fid_refstats.fid_from_ref_stats_files_fast",
        _fake_fid_from_ref_stats_files_fast,
    )

    features_dir = tmp_path / "fid_features"
    for sid in _SCAN_IDS:
        _write_fake_feature(features_dir / "gt" / f"{sid}.pt")
        _write_fake_feature(features_dir / "pred" / f"{sid}.pt")
    gt_index = _build_feature_index(features_dir / "gt")
    pred_index = _build_feature_index(features_dir / "pred")

    ref_stats = {"n_volumes": len(_SCAN_IDS)}  # matches gt_ids count exactly
    result = sl._fid_subset(
        gt_index, pred_index, set(_SCAN_IDS), set(_SCAN_IDS), ref_stats=ref_stats
    )
    assert called["fast_path"] is True
    assert result["real_n"] == len(_SCAN_IDS)
    assert result["FID_2p5D_Avg"] == 1.0
