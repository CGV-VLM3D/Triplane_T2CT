"""orchestrate.py: analysis/ output layout, and top-level (pre-existing) files left untouched."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval.analysis.orchestrate import run_metrics_analysis
from tests.eval_analysis._helpers import write_mha

_SCAN_IDS = ["valid_1_a_1", "valid_1004_a_2"]
_SUBGROUP_CFG_PATH = "/workspace/configs/eval/subgroup/default.yaml"


def _make_run_dir(tmp_path: Path) -> Path:
    out_dir = tmp_path / "run"
    pred_dir = out_dir / "predictions"
    gt_dir = tmp_path / "gt"
    tiny = np.zeros((4, 4, 4), dtype=np.int16)
    for sid in _SCAN_IDS:
        write_mha(pred_dir / f"{sid}.mha", tiny)
        write_mha(gt_dir / f"{sid}.mha", tiny)
    # pre-existing top-level artifact that must survive untouched
    (out_dir / "metrics.json").write_text(json.dumps({"FID_2p5D_Avg": 4.04}, indent=2))
    return out_dir


def test_analysis_subfolder_layout(tmp_path: Path):
    out_dir = _make_run_dir(tmp_path)
    gt_dir = tmp_path / "gt"
    run_metrics_analysis(
        scan_ids=_SCAN_IDS,
        pred_dir=out_dir / "predictions",
        gt_dir=gt_dir,
        out_dir=out_dir,
        subgroup_config_path=_SUBGROUP_CFG_PATH,
        per_sample=True,
        subgroup_flag=True,
        seg_backend=None,  # dice/hd95 off — no GPU needed
    )

    analysis_dir = out_dir / "analysis"
    assert (analysis_dir / "per_sample.csv").is_file()
    assert (analysis_dir / "subgroup" / "per_abnormality.csv").is_file()
    assert (analysis_dir / "subgroup" / "label_burden.csv").is_file()
    assert (analysis_dir / "subgroup" / "organ_cluster.csv").is_file()
    assert (analysis_dir / "SUMMARY.md").is_file()


def test_top_level_files_untouched(tmp_path: Path):
    out_dir = _make_run_dir(tmp_path)
    gt_dir = tmp_path / "gt"
    before = (out_dir / "metrics.json").read_text()

    run_metrics_analysis(
        scan_ids=_SCAN_IDS,
        pred_dir=out_dir / "predictions",
        gt_dir=gt_dir,
        out_dir=out_dir,
        subgroup_config_path=_SUBGROUP_CFG_PATH,
        per_sample=True,
        seg_backend=None,
    )
    after = (out_dir / "metrics.json").read_text()
    assert before == after


def test_no_analysis_dir_when_all_flags_false(tmp_path: Path):
    out_dir = _make_run_dir(tmp_path)
    gt_dir = tmp_path / "gt"
    run_metrics_analysis(
        scan_ids=_SCAN_IDS,
        pred_dir=out_dir / "predictions",
        gt_dir=gt_dir,
        out_dir=out_dir,
        subgroup_config_path=_SUBGROUP_CFG_PATH,
        # every flag left at its False default
    )
    assert not (out_dir / "analysis").exists()
