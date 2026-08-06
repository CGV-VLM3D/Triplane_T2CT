"""condition_setlevel.run: per-condition FID/FVD is scored as its OWN subset (no pooling across
conditions — that premise-violating pool is what run_eval.py refuses), and each condition's
predictions are handed to ``CTGenEvaluator`` under the run's GT-by-stem view (``gt_view/``), not
the raw census.

``CTGenEvaluator`` itself is monkeypatched (no CT-CLIP/GPU needed) — this test is about the
ROUTING (which files land in which subset dir, which GT dir is passed, the small-n skip), not
FID/FVD numerics (already covered by the production code CTGenEvaluator itself came from).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.eval.analysis import condition_setlevel
from tests.eval_analysis._helpers import write_mha


class _FakeEvaluator:
    """Records (pred_dir contents, gt_dir) per call; returns deterministic fake metrics."""

    calls: list[dict] = []

    def __init__(self, gt_dir, metrics, ctclip_ckpt=None, fid_profile="docker_n300"):
        self.gt_dir = Path(gt_dir)
        self.metrics = metrics
        self.fid_profile = fid_profile

    def evaluate(self, pred_dir, out_dir):
        pred_dir = Path(pred_dir)
        stems = sorted(p.stem for p in pred_dir.glob("*.mha"))
        _FakeEvaluator.calls.append(
            {"gt_dir": self.gt_dir, "pred_stems": stems, "out_dir": Path(out_dir)}
        )
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        return {
            "FID_2p5D_Avg": 42.0 + len(stems),
            "FID_2p5D_XY": 1.0,
            "FID_2p5D_YZ": 2.0,
            "FID_2p5D_XZ": 3.0,
            "FVD_CTCLIP": 0.5,
            "fid_num_images": len(stems),
        }


@pytest.fixture(autouse=True)
def _patch_evaluator(monkeypatch):
    _FakeEvaluator.calls = []
    import src.eval.tasks.ctgen as ctgen_mod

    monkeypatch.setattr(ctgen_mod, "CTGenEvaluator", _FakeEvaluator)
    yield
    _FakeEvaluator.calls = []


def _setup(tmp_path: Path, condition_sizes: dict[str, int]) -> tuple[Path, Path, Path]:
    """Build pred_dir/gt_view_dir/per_sample.csv for the given {condition: n} sizes."""
    pred_dir = tmp_path / "predictions"
    gt_view_dir = tmp_path / "gt_view"
    rows = []
    for condition, n in condition_sizes.items():
        for i in range(n):
            sid = f"t{i}__{condition}"
            write_mha(pred_dir / f"{sid}.mha", np.zeros((2, 2, 2), np.int16))
            write_mha(gt_view_dir / f"{sid}.mha", np.zeros((2, 2, 2), np.int16))
            rows.append(
                {"sample_id": sid, "target_id": f"t{i}", "condition": condition}
            )
    csv_path = tmp_path / "per_sample.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return pred_dir, gt_view_dir, csv_path


def test_scores_each_condition_as_its_own_subset(tmp_path):
    pred_dir, gt_view_dir, csv_path = _setup(
        tmp_path, {"gt": 12, "label_matched_swap": 12, "label_mismatched_swap": 12}
    )
    out_dir = tmp_path / "condition_fid"

    df = condition_setlevel.run(pred_dir, gt_view_dir, csv_path, out_dir)

    assert set(df["condition"]) == {"gt", "label_matched_swap", "label_mismatched_swap"}
    assert (df["n"] == 12).all()
    # each call saw ONLY its own condition's stems -- no pooling across conditions
    assert len(_FakeEvaluator.calls) == 3
    for call in _FakeEvaluator.calls:
        conditions_in_call = {s.split("__")[1] for s in call["pred_stems"]}
        assert len(conditions_in_call) == 1
        assert len(call["pred_stems"]) == 12


def test_uses_gt_view_not_the_raw_census(tmp_path):
    pred_dir, gt_view_dir, csv_path = _setup(tmp_path, {"gt": 10})
    condition_setlevel.run(pred_dir, gt_view_dir, csv_path, tmp_path / "out")

    assert _FakeEvaluator.calls[0]["gt_dir"] == gt_view_dir


def test_small_condition_is_skipped_not_blanked(tmp_path):
    pred_dir, gt_view_dir, csv_path = _setup(
        tmp_path, {"gt": 10, "label_matched_swap": 3}
    )
    df = condition_setlevel.run(pred_dir, gt_view_dir, csv_path, tmp_path / "out")

    assert list(df["condition"]) == ["gt"]  # matched (n=3) skipped, not a NaN row
    assert len(_FakeEvaluator.calls) == 1


def test_subset_symlink_dirs_are_cleaned_up(tmp_path):
    pred_dir, gt_view_dir, csv_path = _setup(tmp_path, {"gt": 10})
    out_dir = tmp_path / "out"
    condition_setlevel.run(pred_dir, gt_view_dir, csv_path, out_dir)

    assert not (out_dir / "_pred_subset").exists()


def test_missing_condition_column_raises(tmp_path):
    pred_dir = tmp_path / "predictions"
    csv_path = tmp_path / "per_sample.csv"
    pd.DataFrame([{"sample_id": "a", "target_id": "a"}]).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="condition"):
        condition_setlevel.run(
            pred_dir, tmp_path / "gt_view", csv_path, tmp_path / "out"
        )


def test_csv_written_with_expected_columns(tmp_path):
    pred_dir, gt_view_dir, csv_path = _setup(
        tmp_path, {"gt": 10, "label_mismatched_swap": 10}
    )
    out_dir = tmp_path / "out"
    condition_setlevel.run(pred_dir, gt_view_dir, csv_path, out_dir)

    on_disk = pd.read_csv(out_dir / "condition_fid_fvd.csv")
    assert set(on_disk["condition"]) == {"gt", "label_mismatched_swap"}
    assert "FID_2p5D_Avg" in on_disk.columns and "FVD_CTCLIP" in on_disk.columns
