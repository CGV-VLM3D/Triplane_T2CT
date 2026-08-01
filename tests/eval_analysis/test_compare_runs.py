"""compare_runs.py: per-sample deltas, missing/mismatched sample handling, determinism, no CI."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.compare_runs import compute_deltas, pair_runs, run, subgroup_deltas


def _baseline_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_id": ["s1", "s2", "s3"],
            "clip_t2i": [50.0, 55.0, 60.0],
            "label_Cardiomegaly": [1, 0, 1],
            "burden_band": ["1_3", "all_zero", "4_7"],
            "cluster_lung_parenchyma_airway": [True, False, True],
        }
    )


def _treatment_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_id": [
                "s1",
                "s2",
                "s4",
            ],  # s3 missing from treatment, s4 missing from baseline
            "clip_t2i": [45.0, 50.0, 70.0],
        }
    )


def test_pair_runs_inner_joins_and_drops_mismatched():
    merged = pair_runs(_baseline_df(), _treatment_df())
    assert set(merged["target_id"]) == {"s1", "s2"}  # only the common scans


def test_compute_deltas_arithmetic():
    merged = pair_runs(_baseline_df(), _treatment_df())
    merged = compute_deltas(merged)
    row_s1 = merged[merged["target_id"] == "s1"].iloc[0]
    assert abs(row_s1["delta_clip_t2i"] - (45.0 - 50.0)) < 1e-9


def test_subgroup_deltas_uses_baseline_side_labels():
    merged = pair_runs(_baseline_df(), _treatment_df())
    merged = compute_deltas(merged)
    tables = subgroup_deltas(merged)
    assert "per_abnormality" in tables
    row = tables["per_abnormality"]
    cardiomegaly_row = row[row["label"] == "Cardiomegaly"].iloc[0]
    assert cardiomegaly_row["n"] == 1  # only s1 is positive AND survived the pairing


def test_deterministic(tmp_path: Path):
    b_csv = tmp_path / "b.csv"
    t_csv = tmp_path / "t.csv"
    _baseline_df().to_csv(b_csv, index=False)
    _treatment_df().to_csv(t_csv, index=False)
    r1 = run(b_csv, t_csv, tmp_path / "out1")
    r2 = run(b_csv, t_csv, tmp_path / "out2")
    pd.testing.assert_frame_equal(
        r1["per_sample_delta"].reset_index(drop=True),
        r2["per_sample_delta"].reset_index(drop=True),
    )


def test_no_ci_columns_in_output():
    merged = pair_runs(_baseline_df(), _treatment_df())
    merged = compute_deltas(merged)
    tables = subgroup_deltas(merged)
    for df in tables.values():
        ci_like = [c for c in df.columns if "ci" in c.lower()]
        assert ci_like == []
