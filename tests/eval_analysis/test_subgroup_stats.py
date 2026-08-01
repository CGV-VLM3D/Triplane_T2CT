"""subgroup.py: groupby reuse (values match direct pandas groupby over the same per_sample
table — no recomputation), n_valid+n_missing consistency, and no CI columns present."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.eval.analysis import subgroup
from src.eval.analysis.labels import LABEL_NAMES


def _synthetic_per_sample(n=20) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        row = {"sample_id": f"s{i}", "target_id": f"s{i}"}
        for name in LABEL_NAMES:
            row[f"label_{name}"] = int(rng.random() < 0.2)
        row["burden_band"] = rng.choice(["all_zero", "1_3", "4_7", "8_plus"])
        row["cluster_lung_parenchyma_airway"] = bool(rng.random() < 0.5)
        row["cluster_pleural_pericardial"] = bool(rng.random() < 0.5)
        row["clip_t2i"] = (
            rng.normal(50, 5) if rng.random() > 0.1 else np.nan
        )  # some missing
        row["clip_i2i"] = rng.normal(30, 5)
        row["dice_to_input_mask"] = np.nan
        row["dice_to_gt_mask"] = np.nan
        row["hd95_mm_to_input_mask"] = np.nan
        row["hd95_mm_to_gt_mask"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def test_per_abnormality_matches_direct_groupby():
    df = _synthetic_per_sample()
    result = subgroup.per_abnormality(df)
    assert len(result) == 18  # all 18 labels always present, even if n=0 for some

    for name in LABEL_NAMES:
        col = f"label_{name}"
        direct_n = int((df[col] == 1).sum())
        row = result[result["label"] == name].iloc[0]
        assert row["n"] == direct_n
        if direct_n > 0:
            direct_mean = df.loc[df[col] == 1, "clip_t2i"].dropna().mean()
            if pd.isna(direct_mean):
                assert pd.isna(row["clip_t2i_mean"])
            else:
                assert abs(row["clip_t2i_mean"] - direct_mean) < 1e-9


def test_n_valid_plus_n_missing_equals_n():
    df = _synthetic_per_sample()
    result = subgroup.label_burden(
        df, {"all_zero": (0, 0), "1_3": (1, 3), "4_7": (4, 7), "8_plus": (8, 18)}
    )
    for _, row in result.iterrows():
        assert row["clip_t2i_n_valid"] + row["clip_t2i_n_missing"] == row["n"]


def test_organ_cluster_overlapping_membership_counted_in_both():
    df = _synthetic_per_sample()
    result = subgroup.organ_cluster(
        df, {"lung_parenchyma_airway": [], "pleural_pericardial": []}
    )
    n_lung = int(df["cluster_lung_parenchyma_airway"].sum())
    n_pleural = int(df["cluster_pleural_pericardial"].sum())
    assert result.set_index("cluster").loc["lung_parenchyma_airway", "n"] == n_lung
    assert result.set_index("cluster").loc["pleural_pericardial", "n"] == n_pleural


def test_no_ci_columns_present():
    df = _synthetic_per_sample()
    result = subgroup.per_abnormality(df)
    ci_like = [c for c in result.columns if "ci" in c.lower() or "conf" in c.lower()]
    assert ci_like == [], f"unexpected CI-like columns: {ci_like}"
    expected_stats = {"mean", "std", "median", "n_valid", "n_missing"}
    for stat in expected_stats:
        assert any(c.endswith(f"_{stat}") for c in result.columns)
