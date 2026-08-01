"""labels.py: normal/disease, burden bands, overlapping cluster membership, patient_id parsing."""

from __future__ import annotations

import numpy as np

from src.eval.analysis.labels import (
    LABEL_NAMES,
    burden_band,
    cluster_membership,
    label_burden,
    label_class,
    patient_id_of,
)

_BANDS = {"all_zero": (0, 0), "1_3": (1, 3), "4_7": (4, 7), "8_plus": (8, 18)}


def _vec_with_n_positive(n: int) -> np.ndarray:
    v = np.zeros(18, dtype=np.int64)
    v[:n] = 1
    return v


def test_all_zero_is_normal():
    v = np.zeros(18, dtype=np.int64)
    assert label_burden(v) == 0
    assert label_class(v) == "normal"


def test_any_positive_is_disease():
    v = _vec_with_n_positive(1)
    assert label_class(v) == "disease"


def test_burden_band_boundaries():
    # 0,1,3,4,7,8 per the plan's explicit boundary test list
    assert burden_band(0, _BANDS) == "all_zero"
    assert burden_band(1, _BANDS) == "1_3"
    assert burden_band(3, _BANDS) == "1_3"
    assert burden_band(4, _BANDS) == "4_7"
    assert burden_band(7, _BANDS) == "4_7"
    assert burden_band(8, _BANDS) == "8_plus"


def test_cluster_membership_overlap():
    """A sample positive in labels from two different clusters belongs to BOTH (multi-membership)."""
    clusters = {
        "cluster_a": [LABEL_NAMES[0], LABEL_NAMES[1]],
        "cluster_b": [LABEL_NAMES[1], LABEL_NAMES[2]],
    }
    vec = np.zeros(18, dtype=np.int64)
    vec[1] = 1  # positive only in the label shared by both clusters
    membership = cluster_membership(vec, clusters)
    assert membership["cluster_a"] is True
    assert membership["cluster_b"] is True


def test_cluster_membership_none():
    clusters = {"cluster_a": [LABEL_NAMES[0]]}
    vec = np.zeros(18, dtype=np.int64)
    assert cluster_membership(vec, clusters)["cluster_a"] is False


def test_patient_id_of():
    assert patient_id_of("valid_1000_a_1") == "valid_1000"
    assert patient_id_of("valid_1_a_2") == "valid_1"
    assert patient_id_of("train_20000_b_1") == "train_20000"
