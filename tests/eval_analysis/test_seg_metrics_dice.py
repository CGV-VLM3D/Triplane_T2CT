"""seg_metrics.py: empty-mask Dice policy with tiny synthetic organ-group masks.

Policy (plan §3.8, user-confirmed): both-empty -> NaN (excluded); GT-only-empty or
pred-only-empty -> 0; full/partial overlap -> the standard Dice formula.
"""

from __future__ import annotations

import math

import numpy as np

from src.eval.analysis.seg_metrics import _dice_per_organ


def _mask(shape=(8, 8, 8)):
    return np.zeros(shape, dtype=np.int64)


def test_both_empty_is_nan():
    pred = _mask()
    ref = _mask()
    # organ 1 (lung) present nowhere in either -> NaN, excluded from _mean
    dice = _dice_per_organ(pred, ref)
    assert math.isnan(dice["lung"])


def test_gt_only_empty_is_zero():
    pred = _mask()
    pred[0:2, 0:2, 0:2] = 1  # pred has organ 1, ref does not
    ref = _mask()
    dice = _dice_per_organ(pred, ref)
    assert dice["lung"] == 0.0


def test_pred_only_empty_is_zero():
    pred = _mask()
    ref = _mask()
    ref[0:2, 0:2, 0:2] = 1
    dice = _dice_per_organ(pred, ref)
    assert dice["lung"] == 0.0


def test_perfect_overlap_is_one():
    pred = _mask()
    ref = _mask()
    pred[0:4, 0:4, 0:4] = 1
    ref[0:4, 0:4, 0:4] = 1
    dice = _dice_per_organ(pred, ref)
    assert dice["lung"] == 1.0


def test_partial_overlap_matches_hand_formula():
    pred = _mask()
    ref = _mask()
    pred[0:4, 0:4, 0:4] = 1  # 64 voxels
    ref[0:4, 0:4, 0:2] = 1  # 32 voxels, fully inside pred's region -> intersection 32
    dice = _dice_per_organ(pred, ref)
    expected = 2 * 32 / (64 + 32)  # = 0.6667
    assert abs(dice["lung"] - expected) < 1e-6


def test_mean_excludes_nan_but_includes_zero():
    pred = _mask()
    ref = _mask()
    # organ 1 (lung): both empty -> NaN, excluded
    # organ 2 (heart): gt-only-empty -> 0, included
    pred[0, 0, 0] = 2
    dice = _dice_per_organ(pred, ref)
    assert math.isnan(dice["lung"])
    assert dice["heart"] == 0.0
    # mean over {heart:0, aorta:NaN, esophagus:NaN} (all others both-empty) -> 0.0, not NaN
    assert dice["_mean"] == 0.0
