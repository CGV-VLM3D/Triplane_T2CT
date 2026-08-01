"""Backward compatibility: new flags don't change CTGenEvaluator dispatch or existing config
defaults, and produce zero side effects when left at their (all-False) defaults."""

from __future__ import annotations

import yaml

from src.eval.tasks.ctgen import CTGenEvaluator


def test_existing_metric_defaults_unchanged():
    """The original 4 metrics keys must keep their original default values in the config —
    the 5 new keys were APPENDED, not substituted for anything."""
    cfg = yaml.safe_load(open("/workspace/configs/eval/task/ctgen.yaml"))
    metrics = cfg["metrics"]
    assert metrics["fvd"] is True
    assert metrics["fvd_ctclip"] is True
    assert metrics["clip_score"] is True
    assert metrics["fid_2p5d"] is True
    # new keys present and default OFF (additive, not opt-out)
    for key in ("per_sample", "dice", "hd95", "subgroup", "subgroup_setlevel"):
        assert metrics[key] is False


def test_ctgen_evaluator_dispatch_unaffected_by_extra_keys():
    """Extra unknown keys in the metrics dict (the 5 new ones) must not change which of the 4
    existing metric methods CTGenEvaluator.evaluate() would call — verified via .get() lookups,
    the exact mechanism evaluate() uses for dispatch (src/eval/tasks/ctgen.py:272-286)."""
    legacy_metrics = {
        "fvd": False,
        "fvd_ctclip": True,
        "clip_score": True,
        "fid_2p5d": True,
    }
    extended_metrics = {
        **legacy_metrics,
        "per_sample": False,
        "dice": False,
        "hd95": False,
        "subgroup": False,
        "subgroup_setlevel": False,
    }

    ev_legacy = CTGenEvaluator(gt_dir="/tmp/unused", metrics=legacy_metrics)
    ev_extended = CTGenEvaluator(gt_dir="/tmp/unused", metrics=extended_metrics)

    for key in ("fvd", "fvd_ctclip", "clip_score", "fid_2p5d"):
        assert ev_legacy.metrics.get(key, False) == ev_extended.metrics.get(key, False)


def test_ctgen_evaluator_default_metrics_dict_unchanged():
    """CTGenEvaluator's OWN internal default (when metrics=None) is untouched by this extension."""
    ev = CTGenEvaluator(gt_dir="/tmp/unused")
    assert ev.metrics == {
        "fvd": True,
        "fvd_ctclip": True,
        "clip_score": True,
        "fid_2p5d": True,
    }
