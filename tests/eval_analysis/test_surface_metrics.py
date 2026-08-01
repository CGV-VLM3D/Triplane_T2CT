"""seg_metrics.py surface distances: HD95 / HD / ASSD in mm.

Two things are guarded here:

1. **Equivalence to MONAI's public API.** ``_surface_metrics_per_organ`` derives all three
   metrics from ONE ``get_edge_surface_distance`` pass instead of calling
   ``compute_hausdorff_distance`` twice + ``compute_average_surface_distance`` once (3x the
   cost). ``test_matches_monai_public_api`` pins that the fused version returns exactly what the
   public functions return — if a MONAI upgrade changes either reduction, this fails.
2. **Spacing/axis order and empty-mask policy.** Verified interactively against MONAI 2026-07-27
   (see plan §7): a 1-voxel shift along one axis, with anisotropic spacing, must report the mm
   distance for THAT axis's spacing — swapping the spacing tuple order silently gives a
   different (wrong) answer.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from monai.metrics import compute_average_surface_distance, compute_hausdorff_distance

from src.eval.analysis.seg_metrics import _surface_metrics_per_organ


def _mask(shape=(10, 10, 10)):
    return np.zeros(shape, dtype=np.int64)


def _hd95(pred, ref, spacing_xyz):
    return _surface_metrics_per_organ(pred, ref, spacing_xyz)["hd95_mm"]


def test_hd95_anisotropic_spacing_matches_shifted_axis():
    pred = _mask()
    ref = _mask()
    pred[5, 5, 5] = 1  # organ 1 (lung), single voxel
    ref[5, 5, 6] = 1  # shifted 1 voxel along the LAST axis
    spacing_xyz = (1.0, 1.0, 3.0)  # last axis has spacing 3mm
    hd95 = _hd95(pred, ref, spacing_xyz)
    assert abs(hd95["lung"] - 3.0) < 1e-4, (
        "1-voxel shift on the 3mm axis should read 3.0mm"
    )


def test_hd95_wrong_spacing_order_gives_wrong_answer():
    """Sanity check that axis/spacing-order actually matters (guards against a silent swap bug
    upstream — if this assertion ever fails, spacing order stopped mattering, which would be
    surprising and worth re-checking)."""
    pred = _mask()
    ref = _mask()
    pred[5, 5, 5] = 1
    ref[5, 5, 6] = 1  # shift is on the LAST array axis
    correct = _hd95(pred, ref, (1.0, 1.0, 3.0))["lung"]
    wrong = _hd95(pred, ref, (3.0, 1.0, 1.0))["lung"]  # 3mm on the WRONG axis
    assert abs(correct - 3.0) < 1e-4
    assert abs(wrong - 1.0) < 1e-4
    assert correct != wrong


def test_hd95_both_empty_is_nan():
    pred = _mask()
    ref = _mask()
    hd95 = _hd95(pred, ref, (1.0, 1.0, 1.0))
    assert math.isnan(hd95["lung"])


def test_hd95_one_empty_is_nan():
    pred = _mask()
    pred[5, 5, 5] = 1
    ref = _mask()
    hd95 = _hd95(pred, ref, (1.0, 1.0, 1.0))
    assert math.isnan(
        hd95["lung"]
    )  # MONAI: Hausdorff undefined without two non-empty sets


def test_one_empty_side_is_nan_for_every_metric():
    """Our one deliberate deviation from upstream, pinned in both directions.

    MONAI is self-inconsistent when exactly one side of an organ is empty: HD95 returns NaN
    (``torch.quantile`` of an all-inf tensor is NaN) while HD and ASSD return ``+inf``. An inf
    reaching ``per_sample.csv`` would turn every downstream mean into inf, so we map non-finite
    to NaN. The first half of this test documents upstream's actual behaviour so the deviation
    stays visible; the second half pins ours.
    """
    empty, present = _mask(), _mask()
    present[5, 5, 5] = 1

    # BOTH directions. The one that actually occurs in production is (pred empty, ref present):
    # the reference ts_seg always has the organ, and a weak model can fail to generate it.
    for label, pred, ref in (
        ("pred empty, ref present", empty, present),
        ("pred present, ref empty", present, empty),
    ):
        p_bin = torch.from_numpy((pred == 1).astype(np.float32))[None, None]
        r_bin = torch.from_numpy((ref == 1).astype(np.float32))[None, None]
        assert math.isnan(
            float(compute_hausdorff_distance(p_bin, r_bin, percentile=95)[0, 0])
        ), label
        assert math.isinf(
            float(compute_hausdorff_distance(p_bin, r_bin, percentile=None)[0, 0])
        ), label
        assert math.isinf(
            float(compute_average_surface_distance(p_bin, r_bin, symmetric=True)[0, 0])
        ), label

        out = _surface_metrics_per_organ(pred, ref, (1.0, 1.0, 1.0))
        for metric in ("hd95_mm", "hd_mm", "assd_mm"):
            assert math.isnan(out[metric]["lung"]), f"{label}/{metric}"
            assert math.isnan(out[metric]["_mean"]), (
                f"{label}/{metric}"
            )  # no finite organ
        assert out["_scored"]["n"] == 0, label  # nothing was scorable


def test_scored_record_names_the_missing_organ_and_side():
    """``_scored`` must expose the varying ``_mean`` denominator AND which side went empty."""
    pred = _mask()
    ref = _mask()
    for organ_id in (1, 2, 3):  # lung/heart/aorta present on both sides
        pred[organ_id, 5, 5] = organ_id
        ref[organ_id, 5, 6] = organ_id
    ref[4, 5, 5] = (
        4  # esophagus in the reference only -> the model failed to generate it
    )

    out = _surface_metrics_per_organ(pred, ref, (1.0, 1.0, 1.0))
    assert out["_scored"]["n"] == 3  # the mean averaged 3 organs, not 4
    assert (
        out["_scored"]["missing"] == "esophagus:pred"
    )  # and the model's side is named
    assert math.isnan(out["hd95_mm"]["esophagus"])

    ref[4, 5, 5] = 0  # now absent on BOTH sides
    both = _surface_metrics_per_organ(pred, ref, (1.0, 1.0, 1.0))
    assert both["_scored"]["missing"] == "esophagus:both"


def test_omitting_an_organ_must_not_look_better_than_generating_it_badly():
    """The hazard `_scored` exists to expose, pinned end to end.

    A model that generates an organ badly is penalised by the surface mean; a model that omits it
    entirely has it dropped from the denominator and can score BETTER. Dice penalises both. The
    numbers are allowed to move, but `n` must always reveal which mean covered fewer organs.
    """
    ref = _mask((40, 40, 40))
    ref[8:20, 8:20, 8:20] = 1
    ref[24:30, 8:14, 8:14] = 2
    ref[10:14, 24:30, 8:30] = 3
    ref[16:18, 16:18, 8:30] = 4  # esophagus: thin and long, the hardest organ

    generated_badly = ref.copy()
    generated_badly[16:18, 16:18, 8:30] = 0
    generated_badly[26:28, 26:28, 8:30] = 4  # present, but in the wrong place
    omitted = ref.copy()
    omitted[16:18, 16:18, 8:30] = 0  # not generated at all

    bad = _surface_metrics_per_organ(generated_badly, ref, (1.0, 1.0, 1.0))
    gone = _surface_metrics_per_organ(omitted, ref, (1.0, 1.0, 1.0))

    assert bad["_scored"]["n"] == 4 and bad["_scored"]["missing"] == ""
    assert gone["_scored"]["n"] == 3 and gone["_scored"]["missing"] == "esophagus:pred"
    # the omission scores better on the surface mean despite being anatomically worse
    assert gone["hd95_mm"]["_mean"] < bad["hd95_mm"]["_mean"]


def test_hd_exposes_stray_voxel_that_hd95_hides():
    """The reason plain HD is recorded at all: one hallucinated voxel far from the organ leaves
    HD95 untouched but blows HD up. If this stops holding, HD has lost its purpose."""
    pred = np.zeros((60, 60, 60), dtype=np.int64)
    ref = np.zeros((60, 60, 60), dtype=np.int64)
    pred[20:40, 20:40, 20:40] = 1
    ref[20:40, 20:40, 20:40] = 1
    pred[58, 58, 58] = 1  # stray segmentation voxel, far from the organ

    out = _surface_metrics_per_organ(pred, ref, (0.75, 0.75, 1.3))
    assert out["hd95_mm"]["lung"] == 0.0
    assert out["hd_mm"]["lung"] > 25.0
    assert out["assd_mm"]["lung"] > 0.0  # ASSD notices it too, but only faintly


def test_matches_monai_public_api():
    """The fused single-pass helper == MONAI's three public calls, organ by organ.

    This is the guard that lets us keep MONAI's copied reduction lines in ``seg_metrics.py``:
    if upstream's reduction ever changes, the copy and the public API diverge and this fails.
    """
    pred = np.zeros((24, 24, 24), dtype=np.int64)
    ref = np.zeros((24, 24, 24), dtype=np.int64)
    pred[4:14, 4:14, 4:14] = 1  # lung: overlapping blocks, offset by 1 voxel
    ref[5:15, 4:14, 4:14] = 1
    pred[16:20, 16:20, 6:12] = 2  # heart: offset on another axis
    ref[16:20, 15:19, 6:12] = 2
    pred[2, 2, 2] = 3  # aorta: single voxel each, far apart
    ref[20, 20, 20] = 3
    # organ 4 (esophagus) is left empty on both sides -> NaN for every metric

    spacing_xyz = (0.75, 0.9, 1.3)
    fused = _surface_metrics_per_organ(pred, ref, spacing_xyz)

    for organ_id, name in enumerate(["lung", "heart", "aorta", "esophagus"], start=1):
        p_bin = torch.from_numpy((pred == organ_id).astype(np.float32))[None, None]
        r_bin = torch.from_numpy((ref == organ_id).astype(np.float32))[None, None]
        expected = {
            "hd95_mm": compute_hausdorff_distance(
                p_bin, r_bin, percentile=95, spacing=spacing_xyz
            )[0, 0],
            "hd_mm": compute_hausdorff_distance(
                p_bin, r_bin, percentile=None, spacing=spacing_xyz
            )[0, 0],
            "assd_mm": compute_average_surface_distance(
                p_bin, r_bin, symmetric=True, spacing=spacing_xyz
            )[0, 0],
        }
        for metric, want in expected.items():
            got = fused[metric][name]
            if math.isnan(float(want)):
                assert math.isnan(got), f"{metric}/{name}: expected NaN, got {got}"
            else:
                assert abs(got - float(want)) < 1e-5, (
                    f"{metric}/{name}: fused {got} != MONAI {float(want)}"
                )
