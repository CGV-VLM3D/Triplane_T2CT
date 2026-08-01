"""Dice / surface distances (HD95, HD, ASSD) between a generated CT's segmentation and reference
organ masks (requirement E).

Reference-mask loading (verified 2026-07-27 against a real generated volume — see plan §3.8 and
§7): the target/donor organ mask is reconstructed from the raw ``ts_seg`` NIfTI resized (NEAREST)
directly onto the prediction's own grid, with **no** ``Orientationd(RAS)`` reorientation. The
training-time mask-precompute scripts (e.g. ``scripts/precompute_wan_mask_latents.py``) DO apply
``Orientationd(RAS)`` because they build the conditioning mask in the model's internal RAS
working space; but every generated ``.mha`` is saved in **LPS** content (the ``ras_to_lps`` flip
applied at save time, matching the raw/no-reorient GT convention) — so comparing a *segmentation
of the saved prediction* against a reference mask must skip that RAS step, or the two are in
different coordinate spaces. Empirically: skipping ``Orientationd(RAS)`` gave lung Dice 0.9645
(matching the historically-validated ~0.97 for wan_mask); including it gave a silently-wrong 0.51.

**How the two grids are matched, and what that costs the mm metrics** (measured 2026-08-01).
There is exactly ONE spacing in play: the prediction's (``sitk.ReadImage(gen).GetSpacing()``),
because MONAI's distance functions take a single ``spacing`` for both inputs and require them on
the same grid. ``load_reference_organ_mask`` gets them there by ``Resized(spatial_size=<the
prediction's dims>, mode="nearest")`` — a corner-to-corner resample of the whole reference array,
**not** a center crop, and the reference's own spacing is discarded. So the two volumes' extents
are force-matched and the reference anatomy is implicitly rescaled by the ratio of their physical
FOVs. Over 40 valid scans against the wan_mask_v2 grid (512x512x253 @ 0.75/0.75/1.3 = 384 x 384 x
328.9 mm), reference FOV ranged 297-500 mm in-plane and 250-401 mm in Z, i.e. a median rescale of
6.6 % in-plane / 8.8 % in Z, up to 31.6 %. Consequences: Dice is a dimensionless overlap on the
shared grid and is unaffected; ``hd95_mm``/``hd_mm``/``assd_mm`` are in *prediction-grid* mm and
carry that per-sample scale error, so read them as a model-vs-model comparison at one fixed
generation spacing, not as absolute anatomical millimetres.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import torch
from monai.metrics import compute_dice

from src.data.organ_groups import GROUP_NAMES, NUM_GROUPS, apply_grouping
from src.eval.analysis.segment import segment_volume

log = logging.getLogger(__name__)

_TS_SEG_ROOT = Path("/workspace/datasets/datasets/CT-RATE/dataset/ts_seg/ts_total")
_ORGAN_NAMES = [
    GROUP_NAMES[i] for i in range(1, NUM_GROUPS + 1)
]  # ["lung","heart","aorta","esophagus"]

# Column prefixes for the surface-distance family, all three derived from ONE distance pass:
#   hd95_mm  — 95th percentile, outlier-robust, the medical-segmentation convention (headline)
#   hd_mm    — plain max distance; a stray segmented voxel far from the anatomy blows it up, which
#              is exactly what makes it useful as a "something is grossly wrong" flag
#   assd_mm  — symmetric average surface distance; the whole-boundary average, an axis neither
#              Dice (volume overlap) nor HD95 (local worst case) covers
SURFACE_METRICS: tuple[str, ...] = ("hd95_mm", "hd_mm", "assd_mm")


def _ts_seg_path(scan_id: str, split: str = "valid_fixed") -> Path:
    """``valid_1000_a_1`` -> ``ts_total/valid_fixed/valid_1000/valid_1000_a/valid_1000_a_1.nii.gz``."""
    parts = scan_id.split("_")
    patient = "_".join(parts[:2])
    series = "_".join(parts[:3])
    return _TS_SEG_ROOT / split / patient / series / f"{scan_id}.nii.gz"


def load_reference_organ_mask(
    scan_id: str, target_shape_xyz: tuple[int, int, int], split: str = "valid_fixed"
) -> np.ndarray | None:
    """Load ``scan_id``'s ts_seg mask, resized (NEAREST, no reorientation) to a target grid.

    Args:
        scan_id: the scan whose ts_seg mask to load (target's own id for ``dice_to_gt_mask``;
            the conditioning-mask source id for ``dice_to_input_mask``).
        target_shape_xyz: the prediction's own grid shape ``(X, Y, Z)`` — mask and prediction
            must be compared voxel-index-wise on this grid (see module docstring; no physical
            registration is possible or meaningful here).
        split: CT-RATE split subdirectory under ``ts_seg/ts_total``.

    Returns:
        Grouped organ mask ``(X, Y, Z)`` with values ``{0..4}`` (0=background,
        1..4=``NUM_GROUPS`` per :data:`src.data.organ_groups.GROUP_NAMES`), or ``None`` if the
        ts_seg file does not exist.
    """
    import monai.transforms as T

    ts_path = _ts_seg_path(scan_id, split)
    if not ts_path.is_file():
        return None

    transform = T.Compose(
        [
            T.LoadImaged(keys="mask"),
            T.EnsureChannelFirstd(keys="mask"),
            # NOTE: no Orientationd here — see module docstring.
            T.EnsureTyped(keys="mask", dtype=torch.float32),
            T.Resized(keys="mask", spatial_size=target_shape_xyz, mode="nearest"),
        ]
    )
    data = transform({"mask": str(ts_path)})
    arr = data["mask"][0].numpy()  # (X, Y, Z)
    return apply_grouping(np.round(arr).astype(np.int64))


def _dice_per_organ(
    pred_grouped: np.ndarray, ref_grouped: np.ndarray
) -> dict[str, float]:
    """Per-organ Dice with the plan's empty-mask policy.

    ``compute_dice(..., ignore_empty=False)`` matches our policy for the one-empty cases
    (GT-only-empty or pred-only-empty -> 0, verified 2026-07-27) but gives 1 — not our desired
    NaN — for both-empty organs; those are overridden to NaN explicitly below.
    """
    pred_t = torch.from_numpy(pred_grouped)[None, None].float()
    ref_t = torch.from_numpy(ref_grouped)[None, None].float()
    dice = compute_dice(
        pred_t,
        ref_t,
        include_background=False,
        num_classes=NUM_GROUPS + 1,
        ignore_empty=False,
    )[0]  # (NUM_GROUPS,)

    out: dict[str, float] = {}
    valid: list[float] = []
    for i, name in enumerate(_ORGAN_NAMES):
        organ_id = i + 1
        both_empty = (pred_grouped == organ_id).sum() == 0 and (
            ref_grouped == organ_id
        ).sum() == 0
        value = float("nan") if both_empty else float(dice[i])
        out[name] = value
        if not math.isnan(value):
            valid.append(value)
    out["_mean"] = float(np.mean(valid)) if valid else float("nan")
    return out


def _surface_metrics_per_organ(
    pred_grouped: np.ndarray,
    ref_grouped: np.ndarray,
    spacing_xyz: tuple[float, float, float],
) -> dict[str, dict]:
    """Per-organ HD95 / HD / ASSD (mm) from ONE surface-distance pass per organ.

    MONAI's ``compute_hausdorff_distance`` and ``compute_average_surface_distance`` both start
    from the same ``get_edge_surface_distance(..., symmetric=True)`` call and differ only in how
    they reduce the two directed distance tensors, so calling the public functions once per
    metric would recompute that pass three times (measured ~7 s each at 480x480x256 for all four
    organs). This helper runs it once and applies all three upstream reductions.

    **DUPLICATION INTENTIONAL** — the lines below are copied byte-for-byte (indentation included)
    from MONAI 1.4.0: ``monai/metrics/hausdorff_distance.py:188-198`` (the shared
    ``get_edge_surface_distance`` call + the HD reduction) and
    ``monai/metrics/surface_distance.py:183-184`` (the ASSD reduction; its identical call is at
    :175-182). Every deviation carries a ``MODIFIED`` comment naming the original expression.
    Numeric equivalence to both public functions is enforced by
    ``tests/eval_analysis/test_surface_metrics.py::test_matches_monai_public_api``.

    Empty-mask handling is MONAI's own except for one deliberate deviation, marked ``OUR POLICY``
    in the body: an organ present on exactly one side yields ``+inf`` from upstream HD/ASSD (but
    NaN from HD95), and we pin all three to NaN so a single undefined organ cannot turn every
    downstream mean into inf.

    Args:
        pred_grouped: segmentation of the generated volume, ``(X, Y, Z)`` values ``{0..4}``.
        ref_grouped: reference organ mask on the same grid, ``(X, Y, Z)`` values ``{0..4}``.
        spacing_xyz: voxel size in mm, matching the ``(X, Y, Z)`` array axes.

    Returns:
        ``{metric: {organ: mm, ..., "_mean": mm}}`` for each of :data:`SURFACE_METRICS`, plus a
        ``"_scored"`` entry ``{"n": int, "missing": str}`` recording how many organs ``_mean``
        actually averaged and which ones it could not score. ``_mean`` averages the non-NaN
        organs, so **its denominator varies between samples and between models**: a model that
        fails to generate an organ has it dropped (undefined distance) where Dice would score it
        0. Without ``_scored`` a 3-organ mean and a 4-organ mean are indistinguishable in the
        aggregate, and omitting the hardest organ silently flatters the model. ``missing`` names
        the side that was empty (``"esophagus:pred"``) so the common case — the model generated
        nothing there — is visible directly in ``per_sample.csv``.
    """
    # Imported (not copied) so a MONAI upgrade that moves either symbol fails loudly. Kept inline,
    # like this module's other heavy imports, so that merely importing seg_metrics — which
    # subgroup.py and scripts/compare_runs.py do just to reach SURFACE_METRICS — does not depend
    # on a private MONAI path.
    from monai.metrics.hausdorff_distance import _compute_percentile_hausdorff_distance
    from monai.metrics.utils import get_edge_surface_distance

    out: dict[str, dict] = {m: {} for m in SURFACE_METRICS}
    valid: dict[str, list[float]] = {m: [] for m in SURFACE_METRICS}
    missing: list[str] = []

    for i, name in enumerate(_ORGAN_NAMES):
        organ_id = i + 1
        p_bin = torch.from_numpy(
            (pred_grouped == organ_id).astype(np.float32)
        )  # (X, Y, Z)
        r_bin = torch.from_numpy(
            (ref_grouped == organ_id).astype(np.float32)
        )  # (X, Y, Z)
        p_empty, r_empty = not bool(p_bin.any()), not bool(r_bin.any())

        # --- MONAI verbatim: the identical call both public functions make (one per (b, c)) ---
        _, distances, _ = get_edge_surface_distance(
            p_bin,  # MODIFIED: MONAI `y_pred[b, c]` — we loop organs, not (batch, channel)
            r_bin,  # MODIFIED: MONAI `y[b, c]`
            distance_metric="euclidean",  # MODIFIED: MONAI `distance_metric` (its own default)
            spacing=spacing_xyz,  # MODIFIED: MONAI `spacing_list[b]` (identical for a single image)
            symmetric=True,  # MODIFIED: MONAI `symmetric=not directed` (HD) / `symmetric` (ASSD)
            class_index=organ_id,  # MODIFIED: MONAI `c`
        )

        # The formatter would reflow the lines below (they exceed our line length); the
        # off/on pair keeps them line-for-line diffable against MONAI's source.
        # fmt: off
        # --- MONAI verbatim: hausdorff_distance.py, run twice for the two percentiles instead of
        #     MONAI's single `percentile` argument (MODIFIED: literal 95 / None in place of it) ---
        percentile_distances = [_compute_percentile_hausdorff_distance(d, 95) for d in distances]
        max_distance = torch.max(torch.stack(percentile_distances))
        hd95_value = max_distance  # MODIFIED: MONAI `hd[b, c] = max_distance`

        percentile_distances = [_compute_percentile_hausdorff_distance(d, None) for d in distances]
        max_distance = torch.max(torch.stack(percentile_distances))
        hd_value = max_distance  # MODIFIED: MONAI `hd[b, c] = max_distance`

        # --- MONAI verbatim: surface_distance.py ---
        surface_distance = torch.cat(distances)
        # MODIFIED: MONAI `asd[b, c] = ...`; the right-hand side is unchanged.
        assd_value = torch.tensor(np.nan) if surface_distance.shape == (0,) else surface_distance.mean()
        # fmt: on

        for metric, tensor_value in (
            ("hd95_mm", hd95_value),
            ("hd_mm", hd_value),
            ("assd_mm", assd_value),
        ):
            value = float(tensor_value)
            # OUR POLICY (not MONAI's): when exactly one side of an organ is empty, upstream is
            # self-inconsistent — HD95 comes out NaN (quantile of an all-inf tensor is NaN) while
            # HD and ASSD come out +inf (verified against the public API 2026-08-01). The distance
            # is undefined either way, and a single inf would turn every downstream mean
            # (`_mean` here, subgroup means, SUMMARY.md) into inf, so all three are pinned to NaN.
            # inf can only arise from that empty-side case; genuine surface distances are finite.
            if not math.isfinite(value):
                value = float("nan")
            out[metric][name] = value
            if not math.isnan(value):
                valid[metric].append(value)

        # All three metrics go NaN under exactly the same condition (an empty side — pinned by
        # test_one_empty_side_is_nan_for_every_metric), so one unscored-organ record covers them.
        if math.isnan(out["hd95_mm"][name]):
            side = (
                "both"
                if (p_empty and r_empty)
                else "pred"
                if p_empty
                else "ref"
                if r_empty
                else "unknown"  # NaN with both sides populated — record it rather than mislabel
            )
            missing.append(f"{name}:{side}")

    for metric in SURFACE_METRICS:
        out[metric]["_mean"] = (
            float(np.mean(valid[metric])) if valid[metric] else float("nan")
        )
    out["_scored"] = {
        "n": len(_ORGAN_NAMES) - len(missing),
        "missing": ";".join(missing),
    }
    return out


def compute_seg_metrics(
    gen_mha_path: str | Path,
    target_id: str,
    cond_mask_source_id: str | None,
    backend: str = "totalseg",
    compute_hd95: bool = True,
) -> dict:
    """Segment a generated volume and compute Dice + surface distances to both reference masks.

    Args:
        gen_mha_path: path to the generated ``.mha``.
        target_id: scan id whose own ts_seg is the ``dice_to_gt_mask`` / ``hd95_mm_to_gt_mask``
            reference (the target patient's real anatomy).
        cond_mask_source_id: scan id whose ts_seg was the model's actual input conditioning mask
            (``dice_to_input_mask`` / ``hd95_mm_to_input_mask`` reference); ``None`` if this
            sample has no designated input mask (non-mask model, or ``condition=null``).
        backend: segmentation backend, see :func:`src.eval.analysis.segment.segment_volume`.
        compute_hd95: also compute the surface-distance family — ``hd95_mm_*``, ``hd_mm_*`` and
            ``assd_mm_*`` (:data:`SURFACE_METRICS`). No extra segmentation cost (same label
            maps) and all three share one distance pass, so the three cost what HD95 alone did.

    Returns:
        dict with keys ``dice_to_gt_mask``, ``dice_to_gt_mask_<organ>`` (x4), and the same for
        ``dice_to_input_mask`` and for every :data:`SURFACE_METRICS` entry against both
        references (all NaN if not applicable/available); ``n_organs_scored_to_<ref>`` +
        ``organs_missing_to_<ref>``, the varying denominator behind every surface-distance mean
        (see :func:`_surface_metrics_per_organ`); plus ``status``
        (``"ok"``/``"partial"``/``"failed"``) and ``failure_reason``.
    """
    _nan_keys = [
        f"{metric}_to_{ref}{suffix}"
        for metric in ("dice", *SURFACE_METRICS)
        for ref in ("gt_mask", "input_mask")
        for suffix in ("", *(f"_{o}" for o in _ORGAN_NAMES))
    ] + [
        f"{key}_to_{ref}"
        for key in ("n_organs_scored", "organs_missing")
        for ref in ("gt_mask", "input_mask")
    ]
    result: dict = dict.fromkeys(_nan_keys, float("nan"))

    try:
        pred_label = segment_volume(gen_mha_path, backend=backend)
    except Exception:
        log.exception("Segmentation failed for %s", gen_mha_path)
        result["status"] = "failed"
        result["failure_reason"] = "seg_failed"
        return result

    pred_grouped = apply_grouping(pred_label)
    target_shape = pred_grouped.shape

    import SimpleITK as sitk

    spacing_xyz = sitk.ReadImage(str(gen_mha_path)).GetSpacing()

    reasons: list[str] = []
    any_ref = False

    gt_mask = load_reference_organ_mask(target_id, target_shape)
    if gt_mask is None:
        reasons.append("gt_mask_missing")
    else:
        any_ref = True
        dice = _dice_per_organ(pred_grouped, gt_mask)
        result["dice_to_gt_mask"] = dice["_mean"]
        for name in _ORGAN_NAMES:
            result[f"dice_to_gt_mask_{name}"] = dice[name]
        if compute_hd95:
            surf = _surface_metrics_per_organ(pred_grouped, gt_mask, spacing_xyz)
            for metric in SURFACE_METRICS:
                result[f"{metric}_to_gt_mask"] = surf[metric]["_mean"]
                for name in _ORGAN_NAMES:
                    result[f"{metric}_to_gt_mask_{name}"] = surf[metric][name]
            result["n_organs_scored_to_gt_mask"] = surf["_scored"]["n"]
            result["organs_missing_to_gt_mask"] = surf["_scored"]["missing"]

    if cond_mask_source_id is None:  # conditioning에 사용된 진짜 마스크
        reasons.append("no_input_mask_designated")
    else:
        input_mask = load_reference_organ_mask(cond_mask_source_id, target_shape)
        if input_mask is None:
            reasons.append("input_mask_missing")
        else:
            any_ref = True
            dice = _dice_per_organ(pred_grouped, input_mask)
            result["dice_to_input_mask"] = dice["_mean"]
            for name in _ORGAN_NAMES:
                result[f"dice_to_input_mask_{name}"] = dice[name]
            if compute_hd95:
                surf = _surface_metrics_per_organ(pred_grouped, input_mask, spacing_xyz)
                for metric in SURFACE_METRICS:
                    result[f"{metric}_to_input_mask"] = surf[metric]["_mean"]
                    for name in _ORGAN_NAMES:
                        result[f"{metric}_to_input_mask_{name}"] = surf[metric][name]
                result["n_organs_scored_to_input_mask"] = surf["_scored"]["n"]
                result["organs_missing_to_input_mask"] = surf["_scored"]["missing"]

    if not any_ref:
        result["status"] = "partial"
    elif reasons:
        result["status"] = "partial"
    else:
        result["status"] = "ok"
    result["failure_reason"] = ";".join(reasons) if reasons else ""
    return result


__all__ = ["SURFACE_METRICS", "load_reference_organ_mask", "compute_seg_metrics"]
