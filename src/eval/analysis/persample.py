"""Build ``analysis/per_sample.csv`` — the atomic, 1-row-per-generated-sample table every
subgroup/summary output is derived from via groupby (never recomputed).

Primary key = ``sample_id`` (see plan §3.5). In a plain run (no mask-intervention manifest),
``sample_id == target_id == scan_id`` and this is exactly "1 scan = 1 row", fully
backward-compatible with the existing per-scan filename convention. A manifest (JSONL, one
mask-intervention sample per line) is consumed when present; label/GT joins always key on
``target_id``, input-mask joins key on ``cond_mask_source_id`` — see plan §3.9.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from src.eval.analysis import labels as labels_mod
from src.eval.analysis import seg_metrics
from src.eval.analysis.subgroup_config import SubgroupConfig

log = logging.getLogger(__name__)


_REQUIRED_MANIFEST_FIELDS = ("sample_id", "target_id")


def _is_safe_id(value: str) -> bool:
    """Reject ids that could enable path traversal once interpolated into file paths
    (e.g. ``pred_dir / f"{sample_id}.mha"``, ``ts_seg/.../{scan_id}.nii.gz``)."""
    return bool(value) and "/" not in value and "\\" not in value and ".." not in value


def _read_manifest(manifest_path: Path) -> dict[str, dict]:
    """Read a JSONL mask-intervention manifest into ``{sample_id: record}``.

    A hand-edited manifest is untrusted input: malformed lines (invalid JSON, missing required
    fields, or id values containing path-traversal characters) are skipped with a warning, not
    raised — the same skip+warn policy used for missing metric prerequisites elsewhere in this
    pipeline (plan §3.3), and a defensive guard against a manifest id like ``"../../etc/passwd"``
    reaching a file path (code review finding, 2026-07-27).
    """
    records: dict[str, dict] = {}
    for lineno, line in enumerate(
        Path(manifest_path).read_text().splitlines(), start=1
    ):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            log.warning(
                "Manifest %s line %d: invalid JSON — skipping.", manifest_path, lineno
            )
            continue

        missing = [f for f in _REQUIRED_MANIFEST_FIELDS if f not in rec]
        if missing:
            log.warning(
                "Manifest %s line %d: missing required field(s) %s — skipping.",
                manifest_path,
                lineno,
                missing,
            )
            continue

        id_fields = {
            "sample_id": rec["sample_id"],
            "target_id": rec["target_id"],
            "cond_mask_source_id": rec.get("cond_mask_source_id"),
        }
        unsafe = [
            name
            for name, value in id_fields.items()
            if value is not None and not _is_safe_id(value)
        ]
        if unsafe:
            log.warning(
                "Manifest %s line %d: unsafe id value(s) in %s — skipping.",
                manifest_path,
                lineno,
                unsafe,
            )
            continue

        records[rec["sample_id"]] = rec
    return records


def build_per_sample(
    scan_ids: list[str],
    pred_dir: str | Path,
    gt_dir: str | Path,
    subgroup_cfg: SubgroupConfig,
    label_split: str = "valid",
    clip_scores: dict[str, tuple[float, float]] | None = None,
    is_mask_model: bool = False,
    manifest_path: str | Path | None = None,
    seg_backend: str | None = None,
    compute_hd95: bool = True,
    seg_max_n: int | None = None,
) -> pd.DataFrame:
    """Build the per-sample table for one eval run.

    Args:
        scan_ids: the eval case scan ids for this run (plain-run "1 scan = 1 row" driver when
            no manifest is given).
        pred_dir: directory of generated ``.mha`` predictions (filename stem = ``sample_id``).
        gt_dir: directory of GT ``.mha`` (filename stem = ``target_id``).
        subgroup_cfg: loaded subgroup definitions (clusters, burden bands).
        label_split: which 18-label CSV to join against (``"valid"`` for the eval set).
        clip_scores: optional ``{sample_id: (clip_t2i, clip_i2i)}`` from
            :mod:`src.eval.analysis.clip_persample` (``None`` -> both columns NaN).
        is_mask_model: whether this model is mask-conditioned. In a plain run (no manifest),
            this decides whether ``cond_mask_source_id`` defaults to the sample's own id
            (mask model, ``condition="gt"``) or ``None`` (non-mask model, ``condition="none"``,
            so ``dice_to_input_mask``/``hd95_mm_to_input_mask`` are left NA per plan §3.9).
        manifest_path: optional JSONL mask-intervention manifest (plan §3.9); when given, it
            fully determines ``sample_id``/``target_id``/``condition``/``cond_mask_source_id``/
            ``seed``/``cfg_scale_*``/``source_label_*``/``label_overlap`` per row instead of the
            plain-run defaults.
        seg_backend: segmentation backend for Dice/HD95 (``None`` disables both — all-NaN,
            ``failure_reason="dice_disabled"``).
        compute_hd95: also compute the surface-distance family — ``hd95_mm``/``hd_mm``/``assd_mm``
            (:data:`src.eval.analysis.seg_metrics.SURFACE_METRICS`) — when ``seg_backend`` is set.
            Adds no extra segmentation cost (same label maps as Dice) and all three share one
            distance pass.
        seg_max_n: cap segmentation to the first N samples (``None`` = all); samples beyond the
            cap get ``failure_reason="seg_max_n_exceeded"``.

    Returns:
        A wide-form DataFrame, one row per generated sample, per the plan §3.5 schema.
    """
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)
    label_map = labels_mod.load_label_csv(label_split)
    manifest = _read_manifest(manifest_path) if manifest_path else None

    # Build the row-plan: list of (sample_id, target_id, condition, cond_mask_source_id, seed,
    # cfg_scale_text, cfg_scale_mask, run_id, source_labels_vec, label_overlap).
    row_plan: list[dict] = []
    if manifest is not None:
        for sample_id, rec in manifest.items():
            row_plan.append(rec)
    else:
        for scan_id in scan_ids:
            row_plan.append(
                {
                    "sample_id": scan_id,
                    "target_id": scan_id,
                    "condition": "gt" if is_mask_model else "none",
                    "cond_mask_source_id": scan_id if is_mask_model else None,
                    "seed": None,
                    "cfg_scale_text": None,
                    "cfg_scale_mask": None,
                    "run_id": None,
                    "source_labels": None,
                    "label_overlap": None,
                }
            )

    n_seg_done = 0
    rows: list[dict] = []
    for plan in row_plan:
        sample_id = plan["sample_id"]
        target_id = plan["target_id"]
        cond_mask_source_id = plan.get("cond_mask_source_id")

        gen_path = pred_dir / f"{sample_id}.mha"
        gt_path = gt_dir / f"{target_id}.mha"

        row: dict = {
            "sample_id": sample_id,
            "target_id": target_id,
            "scan_id": target_id,
            "patient_id": labels_mod.patient_id_of(target_id),
            "gen_path": str(gen_path),
            "gt_path": str(gt_path),
            "condition": plan.get("condition"),
            "cond_mask_source_id": cond_mask_source_id,
            "seed": plan.get("seed"),
            "cfg_scale_text": plan.get("cfg_scale_text"),
            "cfg_scale_mask": plan.get("cfg_scale_mask"),
            "run_id": plan.get("run_id"),
            "label_overlap": plan.get("label_overlap"),
        }

        failure_reasons: list[str] = []
        if not gen_path.is_file():
            failure_reasons.append("gen_path_missing")

        vec = label_map.get(target_id)
        if vec is None:
            failure_reasons.append("label_missing")
            row["label_burden"] = None
            row["label_class"] = None
            row["labels_missing"] = True
            for name in labels_mod.LABEL_NAMES:
                row[f"label_{name}"] = None
            for cluster_name in subgroup_cfg.organ_clusters:
                row[f"cluster_{cluster_name}"] = None
            row["burden_band"] = None
        else:
            burden = labels_mod.label_burden(vec)
            row["label_burden"] = burden
            row["label_class"] = labels_mod.label_class(vec)
            row["labels_missing"] = False
            for name, val in zip(labels_mod.LABEL_NAMES, vec):
                row[f"label_{name}"] = int(val)
            for cluster_name, is_member in labels_mod.cluster_membership(
                vec, subgroup_cfg.organ_clusters
            ).items():
                row[f"cluster_{cluster_name}"] = is_member
            row["burden_band"] = labels_mod.burden_band(
                burden, subgroup_cfg.label_burden_bands
            )

        source_labels_vec = plan.get("source_labels")
        for name in labels_mod.LABEL_NAMES:
            value = source_labels_vec.get(name) if source_labels_vec else None
            row[f"source_label_{name}"] = int(value) if value is not None else None

        if clip_scores is not None and sample_id in clip_scores:
            row["clip_t2i"], row["clip_i2i"] = clip_scores[sample_id]
        else:
            row["clip_t2i"] = float("nan")
            row["clip_i2i"] = float("nan")
            if clip_scores is not None:
                failure_reasons.append("clip_missing")

        if seg_backend is None:
            failure_reasons.append("dice_disabled")
        elif not gen_path.is_file():
            pass  # already flagged gen_path_missing; skip segmentation
        elif seg_max_n is not None and n_seg_done >= seg_max_n:
            failure_reasons.append("seg_max_n_exceeded")
        else:
            n_seg_done += 1
            seg_result = seg_metrics.compute_seg_metrics(
                gen_mha_path=gen_path,
                target_id=target_id,
                cond_mask_source_id=cond_mask_source_id,
                backend=seg_backend,
                compute_hd95=compute_hd95,
            )
            # drop seg_metrics' own status/failure_reason — persample computes its own below
            # (row-level status folds in gen_path/label/clip reasons too, not just segmentation)
            seg_result.pop("status")
            seg_failure_reason = seg_result.pop("failure_reason")
            row.update(seg_result)
            if seg_failure_reason:
                failure_reasons.append(seg_failure_reason)

        if not gen_path.is_file():
            row["status"] = "failed"
        elif failure_reasons:
            row["status"] = "partial"
        else:
            row["status"] = "ok"
        row["failure_reason"] = ";".join(failure_reasons)

        rows.append(row)

    return pd.DataFrame(rows)


__all__ = ["build_per_sample"]
