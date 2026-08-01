"""Generation-side reading of a mask-intervention manifest (docs/mask_intervention_manifest.md).

Two readers exist on purpose, with opposite failure policies:

* :func:`read_manifest_rows` (here, GENERATION side) **raises**. A row it cannot read is a volume
  that will never be generated, and the gap would only surface much later as a missing
  prediction — after hours of GPU time.
* ``src.eval.analysis.persample._read_manifest`` (SCORING side) warns and skips, because a
  hand-edited manifest must not abort the scoring of predictions that already exist.

⚠ SCOPE — a manifest run is diagnostic-only (CLIPScore-T2I, ``dice_to_input_mask``,
``dice_to_gt_mask``). FID/FVD assume "1 volume = 1 independent sample", which a manifest run
breaks by generating one target several times; ``scripts/run_eval.py`` refuses those metrics
outright when ``task.manifest`` is set.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

REQUIRED_FIELDS = (
    "sample_id",
    "target_id",
    "condition",
    "cond_mask_source_id",
    "seed",
)


def _is_safe_id(value: str) -> bool:
    """Reject ids that could escape a directory once interpolated into a path.

    Mirrors ``persample._is_safe_id`` (same rule, same reason: these ids become
    ``pred_dir / f"{sample_id}.mha"`` and ``ts_seg/.../{scan_id}.nii.gz``); kept as a separate
    copy so the scoring side's reader stays untouched by generation-side changes.
    """
    return bool(value) and "/" not in value and "\\" not in value and ".." not in value


def read_manifest_rows(manifest_path: str | Path) -> list[dict]:
    """Read a JSONL manifest into a list of rows, in file order.

    Args:
        manifest_path: JSONL written by ``scripts/build_mask_intervention_manifest.py``.

    Returns:
        One dict per row, in file order (generation order).

    Raises:
        ValueError: on invalid JSON, a missing required field, an unsafe id, a duplicate
            ``sample_id`` (two rows would write the same file, silently generating one fewer
            volume than requested), or an empty manifest.
    """
    manifest_path = Path(manifest_path)
    rows: list[dict] = []
    seen: set[str] = set()
    for lineno, line in enumerate(manifest_path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{manifest_path} line {lineno}: invalid JSON") from exc

        missing = [f for f in REQUIRED_FIELDS if f not in row]
        if missing:
            raise ValueError(
                f"{manifest_path} line {lineno}: missing required field(s) {missing}"
            )
        unsafe = [
            name
            for name in ("sample_id", "target_id", "cond_mask_source_id")
            if row[name] is not None and not _is_safe_id(row[name])
        ]
        if unsafe:
            raise ValueError(f"{manifest_path} line {lineno}: unsafe id in {unsafe}")
        if row["sample_id"] in seen:
            raise ValueError(
                f"{manifest_path} line {lineno}: duplicate sample_id "
                f"{row['sample_id']!r} — two rows would write the same file."
            )
        seen.add(row["sample_id"])
        rows.append(row)

    if not rows:
        raise ValueError(f"{manifest_path} contains no rows")
    log.info("Manifest %s: %d rows", manifest_path, len(rows))
    return rows


def check_generation_provenance(
    rows: list[dict], ckpt: str, cfg_text: float, cfg_mask: float
) -> None:
    """Abort when the generation command contradicts the manifest's recorded provenance.

    ``sample_id`` embeds ``s_m`` and the manifest records the checkpoint and both guidance
    scales, so generating those rows with a different ckpt/scale would produce files whose names
    claim settings the volumes do not have — the same class of silent drift the
    ``eval_ep…_sp…_cfg…`` directory convention exists to prevent (CLAUDE.md).
    """
    for field, actual in (
        ("ckpt", str(ckpt)),
        ("cfg_scale_text", float(cfg_text)),
        ("cfg_scale_mask", float(cfg_mask)),
    ):
        recorded = {row.get(field) for row in rows}
        if recorded != {actual}:
            raise ValueError(
                f"manifest {field}={sorted(recorded, key=str)} but this run passes {actual!r} — "
                "regenerate the manifest or fix the command; the sample_id/provenance would "
                "otherwise describe settings these volumes were not generated with."
            )


def build_gt_view(cases, gt_dir: str | Path, view_dir: str | Path) -> Path:
    """Symlink ``<view_dir>/<case.out_stem>.mha -> <gt_dir>/<case.scan_id>.mha`` for every case.

    The upstream CLIP scorer pairs a prediction with its GT purely by filename stem
    (``evaluate_clip.py::_match_gt``), and a manifest run's stems are ``sample_id``s, which do not
    exist in the GT directory. ``third_party/`` is read-only, so instead of patching the matcher
    we hand it a view of the same GT volumes under the names it will look for. Each link points at
    the case's TARGET scan, so CLIP-I2I still compares a generated volume against the scan it was
    generated for, whichever donor supplied the mask.

    Args:
        cases: the run's ``EvalCase`` list (``out_stem`` = generated stem, ``scan_id`` = target).
        gt_dir: real GT directory, keyed by target scan id.
        view_dir: directory of symlinks to create.

    Returns:
        ``view_dir`` (created; existing links are refreshed, so it is idempotent).
    """
    gt_dir = Path(gt_dir)
    view_dir = Path(view_dir)
    view_dir.mkdir(parents=True, exist_ok=True)

    missing: list[str] = []
    for case in cases:
        source = gt_dir / f"{case.scan_id}.mha"
        if not source.is_file():
            missing.append(case.scan_id)
            continue
        link = view_dir / f"{case.out_stem}.mha"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(source.resolve())

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} target GT volume(s) missing from {gt_dir} "
            f"(e.g. {sorted(set(missing))[:3]}) — CLIP would silently score against nothing."
        )
    log.info("GT view for CLIP: %d links in %s", len(cases), view_dir)
    return view_dir


__all__ = [
    "read_manifest_rows",
    "check_generation_provenance",
    "build_gt_view",
    "REQUIRED_FIELDS",
]
