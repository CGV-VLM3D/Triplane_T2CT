"""Per-axis precomputed GT ref-stats cache — extends the "overall"-axis fast path (reusing the
main pipeline's precomputed (mu, Sigma) for the full 3001-census, see ``ctgen.py``/
``_fid_refstats.py``) to every other subgroup axis (per-label/burden/cluster/normal/disease).

Design (deliberately conservative, per explicit correctness requirement — 2026-07-27):
- Reuses ``compute_ref_stats``/``save_ref_stats``/``load_ref_stats``/``fid_from_ref_stats`` from
  ``_fid_refstats.py`` UNCHANGED — no new math, no new numerical algorithm to validate. Only the
  GT feature *subset* fed into ``compute_ref_stats`` is new (a per-axis symlink dir), and that
  function is already the one the main FID pipeline trusts for the full census.
- Every cached npz has a companion ``.ids.json`` sidecar listing the EXACT sorted scan_ids used
  to build it. A count-only staleness check was tried first and found fragile (the "overall"
  axis's gt_ids count didn't match the label-CSV-derived set until that bug was fixed — see
  ``_build_gt_axes``'s docstring); exact id-set comparison closes that whole class of bug: two
  different axes coincidentally having the same volume COUNT can never be confused for each
  other, and a subgroup_config edit that changes which labels feed an axis is detected as a
  cache miss (recompute), never a silent wrong-answer reuse.
- A missing/stale cache is not an error: callers fall back to the existing (already-verified)
  full re-stack path. This cache is a pure speed optimization, on by default only when
  ``scripts/precompute_subgroup_fid_refstats.py`` has actually been run at least once for a
  given GT set + subgroup config.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from pathlib import Path

from src.eval.tasks._fid_refstats import (
    compute_ref_stats,
    load_ref_stats,
    save_ref_stats,
)

log = logging.getLogger(__name__)


def _axis_key(axis_name: str) -> str:
    """Filesystem-safe key for an axis name, e.g. ``"label:Cardiomegaly"`` -> ``"label_Cardiomegaly"``."""
    return axis_name.replace(":", "_").replace(" ", "_")


def subgroup_refstats_dir(gt_dir: str | Path, model: str | None = None) -> Path:
    """Cache dir for per-axis GT ref-stats, sibling to the main shared GT feature cache
    (``ctgen.py::_shared_gt_feat_dir``) so it is keyed by the same GT-set + FID-params identity.

    Args:
        gt_dir: the GT ``.mha`` directory these ref-stats describe.
        model: FID feature-network name. None → the research profile, which is what every
            per-axis npz currently on disk was built with.
    """
    from src.eval.tasks.ctgen import _RESEARCH_FID_MODEL, _shared_gt_feat_dir

    shared_gt = _shared_gt_feat_dir(Path(gt_dir), model or _RESEARCH_FID_MODEL)
    return shared_gt.parent / f"{shared_gt.name}__subgroup_refstats"


def load_cached_axis_refstats(
    cache_dir: Path, axis_name: str, gt_ids: set[str]
) -> dict | None:
    """Load precomputed ref-stats for one axis IF the cached id-set exactly matches ``gt_ids``.

    Returns ``None`` on any cache miss (no cache built yet, or the id-set no longer matches —
    e.g. a subgroup_config edit) — callers must fall back to a full re-stack in that case, this
    function never raises for a routine miss.
    """
    key = _axis_key(axis_name)
    npz_path = cache_dir / f"{key}.npz"
    ids_path = cache_dir / f"{key}.ids.json"
    if not npz_path.is_file() or not ids_path.is_file():
        return None
    try:
        cached_ids = set(json.loads(ids_path.read_text()))
    except (json.JSONDecodeError, OSError):
        log.warning("Corrupt ids sidecar %s — treating as cache miss.", ids_path)
        return None
    if cached_ids != gt_ids:
        return None
    return load_ref_stats(npz_path)


def precompute_axis_refstats(
    gt_index: dict[str, Path], cache_dir: Path, axis_name: str, gt_ids: set[str]
) -> Path | None:
    """Compute + cache one axis's GT ref-stats (one-time cost; reused by every future eval run
    against the same GT set + subgroup config, exactly like the main "overall" refstats.npz).

    Returns the written npz path, or ``None`` if no GT feature files were found for ``gt_ids``
    (nothing to compute — logged, not raised).
    """
    key = _axis_key(axis_name)
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz_path = cache_dir / f"{key}.npz"
    ids_path = cache_dir / f"{key}.ids.json"

    tmpdir = tempfile.mkdtemp(dir=cache_dir)
    try:
        subset_dir = Path(tmpdir)
        n_linked = 0
        for sid in gt_ids:
            f = gt_index.get(sid)
            if f is not None:
                os.symlink(f.resolve(), subset_dir / f.name)
                n_linked += 1
        if n_linked == 0:
            log.warning(
                "No cached GT features found for axis %r — skipping.", axis_name
            )
            return None
        stats = compute_ref_stats(subset_dir)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    save_ref_stats(stats, npz_path)
    ids_path.write_text(json.dumps(sorted(gt_ids)))
    log.info(
        "Precomputed ref-stats for axis %r (n=%d) -> %s", axis_name, n_linked, npz_path
    )
    return npz_path


__all__ = [
    "subgroup_refstats_dir",
    "load_cached_axis_refstats",
    "precompute_axis_refstats",
]
