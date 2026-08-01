"""Subgroup-level FID / FVD — set-level metrics (no per-sample decomposition exists), computed by
re-stacking the ALREADY-CACHED per-volume features from a completed ``fid_2p5d`` run
(``out_dir/fid_features/{gt,pred}/``). No feature re-extraction, no re-generation — only the
final Frechet-distance aggregation is redone per subgroup (GPU when available, via
``_fid_refstats.py``'s ``*_fast`` helpers, with an automatic CPU fallback — see that module's
comment for why this was needed even after GT-restacking was eliminated).

"Computable" != "statistically valid" (plan §3.7): every row reports ``real_n``/``gen_n``/
``real_patients`` and a ``below_threshold`` flag (a "small sample" marker only — it does NOT
claim n >= threshold makes the estimate valid; user-confirmed 2026-07-27, threshold=100).
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.analysis.labels import LABEL_NAMES, load_label_csv, patient_id_of
from src.eval.analysis.subgroup_config import SubgroupConfig

log = logging.getLogger(__name__)

# axes given FVD_CTCLIP too (bounded cost — full CT-CLIP re-encode per axis; see module docstring)
_FVD_AXES = {"overall", "normal", "disease"}


def _build_feature_index(feat_dir: Path) -> dict[str, Path]:
    """One-time ``{scan_id: path}`` index of a cached-feature directory.

    Perf fix (live n=50 test, 2026-07-27): the previous per-id lookup did
    ``feat_dir.glob(f"{scan_id}.*")`` — which re-lists the ENTIRE directory (thousands of GT
    feature files) for every single id — making every axis's cost O(n_ids_in_axis * dir_size)
    instead of O(dir_size). That glob-per-id pattern was the actual dominant cost of a live
    n=50 run (~20 minutes for all 29 axes; even the smallest axis, 226 GT ids, took ~29s alone).
    Building this index once per directory and doing O(1) dict lookups fixes it directly.
    """
    return {p.stem: p for p in feat_dir.iterdir() if p.is_file()}


def _fid_subset(
    gt_index: dict[str, Path],
    pred_index: dict[str, Path],
    gt_ids: set[str],
    pred_ids: set[str],
    ref_stats: dict | None = None,
    axis_name: str | None = None,
    axis_refstats_cache_dir: Path | None = None,
) -> dict:
    """3-plane FID for one (gt_ids, pred_ids) subset, from pre-indexed cached features.

    Two independent fast paths are tried, in order, before falling back to a full re-stack —
    both use ``fid_from_ref_stats_files_fast`` (GPU Cholesky+eigvalsh Frechet, CPU fallback;
    same mu/Sigma formula as the original ``fid_from_ref_stats``, which is left untouched). The
    GT side is a cached (mu, Sigma) in both cases; the pred side is ALWAYS restacked from just
    ``pred_ids`` (typically small — a subgroup axis's positive predictions, e.g. 3-50 files),
    never the whole cached pred pool:

    1. **Per-axis precomputed ref-stats** (``axis_refstats_cache_dir``, any axis): if
       ``scripts/precompute_subgroup_fid_refstats.py`` has been run for this GT set + subgroup
       config, EVERY axis (not just "overall") gets this speedup. Verified by an EXACT id-set
       match against the cache's ``.ids.json`` sidecar (not a count — a count-only check is
       fragile, see ``_build_gt_axes``'s docstring for the bug that motivated this).
    2. **Main "overall" ref-stats** (``ref_stats``, "overall" axis only): the always-available
       fast path using the main FID pipeline's own full-census (mu, Sigma)
       (``ctgen.py``/``_fid_refstats.py``) — works even if the precompute script above was
       never run, so "overall" stays fast out of the box (pre-existing behavior, unchanged).

    Restacking the full census from scratch (loading + covariance over hundreds of thousands of
    cached slice-features) was the actual dominant cost in a live n=50 test, not file lookup
    alone — these fast paths are what avoid it. A second perf finding (2026-07-27, GPU Frechet
    verification): both branches originally ALSO required ``pred_ids == set(pred_index.keys())``
    — true only for "overall" (every other axis's pred subset is smaller than the full cached
    pool), so 28 of 29 axes silently fell through to the full-restack branch regardless of
    whether a GT cache existed, making the GT-side caching effort (and, at first, the GPU
    Frechet swap layered on top of it) far less effective than measured in isolation — a live
    re-run after removing the requirement is the actual fix; restacking just the (small) pred
    subset here is cheap by construction. The full-restack branch (both sides uncached) uses
    ``fid_from_file_lists_fast`` (same GPU Frechet, CPU fallback) rather than ``FIDMetric`` (CPU
    scipy sqrtm) for the same reason. See ``_fid_refstats.py``'s module comment for the
    fast-Frechet design.
    """
    pred_files = [pred_index[sid] for sid in pred_ids if sid in pred_index]

    if pred_files:
        if axis_name is not None and axis_refstats_cache_dir is not None:
            from src.eval.analysis.subgroup_refstats import load_cached_axis_refstats
            from src.eval.tasks._fid_refstats import fid_from_ref_stats_files_fast

            cached = load_cached_axis_refstats(
                axis_refstats_cache_dir, axis_name, gt_ids
            )
            if cached is not None:
                log.info(
                    "Subgroup FID: using precomputed per-axis ref-stats for %r "
                    "(gt_n=%d, pred_n=%d).",
                    axis_name,
                    len(gt_ids),
                    len(pred_files),
                )
                fids = fid_from_ref_stats_files_fast(pred_files, cached)
                return {
                    "real_n": cached["n_volumes"],
                    "gen_n": len(pred_files),
                    "real_patients": len({patient_id_of(sid) for sid in gt_ids}),
                    **fids,
                }

        if (
            ref_stats is not None
            and ref_stats["n_volumes"] == len(gt_index)
            and gt_ids == set(gt_index.keys())
        ):
            from src.eval.tasks._fid_refstats import fid_from_ref_stats_files_fast

            log.info(
                "Subgroup FID: using precomputed GT ref-stats fast path "
                "(gt_n=%d, pred_n=%d).",
                len(gt_ids),
                len(pred_files),
            )
            fids = fid_from_ref_stats_files_fast(pred_files, ref_stats)
            return {
                "real_n": ref_stats["n_volumes"],
                "gen_n": len(pred_files),
                "real_patients": len({patient_id_of(sid) for sid in gt_ids}),
                **fids,
            }

    gt_files = [gt_index[sid] for sid in gt_ids if sid in gt_index]

    result = {
        "real_n": len(gt_files),
        "gen_n": len(pred_files),
        "real_patients": len({patient_id_of(f.stem) for f in gt_files}),
    }
    if not gt_files or not pred_files:
        result.update(
            {
                k: float("nan")
                for k in ("FID_2p5D_Avg", "FID_2p5D_XY", "FID_2p5D_YZ", "FID_2p5D_XZ")
            }
        )
        return result

    from src.eval.tasks._fid_refstats import fid_from_file_lists_fast

    fids = fid_from_file_lists_fast(gt_files, pred_files)
    result.update(fids)
    return result


def _fvd_ctclip_subset(
    pred_dir: Path, gt_dir: Path, pred_ids: set[str], tmp_root: Path
) -> float:
    """FVD_CTCLIP over a pred subset, by symlinking the subset into a temp dir and calling the
    unchanged upstream-faithful ``compute_fvd_ctclip`` (no modification to that module)."""
    from src.eval.tasks._fvd_ctclip import compute_fvd_ctclip

    subset_dir = tmp_root / "fvd_subset"
    if subset_dir.exists():
        shutil.rmtree(subset_dir)
    subset_dir.mkdir(parents=True)
    n_linked = 0
    for sid in pred_ids:
        src = pred_dir / f"{sid}.mha"
        if src.is_file():
            (subset_dir / src.name).symlink_to(src.resolve())
            n_linked += 1
    try:
        if n_linked == 0:
            return float("nan")
        return compute_fvd_ctclip(subset_dir, gt_dir)
    finally:
        shutil.rmtree(subset_dir, ignore_errors=True)


def _build_gt_axes(
    gt_label_map: dict[str, np.ndarray],
    subgroup_cfg: SubgroupConfig,
    gt_index: dict[str, Path],
) -> list[tuple[str, set[str]]]:
    """(axis_name, gt_scan_ids) for overall/normal/disease/per-label/burden/cluster — the GT
    side only, independent of any particular eval run's predictions. Reused by both
    ``_build_axes`` (below, adds the pred side for a live run) and
    ``scripts/precompute_subgroup_fid_refstats.py`` (one-time GT ref-stats cache builder,
    since axis membership depends only on the dataset's labels, never on which model is
    being evaluated).

    ``gt_label_map`` is restricted to ids that actually have a cached GT feature (``gt_index``):
    the label CSV (3039 rows) is a superset of the clean GT census with cached features (~3001,
    ``no_chest``/unencodable scans excluded) — without this filter, the "overall" axis's gt_ids
    count (3039) never equals ``ref_stats["n_volumes"]`` (3001), so the ref-stats fast path in
    ``_fid_subset`` silently never engages (perf finding, live n=50 test, 2026-07-27).
    """
    gt_label_map = {sid: vec for sid, vec in gt_label_map.items() if sid in gt_index}
    name_to_idx = {name: i for i, name in enumerate(LABEL_NAMES)}
    axes: list[tuple[str, set[str]]] = []

    all_gt_ids = set(gt_label_map.keys())
    axes.append(("overall", all_gt_ids))

    normal_gt = {sid for sid, vec in gt_label_map.items() if vec.sum() == 0}
    axes.append(("normal", normal_gt))
    axes.append(("disease", all_gt_ids - normal_gt))

    for name in LABEL_NAMES:
        idx = name_to_idx[name]
        gt_ids = {sid for sid, vec in gt_label_map.items() if vec[idx] == 1}
        axes.append((f"label:{name}", gt_ids))

    for band in subgroup_cfg.label_burden_bands:
        lo, hi = subgroup_cfg.label_burden_bands[band]
        gt_ids = {sid for sid, vec in gt_label_map.items() if lo <= vec.sum() <= hi}
        axes.append((f"burden:{band}", gt_ids))

    for cluster_name, cluster_labels in subgroup_cfg.organ_clusters.items():
        idxs = [name_to_idx[label] for label in cluster_labels]
        gt_ids = {sid for sid, vec in gt_label_map.items() if vec[idxs].any()}
        axes.append((f"cluster:{cluster_name}", gt_ids))

    return axes


def _build_axes(
    df: pd.DataFrame,
    gt_label_map: dict[str, np.ndarray],
    subgroup_cfg: SubgroupConfig,
    gt_index: dict[str, Path],
) -> list[tuple[str, set[str], set[str]]]:
    """(axis_name, gt_scan_ids, pred_target_ids) for overall/normal/disease/per-label/burden/cluster."""
    gt_axes = _build_gt_axes(gt_label_map, subgroup_cfg, gt_index)

    pred_by_axis: dict[str, set[str]] = {
        "overall": set(df["target_id"].astype(str)),
        "normal": set(df.loc[df["label_class"] == "normal", "target_id"].astype(str)),
        "disease": set(df.loc[df["label_class"] == "disease", "target_id"].astype(str)),
    }
    for name in LABEL_NAMES:
        col = f"label_{name}"
        pred_by_axis[f"label:{name}"] = (
            set(df.loc[df[col] == 1, "target_id"].astype(str)) if col in df else set()
        )
    for band in subgroup_cfg.label_burden_bands:
        pred_by_axis[f"burden:{band}"] = set(
            df.loc[df["burden_band"] == band, "target_id"].astype(str)
        )
    for cluster_name in subgroup_cfg.organ_clusters:
        col = f"cluster_{cluster_name}"
        pred_by_axis[f"cluster:{cluster_name}"] = (
            set(df.loc[df[col] == True, "target_id"].astype(str))  # noqa: E712
            if col in df
            else set()
        )

    return [(name, gt_ids, pred_by_axis[name]) for name, gt_ids in gt_axes]


def run(
    run_out_dir: str | Path,
    per_sample_csv: str | Path,
    subgroup_cfg: SubgroupConfig,
    analysis_out_dir: str | Path,
    gt_dir: str | Path | None = None,
    compute_fvd: bool = False,
    label_split: str = "valid",
    fid_profile: str = "research",
) -> pd.DataFrame:
    """Compute subgroup FID (+ optional bounded-cost FVD) for all axes, from cached features.

    Args:
        run_out_dir: the eval run's top-level ``out_dir`` (must already contain
            ``fid_features/{gt,pred}/`` from a completed ``task.metrics.fid_2p5d=true`` run).
        per_sample_csv: path to ``analysis/per_sample.csv`` (pred-side subgroup membership).
        subgroup_cfg: loaded subgroup definitions.
        analysis_out_dir: ``analysis/setlevel/`` directory to write the result CSV into.
        gt_dir: the GT ``.mha`` directory this run scored against (``task.gt_dir``, e.g. the
            shared ``_valid_full_3001``) — required when ``compute_fvd=True`` (FVD re-reads raw
            volumes, unlike FID which only needs the cached features). Ignored otherwise.
        compute_fvd: also compute FVD_CTCLIP for the overall/normal/disease axes only (bounded
            cost — a full CT-CLIP re-encode per axis; per-label/cluster/burden FVD is left out
            of this bound intentionally, logged below, not silently dropped).
        label_split: label CSV for the GT (real) side — ``"valid"`` (the full clean census).
        fid_profile: which FID profile's cached features to reuse. Defaults to ``"research"``
            (radimagenet_resnet50), the feature space every per-axis ref-stats npz on disk was
            built with; it must match the profile the run was scored with, since the two write
            different feature dimensionalities into different sub-directories.

    Returns:
        DataFrame, one row per axis, with FID_2p5D_{Avg,XY,YZ,XZ}, real_n, gen_n, real_patients,
        below_threshold, and (if ``compute_fvd``) ``FVD_CTCLIP`` for the 3 bounded axes.
    """
    from src.eval.tasks.ctgen import _FID_PROFILES  # noqa: PLC0415

    run_out_dir = Path(run_out_dir)
    if compute_fvd and gt_dir is None:
        raise ValueError("gt_dir is required when compute_fvd=True")
    if fid_profile not in _FID_PROFILES:
        raise ValueError(
            f"unknown fid_profile {fid_profile!r} — choose from {sorted(_FID_PROFILES)}"
        )
    fid_model = _FID_PROFILES[fid_profile]["model"]
    features_dir = run_out_dir / _FID_PROFILES[fid_profile]["feat_subdir"]
    if not features_dir.is_dir():
        raise FileNotFoundError(
            f"{features_dir} not found — subgroup_setlevel reuses the cached per-volume "
            f"features of the '{fid_profile}' FID profile. Either score this run with "
            f"task.fid_profile={fid_profile}, or point the analysis at the profile this run "
            "actually used via task.subgroup_fid_profile."
        )

    # Build each directory's {scan_id: path} index ONCE (perf fix, live n=50 test — see
    # _build_feature_index docstring) rather than per-id glob lookups repeated across 29 axes.
    gt_index = _build_feature_index(features_dir / "gt")
    pred_index = _build_feature_index(features_dir / "pred")

    df = pd.read_csv(per_sample_csv)
    gt_label_map = load_label_csv(label_split)
    axes = _build_axes(df, gt_label_map, subgroup_cfg, gt_index)

    # Load the precomputed GT (mu, Sigma) once, if available, for the "overall"-axis fast path
    # in _fid_subset (perf finding, live n=50 test: restacking ~3001 GT files per axis was the
    # dominant cost). Best-effort: any failure just means every axis falls back to a full
    # restack — never a correctness risk, only a possible slowdown.
    ref_stats = None
    axis_refstats_cache_dir = None
    if gt_dir is not None:
        try:
            from src.eval.tasks._fid_refstats import load_ref_stats, refstats_path_for
            from src.eval.tasks.ctgen import _shared_gt_feat_dir

            npz_path = refstats_path_for(_shared_gt_feat_dir(Path(gt_dir), fid_model))
            if npz_path.is_file():
                ref_stats = load_ref_stats(npz_path)
        except Exception:
            log.exception(
                "Could not load precomputed GT ref-stats — subgroup FID will restack GT "
                "features for every axis (slower, still correct)."
            )

        # Per-axis ref-stats cache (scripts/precompute_subgroup_fid_refstats.py) — optional,
        # speeds up every axis (not just "overall") when it has been built. A missing/partial
        # cache dir is not an error: load_cached_axis_refstats() reports a per-axis miss and
        # _fid_subset falls back to the (already-verified) full re-stack for that axis.
        from src.eval.analysis.subgroup_refstats import subgroup_refstats_dir

        axis_refstats_cache_dir = subgroup_refstats_dir(gt_dir, model=fid_model)

    if compute_fvd:
        log.info(
            "FVD_CTCLIP computed only for axes %s (bounded cost); other axes get FID only.",
            sorted(_FVD_AXES),
        )

    # Per-axis error isolation (code review finding, 2026-07-27): a failure on one axis (e.g. a
    # GPU OOM re-encoding the FVD subset) must not discard the already-computed FID rows for
    # every other axis. Matches CTGenEvaluator._run_fvd_ctclip's own except -> NaN convention
    # (src/eval/tasks/ctgen.py) rather than letting one bad axis abort the whole function.
    rows = []
    for axis_name, gt_ids, pred_ids in axes:
        try:
            row = _fid_subset(
                gt_index,
                pred_index,
                gt_ids,
                pred_ids,
                ref_stats=ref_stats,
                axis_name=axis_name,
                axis_refstats_cache_dir=axis_refstats_cache_dir,
            )
        except Exception:
            log.exception(
                "Subgroup FID failed for axis %r — recording NaN row.", axis_name
            )
            row = {
                "real_n": len(gt_ids),
                "gen_n": len(pred_ids),
                "real_patients": 0,
                "FID_2p5D_Avg": float("nan"),
                "FID_2p5D_XY": float("nan"),
                "FID_2p5D_YZ": float("nan"),
                "FID_2p5D_XZ": float("nan"),
            }
        row["axis"] = axis_name
        row["below_threshold"] = row["gen_n"] < subgroup_cfg.subgroup_fid_small_n
        if compute_fvd and axis_name in _FVD_AXES:
            try:
                row["FVD_CTCLIP"] = _fvd_ctclip_subset(
                    run_out_dir / "predictions",
                    Path(gt_dir),
                    pred_ids,
                    Path(analysis_out_dir),
                )
            except Exception:
                log.exception(
                    "FVD_CTCLIP subset failed for axis %r — recording NaN.", axis_name
                )
                row["FVD_CTCLIP"] = float("nan")
        rows.append(row)

    result_df = pd.DataFrame(rows)
    analysis_out_dir = Path(analysis_out_dir)
    analysis_out_dir.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(analysis_out_dir / "setlevel_fid_fvd.csv", index=False)
    return result_df


__all__ = ["run"]
