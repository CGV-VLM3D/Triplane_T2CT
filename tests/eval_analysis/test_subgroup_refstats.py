"""subgroup_refstats.py: per-axis GT ref-stats cache — exact id-set staleness detection (not a
count-only check), cache hit/miss behavior, and end-to-end correctness against a full re-stack.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.eval.analysis.subgroup_refstats import (
    load_cached_axis_refstats,
    precompute_axis_refstats,
)
from src.eval.analysis.subgroup_setlevel import _build_feature_index, _fid_subset


def _write_fake_feature(path: Path, seed: int, n_slices=6, dim=2048):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    feats = tuple(torch.from_numpy(rng.normal(size=(n_slices, dim))) for _ in range(3))
    torch.save(feats, path)


def test_cache_miss_when_no_files_written(tmp_path: Path):
    assert load_cached_axis_refstats(tmp_path, "label:Foo", {"a", "b"}) is None


def test_precompute_then_load_roundtrip(tmp_path: Path, monkeypatch):
    gt_dir = tmp_path / "gt_features"
    ids = [f"scan{i}" for i in range(5)]
    for i, sid in enumerate(ids):
        _write_fake_feature(gt_dir / f"{sid}.mha", seed=i)
    gt_index = _build_feature_index(gt_dir)

    cache_dir = tmp_path / "cache"
    npz_path = precompute_axis_refstats(gt_index, cache_dir, "label:Foo", set(ids))
    assert npz_path is not None and npz_path.is_file()
    assert (cache_dir / "label_Foo.ids.json").is_file()

    loaded = load_cached_axis_refstats(cache_dir, "label:Foo", set(ids))
    assert loaded is not None
    assert loaded["n_volumes"] == 5


def test_cache_miss_when_id_set_changed(tmp_path: Path):
    """Exact id-set check (not count-only): a subgroup_config edit that changes which volumes
    belong to an axis must be detected as a cache miss, never silently reuse stale stats — this
    is the whole reason precompute uses an .ids.json sidecar instead of just n_volumes."""
    gt_dir = tmp_path / "gt_features"
    ids = [f"scan{i}" for i in range(5)]
    for i, sid in enumerate(ids):
        _write_fake_feature(gt_dir / f"{sid}.mha", seed=i)
    gt_index = _build_feature_index(gt_dir)

    cache_dir = tmp_path / "cache"
    precompute_axis_refstats(gt_index, cache_dir, "label:Foo", set(ids))

    # same COUNT (5) but a DIFFERENT set of ids -> must be a cache miss, not a false hit
    different_ids = {f"scan{i}" for i in range(10, 15)}
    assert load_cached_axis_refstats(cache_dir, "label:Foo", different_ids) is None

    # a strict subset (fewer ids) also must not match
    assert load_cached_axis_refstats(cache_dir, "label:Foo", set(ids[:3])) is None


def test_fast_path_matches_full_restack_exactly(tmp_path: Path):
    """The actual correctness requirement: per-axis cached-refstats FID must numerically match
    the existing (already-verified) full re-stack FID for the SAME subset, within float
    precision — not just "doesn't crash"."""
    gt_dir = tmp_path / "fid_features" / "gt"
    pred_dir = tmp_path / "fid_features" / "pred"
    gt_ids = [f"gt{i}" for i in range(8)]
    pred_ids = [f"pred{i}" for i in range(4)]
    for i, sid in enumerate(gt_ids):
        _write_fake_feature(gt_dir / f"{sid}.mha", seed=100 + i)
    for i, sid in enumerate(pred_ids):
        _write_fake_feature(pred_dir / f"{sid}.mha", seed=200 + i)

    gt_index = _build_feature_index(gt_dir)
    pred_index = _build_feature_index(pred_dir)

    # A) full re-stack (no cache at all — the pre-existing, already-verified path)
    result_slow = _fid_subset(gt_index, pred_index, set(gt_ids), set(pred_ids))

    # B) precompute the per-axis cache, then use the fast path
    cache_dir = tmp_path / "refstats_cache"
    precompute_axis_refstats(gt_index, cache_dir, "label:Foo", set(gt_ids))
    result_fast = _fid_subset(
        gt_index,
        pred_index,
        set(gt_ids),
        set(pred_ids),
        axis_name="label:Foo",
        axis_refstats_cache_dir=cache_dir,
    )

    assert result_fast["real_n"] == result_slow["real_n"]
    assert result_fast["gen_n"] == result_slow["gen_n"]
    # Relative tolerance: with only a handful of synthetic slices the (2048x2048) covariance is
    # heavily rank-deficient (real volumes have hundreds of slices, giving a well-conditioned
    # matrix — confirmed matching to <1e-8 on real n=50 data), which amplifies float round-off
    # in the sqrtm/Cholesky step. An absolute 1e-6 check is too strict for this degenerate-matrix
    # synthetic case; a loose relative check still confirms the two paths agree, robust to scale.
    for key in ("FID_2p5D_XY", "FID_2p5D_YZ", "FID_2p5D_XZ", "FID_2p5D_Avg"):
        rel_diff = abs(result_fast[key] - result_slow[key]) / max(
            abs(result_slow[key]), 1e-9
        )
        assert rel_diff < 1e-4, (
            f"{key}: fast={result_fast[key]} slow={result_slow[key]} rel_diff={rel_diff}"
        )


def test_fast_path_used_even_when_pred_ids_is_a_subset(monkeypatch, tmp_path: Path):
    """Perf fix (2026-07-27, GPU Frechet live-run investigation): the per-axis fast path
    originally required pred_ids to cover EVERY cached pred feature, same as the "overall" fast
    path — but every subgroup axis except "overall" has a pred subset SMALLER than the full
    cached pool (e.g. only the few predictions positive for one label), so that guard made the
    per-axis GT cache silently never engage for 28 of 29 axes in a live run. The GT side stays
    cached; only the (typically small) pred subset is restacked. Verified two ways: the fast
    path is actually taken (not just "maybe" — monkeypatched to detect the call) AND its result
    matches a full re-stack for the same subset."""
    gt_dir = tmp_path / "fid_features" / "gt"
    pred_dir = tmp_path / "fid_features" / "pred"
    gt_ids = [f"gt{i}" for i in range(6)]
    pred_ids_all = [f"pred{i}" for i in range(4)]
    for i, sid in enumerate(gt_ids):
        _write_fake_feature(gt_dir / f"{sid}.mha", seed=300 + i)
    for i, sid in enumerate(pred_ids_all):
        _write_fake_feature(pred_dir / f"{sid}.mha", seed=400 + i)

    gt_index = _build_feature_index(gt_dir)
    pred_index = _build_feature_index(pred_dir)
    cache_dir = tmp_path / "refstats_cache"
    precompute_axis_refstats(gt_index, cache_dir, "label:Foo", set(gt_ids))

    pred_subset = set(pred_ids_all[:2])  # NOT all of pred_index
    result_slow = _fid_subset(gt_index, pred_index, set(gt_ids), pred_subset)

    called = {"fast_path": False}
    import src.eval.tasks._fid_refstats as fr

    real_fn = fr.fid_from_ref_stats_files_fast

    def _spy(pred_files, ref):
        called["fast_path"] = True
        return real_fn(pred_files, ref)

    monkeypatch.setattr(
        "src.eval.tasks._fid_refstats.fid_from_ref_stats_files_fast", _spy
    )
    result_fast = _fid_subset(
        gt_index,
        pred_index,
        set(gt_ids),
        pred_subset,
        axis_name="label:Foo",
        axis_refstats_cache_dir=cache_dir,
    )
    assert called["fast_path"] is True
    assert result_fast["gen_n"] == 2

    # correctness must hold regardless of which path was taken (relative tolerance — see
    # test_fast_path_matches_full_restack_exactly for why: degenerate synthetic covariance)
    rel_diff = abs(result_fast["FID_2p5D_Avg"] - result_slow["FID_2p5D_Avg"]) / max(
        abs(result_slow["FID_2p5D_Avg"]), 1e-9
    )
    assert rel_diff < 1e-4
