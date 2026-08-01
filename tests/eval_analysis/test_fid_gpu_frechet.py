"""GPU Cholesky+eigvalsh Frechet distance (_fid_refstats.py's *_fast variants) — the perf fix
for the "every subgroup axis costs ~26-32s regardless of subset size" finding (2026-07-27,
live n=50 test): that fixed cost was traced to compute_frechet_distance's CPU scipy.linalg.sqrtm
on the (2048, 2048) covariance product, not to GT/pred restacking (already fixed separately by
subgroup_refstats.py's per-axis cache). These tests verify the new GPU path is numerically
equivalent to the pre-existing, already-validated CPU path (never a silent wrong answer), and
that a rank-deficient covariance (small sample) falls back to the CPU path instead of crashing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.eval.tasks._fid_refstats import (
    PLANES,
    _frechet_distance_fast,
    _mu_sigma_from_files,
    compute_frechet_distance,
    fid_from_file_lists_fast,
    fid_from_ref_stats,
    fid_from_ref_stats_fast,
)


def _random_mu_sigma(dim: int, n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(mu, sigma) as if computed (via the MONAI-matching unbiased estimator) from ``n`` iid
    N(0, I) samples of dimension ``dim``. ``n > dim`` makes sigma well-conditioned (SPD)."""
    rng = np.random.default_rng(seed)
    X = torch.from_numpy(rng.normal(size=(n, dim)))  # float64
    mu = X.mean(0)
    sigma = (X.t().matmul(X) - n * torch.outer(mu, mu)) / (n - 1)
    return mu, sigma


def _write_feature_file(path: Path, n_slices: int, dim: int, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    feats = tuple(torch.from_numpy(rng.normal(size=(n_slices, dim))) for _ in range(3))
    torch.save(feats, path)


def test_frechet_distance_fast_matches_monai_well_conditioned():
    """Well-conditioned (n >> dim) covariances: the GPU Cholesky path should actually engage
    (no fallback) and match compute_frechet_distance tightly."""
    dim = 64
    mu_gt, sig_gt = _random_mu_sigma(dim, n=2000, seed=1)
    mu_pred, sig_pred = _random_mu_sigma(dim, n=1500, seed=2)

    fast = _frechet_distance_fast(mu_gt, sig_gt, mu_pred, sig_pred)
    slow = float(compute_frechet_distance(mu_pred, sig_pred, mu_gt, sig_gt))

    assert np.isfinite(fast)
    assert abs(fast - slow) < 1e-6, f"fast={fast} slow={slow} diff={abs(fast - slow)}"


def test_frechet_distance_fast_falls_back_on_rank_deficient_gt():
    """A singular (rank-1) GT covariance must not crash — Cholesky raises internally and
    _frechet_distance_fast must transparently fall back to compute_frechet_distance (CPU),
    matching it exactly (the except branch calls the very same function)."""
    dim = 16
    v = torch.randn(dim, dtype=torch.float64)
    sig_gt = torch.outer(v, v)  # rank 1 -> not positive definite for dim > 1
    mu_gt = torch.randn(dim, dtype=torch.float64)
    mu_pred, sig_pred = _random_mu_sigma(dim, n=500, seed=3)

    fast = _frechet_distance_fast(mu_gt, sig_gt, mu_pred, sig_pred)
    slow = float(compute_frechet_distance(mu_pred, sig_pred, mu_gt, sig_gt))

    assert np.isfinite(fast)
    assert fast == slow, (
        f"fallback path must match compute_frechet_distance exactly: {fast} != {slow}"
    )


def test_frechet_distance_fast_identical_gt_pred_is_zero():
    """Sanity check independent of the fallback question: identical distributions -> FID 0."""
    dim = 32
    mu, sigma = _random_mu_sigma(dim, n=1000, seed=7)
    fast = _frechet_distance_fast(mu, sigma, mu.clone(), sigma.clone())
    assert abs(fast) < 1e-8


def test_mu_sigma_from_files_infers_dim_from_data(tmp_path: Path):
    """_mu_sigma_from_files must not hardcode a feature dim (branch-3 in subgroup_setlevel.py
    is also exercised by tiny synthetic test fixtures using dim=8, not the real 2048)."""
    files = []
    for i in range(3):
        f = tmp_path / f"v{i}.mha"
        _write_feature_file(f, n_slices=4, dim=8, seed=10 + i)
        files.append(f)
    mu, sigma = _mu_sigma_from_files(files, plane_idx=0)
    assert mu.shape == (8,)
    assert sigma.shape == (8, 8)


def test_fid_from_ref_stats_fast_matches_slow_well_conditioned(tmp_path: Path):
    """End-to-end: fid_from_ref_stats_fast (GPU Frechet) vs the pre-existing, unmodified
    fid_from_ref_stats (always-CPU) on the SAME well-conditioned ref-stats + pred features."""
    dim = 2048
    n_ref = (
        3000  # > dim -> GT-side Cholesky should actually succeed (not just fall back)
    )
    ref: dict = {"n_volumes": 5}
    for i, name in enumerate(PLANES.values()):
        mu, sigma = _random_mu_sigma(dim, n=n_ref, seed=100 + i)
        ref[name] = {"mu": mu, "sigma": sigma, "n": n_ref}

    pred_dir = tmp_path / "pred"
    for i in range(4):
        _write_feature_file(
            pred_dir / f"pred{i}.mha", n_slices=50, dim=dim, seed=200 + i
        )

    slow = fid_from_ref_stats(pred_dir, ref)
    fast = fid_from_ref_stats_fast(pred_dir, ref)

    # Relative tolerance: these are random-Gaussian synthetic ref-stats (unrelated GT/pred
    # distributions), so FID magnitude is in the thousands — an absolute 1e-6 check demands
    # ~1e-10 relative precision, tighter than float64 round-off from two different summation
    # orders (Cholesky+eigvalsh vs scipy sqrtm) can guarantee. 1e-6 relative is still far
    # tighter than the diff actually observed (~2e-8 relative) and matches real production FID
    # magnitudes (single/double digits) far better than an absolute check would.
    for key in ("FID_2p5D_XY", "FID_2p5D_YZ", "FID_2p5D_XZ", "FID_2p5D_Avg"):
        assert np.isfinite(fast[key])
        rel_diff = abs(fast[key] - slow[key]) / max(abs(slow[key]), 1e-9)
        assert rel_diff < 1e-6, (
            f"{key}: fast={fast[key]} slow={slow[key]} rel_diff={rel_diff}"
        )


def test_fid_from_file_lists_fast_matches_FIDMetric_well_conditioned(tmp_path: Path):
    """subgroup_setlevel.py's full-restack fallback branch used to call
    FIDMetric()(synth, real) (CPU scipy sqrtm) directly; it now calls fid_from_file_lists_fast.
    Verify the replacement matches FIDMetric on well-conditioned (n > dim) data."""
    if not hasattr(np, "float_"):  # MONAI fid._sqrtm uses the NumPy-2-removed np.float_
        np.float_ = np.float64
    from monai.metrics import FIDMetric

    dim = 2048
    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    gt_files = []
    for i in range(5):
        f = gt_dir / f"gt{i}.mha"
        _write_feature_file(f, n_slices=500, dim=dim, seed=1000 + i)  # 2500 > 2048
        gt_files.append(f)
    pred_files = []
    for i in range(3):
        f = pred_dir / f"pred{i}.mha"
        _write_feature_file(f, n_slices=50, dim=dim, seed=2000 + i)
        pred_files.append(f)

    fast = fid_from_file_lists_fast(gt_files, pred_files)

    plane_keys = {0: "FID_2p5D_XY", 1: "FID_2p5D_YZ", 2: "FID_2p5D_XZ"}
    slow: dict[str, float] = {}
    for idx, key in plane_keys.items():
        real = torch.vstack(
            [torch.load(f, weights_only=True)[idx].double() for f in gt_files]
        )
        synth = torch.vstack(
            [torch.load(f, weights_only=True)[idx].double() for f in pred_files]
        )
        slow[key] = float(FIDMetric()(synth, real))
    slow["FID_2p5D_Avg"] = sum(slow[k] for k in plane_keys.values()) / 3.0

    for key in slow:
        assert np.isfinite(fast[key])
        assert abs(fast[key] - slow[key]) < 1e-4, (
            f"{key}: fast={fast[key]} slow={slow[key]} diff={abs(fast[key] - slow[key])}"
        )


def test_fid_from_file_lists_fast_matches_FIDMetric_rank_deficient(tmp_path: Path):
    """Same replacement, but on tiny (rank-deficient) subsets like subgroup_setlevel.py's own
    tiny-fixture tests use — must fall back to compute_frechet_distance internally and still
    agree with the old FIDMetric()-based computation (both ultimately run the same CPU sqrtm
    formula on a degenerate matrix; loose relative tolerance per the existing precedent in
    test_subgroup_refstats.py for why an absolute check is too strict here)."""
    if not hasattr(np, "float_"):
        np.float_ = np.float64
    from monai.metrics import FIDMetric

    dim = 2048
    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    gt_files = []
    for i in range(4):
        f = gt_dir / f"gt{i}.mha"
        _write_feature_file(f, n_slices=6, dim=dim, seed=3000 + i)
        gt_files.append(f)
    pred_files = []
    for i in range(2):
        f = pred_dir / f"pred{i}.mha"
        _write_feature_file(f, n_slices=6, dim=dim, seed=4000 + i)
        pred_files.append(f)

    fast = fid_from_file_lists_fast(gt_files, pred_files)

    plane_keys = {0: "FID_2p5D_XY", 1: "FID_2p5D_YZ", 2: "FID_2p5D_XZ"}
    slow: dict[str, float] = {}
    for idx, key in plane_keys.items():
        real = torch.vstack(
            [torch.load(f, weights_only=True)[idx].double() for f in gt_files]
        )
        synth = torch.vstack(
            [torch.load(f, weights_only=True)[idx].double() for f in pred_files]
        )
        slow[key] = float(FIDMetric()(synth, real))
    slow["FID_2p5D_Avg"] = sum(slow[k] for k in plane_keys.values()) / 3.0

    for key in slow:
        assert np.isfinite(fast[key])
        rel_diff = abs(fast[key] - slow[key]) / max(abs(slow[key]), 1e-9)
        assert rel_diff < 1e-4, (
            f"{key}: fast={fast[key]} slow={slow[key]} rel_diff={rel_diff}"
        )
