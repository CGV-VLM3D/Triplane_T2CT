"""Precomputed 2.5D-FID reference statistics (per-plane mu, Sigma) for the GT set.

Motivation: ``_fid_from_cached_features`` re-loads and re-covariances the *entire* GT
feature set on every eval. The GT distribution is model-independent, so its per-plane
``(mu, Sigma)`` can be computed **once** and reused — the eval then only needs the (small)
prediction features. For the full-valid 3001 reference this turns a multi-minute GT
re-stack into an ``np.load``.

Numerical faithfulness: the estimator matches ``monai.metrics.fid`` **exactly** — float64,
the unbiased ``1/(n-1)`` covariance (``_cov``), and the standard Frechet formula
(``compute_frechet_distance``). Verified equivalent to ``FIDMetric`` to ``<2e-9`` on real
cached features. Per-volume cached feature file = ``tuple(xy, yz, zx)``, each
``(n_slices, 2048)`` float32. Plane index → key: ``{0: XY, 1: YZ, 2: XZ}`` (XZ is the
``zx`` plane, matching ``_fid_from_cached_features``).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

if not hasattr(np, "float_"):  # MONAI fid._sqrtm uses the NumPy-2-removed np.float_
    np.float_ = np.float64
from monai.metrics.fid import compute_frechet_distance

log = logging.getLogger(__name__)

PLANES = {0: "XY", 1: "YZ", 2: "XZ"}  # feature-tuple index → reported plane key


def refstats_path_for(shared_gt_dir: Path) -> Path:
    """Sibling ``<cache>__refstats.npz`` path for a shared GT feature-cache dir."""
    return shared_gt_dir.parent / f"{shared_gt_dir.name}__refstats.npz"


def compute_ref_stats(feat_dir: str | Path, names: list[str] | None = None) -> dict:
    """Stream all ``*.mha`` GT feature files → per-plane ``(mu, sigma, n)`` in float64.

    ``sigma`` uses the MONAI-``_cov`` unbiased estimator via
    ``sigma = (S2 - n * outer(mu, mu)) / (n - 1)`` with ``S1=sum x_i``, ``S2=sum x_i x_i^T``.

    Args:
        feat_dir: dir of per-volume feature files, each ``tuple((n_slices, D)×3)``.
        names: exact feature filenames making up the reference set. Pass this whenever the
            caller knows the set — the resulting npz is persisted in the cross-model shared
            cache, so extras picked up by a glob would poison the reference for every later
            run. None keeps the legacy glob-everything behaviour.

    Returns:
        ``{"n_volumes": int, "XY"|"YZ"|"XZ": {"mu": (D,), "sigma": (D, D), "n": int}}``.

    ``D`` is read from the first feature file rather than assumed: the FID profiles use
    different feature networks (radimagenet_resnet50 → 2048, squeezenet1_1 → 1000).
    """
    if names is None:
        files = sorted(p for p in Path(feat_dir).glob("*.mha"))
    else:
        files = [Path(feat_dir) / n for n in names]
    if not files:
        raise FileNotFoundError(f"no feature files in {feat_dir}")
    dim = torch.load(files[0], weights_only=True, map_location="cpu")[0].shape[1]
    S1 = {k: torch.zeros(dim, dtype=torch.float64) for k in PLANES}
    S2 = {k: torch.zeros(dim, dim, dtype=torch.float64) for k in PLANES}
    n = {k: 0 for k in PLANES}
    for f in files:
        feats = torch.load(
            f, weights_only=True, map_location="cpu"
        )  # tuple(xy, yz, zx)
        for k in PLANES:
            X = feats[k].double()  # (n_slices, D)
            S1[k] += X.sum(0)  # (D,)
            S2[k] += X.t().matmul(X)  # (D, D)
            n[k] += X.shape[0]
    out: dict = {"n_volumes": len(files)}
    for k, name in PLANES.items():
        mu = S1[k] / n[k]  # (D,)
        sigma = (S2[k] - n[k] * torch.outer(mu, mu)) / (n[k] - 1)  # (D, D)
        out[name] = {"mu": mu, "sigma": sigma, "n": n[k]}
    return out


def save_ref_stats(stats: dict, out_npz: str | Path) -> None:
    """Atomically save per-plane mu/sigma/n (+ n_volumes) to ``out_npz`` (float64)."""
    payload = {"n_volumes": np.int64(stats["n_volumes"])}
    for name in PLANES.values():
        payload[f"mu_{name}"] = stats[name]["mu"].numpy()
        payload[f"sigma_{name}"] = stats[name]["sigma"].numpy()
        payload[f"n_{name}"] = np.int64(stats[name]["n"])
    out_npz = Path(out_npz)
    tmp = out_npz.with_suffix(out_npz.suffix + ".tmp")
    with open(
        tmp, "wb"
    ) as fh:  # file handle → np.savez does NOT append a second ".npz"
        np.savez(fh, **payload)
    tmp.replace(out_npz)  # atomic: concurrent evals never read a partial npz


def load_ref_stats(npz_path: str | Path) -> dict:
    """Load npz → ``{"n_volumes": int, plane: {"mu", "sigma" (float64 tensors), "n": int}}``."""
    z = np.load(str(npz_path))
    out: dict = {"n_volumes": int(z["n_volumes"])}
    for name in PLANES.values():
        out[name] = {
            "mu": torch.from_numpy(
                z[f"mu_{name}"]
            ).double(),  # (D,) — D from the feature net
            "sigma": torch.from_numpy(z[f"sigma_{name}"]).double(),  # (D, D)
            "n": int(z[f"n_{name}"]),
        }
    return out


def fid_from_ref_stats(
    pred_feat_dir: str | Path, ref: dict, names: list[str] | None = None
) -> dict[str, float]:
    """2.5D-FID of predictions vs precomputed GT ref stats (no GT re-stack).

    Args:
        pred_feat_dir: dir of prediction feature files (same tuple format as GT).
        ref: loaded ref stats from ``load_ref_stats``.
        names: exact feature filenames to score. Pass this whenever the caller knows the
            scored set — a reused ``out_dir`` can hold features from an earlier, different
            set, and globbing would fold those extras into the distribution and return a
            plausible wrong number. None keeps the legacy glob-everything behaviour.

    Returns:
        ``{"FID_2p5D_XY"|"YZ"|"XZ"|"Avg": float}``.
    """
    if names is None:
        files = sorted(p for p in Path(pred_feat_dir).glob("*.mha"))
    else:
        files = [Path(pred_feat_dir) / n for n in names]
    fids: dict[str, float] = {}
    for k, name in PLANES.items():
        X = torch.vstack(
            [
                torch.load(f, weights_only=True, map_location="cpu")[k].double()
                for f in files
            ]
        )  # (n_pred_slices, 2048)
        mu_s = X.mean(0)  # (2048,)
        n = X.shape[0]
        sig_s = (X.t().matmul(X) - n * torch.outer(mu_s, mu_s)) / (
            n - 1
        )  # (2048, 2048)
        fids[f"FID_2p5D_{name}"] = float(
            compute_frechet_distance(mu_s, sig_s, ref[name]["mu"], ref[name]["sigma"])
        )
    fids["FID_2p5D_Avg"] = (
        fids["FID_2p5D_XY"] + fids["FID_2p5D_YZ"] + fids["FID_2p5D_XZ"]
    ) / 3.0
    return fids


# ---------------------------------------------------------------------------------------
# Fast Frechet distance (GPU Cholesky+eigvalsh) — additive, opt-in variants below. Every
# function above this line is UNCHANGED and remains the path used by the main ctgen.py
# pipeline; nothing already validated is touched.
#
# Motivation (perf investigation, 2026-07-27): with per-axis GT ref-stats caching in place
# (subgroup_refstats.py), a live n=50 subgroup run showed EVERY axis costing ~26-32s
# regardless of GT/pred subset size (246 to 3001 volumes) — the GT-restack cost the cache
# targets was no longer the bottleneck. The remaining fixed cost is `compute_frechet_distance`
# itself: CPU `scipy.linalg.sqrtm` on the (2048, 2048) covariance product. This mirrors a
# finding already made and validated elsewhere in this repo
# (`tests/step_convergence/fid_bootstrap_ncurve.py::_frechet_gpu`, "~1000x faster",
# cross-checked <1e-6 vs `compute_frechet_distance`) — same algorithm, applied here.
#
# Tr(sqrtm(Sigma_gt @ Sigma_pred)) = sum(sqrt(eigvals(Sigma_gt @ Sigma_pred))). With
# Sigma_gt = L L^T (Cholesky, requires SPD), the product is similar to the SYMMETRIC
# L^T Sigma_pred L (same eigenvalues) -> torch.linalg.eigvalsh, which has a fast GPU path
# (unlike non-symmetric eigvals / scipy sqrtm). The GT-side covariance is Cholesky'd (not
# the pred-side) because it is virtually always the well-conditioned one here: even the
# smallest subgroup axis has hundreds of GT volumes x 512 slices >> 2048 feature dims,
# whereas a per-label pred subset can be a handful of positives (a few hundred slices,
# rank-deficient). If Cholesky still fails for any reason, this transparently falls back to
# the existing, already-verified `compute_frechet_distance` (CPU) — never a silent
# wrong-answer, only ever a slower-but-correct result.
# ---------------------------------------------------------------------------------------


def _frechet_distance_fast(
    mu_gt: torch.Tensor,
    sigma_gt: torch.Tensor,
    mu_pred: torch.Tensor,
    sigma_pred: torch.Tensor,
) -> float:
    """Frechet distance, GPU Cholesky+eigvalsh with automatic CPU-sqrtm fallback.

    Numerically identical formula to ``compute_frechet_distance`` (float64 throughout); only
    the algorithm computing ``Tr(sqrtm(Sigma_gt @ Sigma_pred))`` differs. See module-level
    comment above for why the GT side (not pred) is the one Cholesky'd.
    """
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mu_gt_t = mu_gt.to(device=device, dtype=torch.float64)
        mu_pred_t = mu_pred.to(device=device, dtype=torch.float64)
        sig_gt_t = sigma_gt.to(device=device, dtype=torch.float64)
        sig_pred_t = sigma_pred.to(device=device, dtype=torch.float64)
        diff = mu_gt_t - mu_pred_t
        L = torch.linalg.cholesky(sig_gt_t)  # raises if sig_gt_t is not SPD
        M = (
            L.transpose(-1, -2) @ sig_pred_t @ L
        )  # symmetric, eigvals == eigvals(Sgt @ Spred)
        M = 0.5 * (M + M.transpose(-1, -2))  # symmetrize away fp round-off
        ev = torch.linalg.eigvalsh(M)  # real, ascending
        tr_sqrt = torch.sqrt(torch.clamp(ev, min=0.0)).sum()
        dist = (
            diff.dot(diff)
            + torch.trace(sig_gt_t)
            + torch.trace(sig_pred_t)
            - 2.0 * tr_sqrt
        )
        return float(dist.cpu())
    except Exception:
        log.warning(
            "Fast GPU Frechet distance failed (likely a rank-deficient covariance from a "
            "very small sample) — falling back to compute_frechet_distance (CPU).",
            exc_info=True,
        )
        return float(
            compute_frechet_distance(
                mu_pred.cpu(), sigma_pred.cpu(), mu_gt.cpu(), sigma_gt.cpu()
            )
        )


def fid_from_ref_stats_files_fast(
    pred_files: list[Path], ref: dict
) -> dict[str, float]:
    """Same math as ``fid_from_ref_stats``, but (a) takes an explicit list of prediction feature
    files instead of globbing a whole directory — so callers can pass an arbitrary SUBSET of a
    larger prediction pool (e.g. one subgroup axis's positive predictions) without symlinking
    them into a separate directory first — and (b) the Frechet-distance step uses
    ``_frechet_distance_fast`` (GPU Cholesky+eigvalsh, CPU fallback) instead of always
    ``compute_frechet_distance`` (CPU scipy sqrtm). Pred mu/Sigma formula is byte-for-byte
    identical to ``fid_from_ref_stats``. Verified <1e-6 vs ``fid_from_ref_stats`` in
    ``tests/eval_analysis/test_fid_gpu_frechet.py``.
    """
    fids: dict[str, float] = {}
    for idx, name in PLANES.items():
        mu_s, sig_s = _mu_sigma_from_files(pred_files, idx)
        fids[f"FID_2p5D_{name}"] = _frechet_distance_fast(
            ref[name]["mu"], ref[name]["sigma"], mu_s, sig_s
        )
    fids["FID_2p5D_Avg"] = (
        fids["FID_2p5D_XY"] + fids["FID_2p5D_YZ"] + fids["FID_2p5D_XZ"]
    ) / 3.0
    return fids


def fid_from_ref_stats_fast(pred_feat_dir: str | Path, ref: dict) -> dict[str, float]:
    """Directory-globbing convenience wrapper around ``fid_from_ref_stats_files_fast`` — same
    call signature as ``fid_from_ref_stats``, for callers that want every file in a directory."""
    files = sorted(p for p in Path(pred_feat_dir).glob("*.mha"))
    return fid_from_ref_stats_files_fast(files, ref)


def _mu_sigma_from_files(
    files: list[Path], plane_idx: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stream a list of per-volume feature files -> one plane's ``(mu, sigma)`` in float64.

    Same unbiased-covariance formula as ``compute_ref_stats`` (verified <2e-9 vs ``FIDMetric``),
    generalized to an arbitrary feature dim (inferred from the data, not hardcoded to 2048) so
    this also works for small synthetic test fixtures.
    """
    S1 = None
    S2 = None
    n = 0
    for f in files:
        X = torch.load(f, weights_only=True, map_location="cpu")[plane_idx].double()
        if S1 is None:
            dim = X.shape[-1]
            S1 = torch.zeros(dim, dtype=torch.float64)
            S2 = torch.zeros(dim, dim, dtype=torch.float64)
        S1 += X.sum(0)
        S2 += X.t().matmul(X)
        n += X.shape[0]
    mu = S1 / n
    sigma = (S2 - n * torch.outer(mu, mu)) / (n - 1)
    return mu, sigma


def fid_from_file_lists_fast(
    gt_files: list[Path], pred_files: list[Path]
) -> dict[str, float]:
    """3-plane FID for arbitrary GT/pred file-list subsets — a drop-in faster replacement for
    ``FIDMetric()(synth, real)`` when both sides are already-loaded per-volume feature file
    lists (no ref-stats cache needed/available). Same mu/Sigma formula as ``compute_ref_stats``
    (verified <2e-9 vs ``FIDMetric``); only the Frechet-distance algorithm differs
    (``_frechet_distance_fast``, GPU Cholesky+eigvalsh with CPU fallback).
    """
    fids: dict[str, float] = {}
    for idx, name in PLANES.items():
        mu_gt, sig_gt = _mu_sigma_from_files(gt_files, idx)
        mu_pred, sig_pred = _mu_sigma_from_files(pred_files, idx)
        fids[f"FID_2p5D_{name}"] = _frechet_distance_fast(
            mu_gt, sig_gt, mu_pred, sig_pred
        )
    fids["FID_2p5D_Avg"] = (
        fids["FID_2p5D_XY"] + fids["FID_2p5D_YZ"] + fids["FID_2p5D_XZ"]
    ) / 3.0
    return fids
