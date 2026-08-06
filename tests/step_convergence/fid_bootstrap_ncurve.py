"""FID-vs-n reliability curve via volume-level bootstrap (GPU Frechet).

Question (user): with reference fixed at the 3001-valid GT, how many GENERATED samples n
are needed before the 2.5D-FID estimate stabilizes? And is the 30-vs-100-step FID gap real
or within the estimator's noise at n=300?

Method: predictions' per-volume features are already cached (300 vols x 3 planes x (512,2048)).
For each n in {50,100,150,200,250,300}, draw 20 bootstrap resamples of n volumes WITH
replacement, compute 2.5D-FID (vs fixed 3001 GT ref-stats), report mean +/- std.

Speed: the Frechet trace term Tr(sqrtm(Sx Sy)) = sum(sqrt(eigvals(Sx Sy))) is computed on GPU
(torch.linalg.eigvals) instead of scipy.linalg.sqrtm on CPU — the same math, ~1000x faster over
the ~1080 resamples. Faithfulness is cross-checked once against monai.compute_frechet_distance.
Reference = 3001 GT (NOT the 300-GT of the earlier stage-2 table), so absolute FID differs there.
"""

import json
from pathlib import Path

import numpy as np
import torch

if not hasattr(np, "float_"):  # MONAI fid._sqrtm uses NumPy-2.0-removed np.float_
    np.float_ = np.float64
from monai.metrics.fid import compute_frechet_distance  # CPU cross-check only

from src.eval.tasks._fid_refstats import PLANES, load_ref_stats

STEPS = [30, 50, 100]
N_GRID = [50, 100, 150, 200, 250, 300]
N_BOOT = 20
SEED = 0
FEAT_DIM = 2048
REF_NPZ = (
    "/workspace/data/vlm3d_eval/_shared_gt_fidfeat/"
    "_valid_full_3001__radimagenet_resnet50__512x512x512__rs1.0x1.0x1.0__pad1_crop1__refstats.npz"
)
PRED_DIR = "/workspace/outputs/report2ct/stepconv_ep074/steps{}/fid_features/pred"
OUT = Path(__file__).parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def per_volume_sufficient_stats(pred_dir: Path):
    """Per-volume (S1, S2, n_slices) per plane. S1=(V,2048), S2=(V,2048,2048), float64 numpy."""
    files = sorted(p for p in pred_dir.glob("*.mha"))
    V = len(files)
    S1 = {k: np.zeros((V, FEAT_DIM), dtype=np.float64) for k in PLANES}
    S2 = {k: np.zeros((V, FEAT_DIM, FEAT_DIM), dtype=np.float64) for k in PLANES}
    nsl = {k: np.zeros(V, dtype=np.float64) for k in PLANES}
    for vi, f in enumerate(files):
        feats = torch.load(f, weights_only=True, map_location="cpu")  # (xy,yz,zx)
        for k in PLANES:
            X = feats[k].double().numpy()  # (n_slices, 2048)
            S1[k][vi] = X.sum(0)
            S2[k][vi] = X.T @ X
            nsl[k][vi] = X.shape[0]
    return V, S1, S2, nsl


def _mu_sigma(counts_g, S1g, S2g, nslg):
    """Weighted (mu, sigma) for a bootstrap sample given per-volume counts (all GPU float64)."""
    N = (counts_g * nslg).sum()  # total slices
    s1 = counts_g @ S1g  # (2048,)
    s2 = torch.einsum("v,vij->ij", counts_g, S2g)  # (2048, 2048)
    mu = s1 / N  # (2048,)
    sig = (s2 - N * torch.outer(mu, mu)) / (N - 1.0)  # (2048, 2048)
    return mu, sig


def _frechet_gpu(mu_x, sig_x, mu_y, sig_y):
    """Frechet distance on GPU (== MONAI sqrtm-trace formula).

    Tr(sqrtm(Sx Sy)) = sum(sqrt(eigvals(Sx Sy))). Sx is SPD (bootstrap N>>2048), so with
    Sx = L L^T the product Sx Sy is similar to the SYMMETRIC L^T Sy L (same eigenvalues) →
    torch.linalg.eigvalsh, which has a fast GPU path (unlike non-symmetric eigvals).
    """
    diff = mu_x - mu_y
    L = torch.linalg.cholesky(sig_x)  # lower, (2048, 2048)
    M = L.transpose(-1, -2) @ sig_y @ L  # symmetric, eigvals == eigvals(Sx Sy)
    M = 0.5 * (M + M.transpose(-1, -2))  # symmetrize away fp round-off
    ev = torch.linalg.eigvalsh(M)  # real, ascending
    tr_sqrt = torch.sqrt(torch.clamp(ev, min=0.0)).sum()
    return float(
        diff.dot(diff) + torch.trace(sig_x) + torch.trace(sig_y) - 2.0 * tr_sqrt
    )


def fid_avg_from_counts(counts_g, S1g, S2g, nslg, ref_g):
    """2.5D-FID (Avg, [XY,YZ,XZ]) for a bootstrap sample."""
    per = []
    for k, name in PLANES.items():
        mu, sig = _mu_sigma(counts_g, S1g[k], S2g[k], nslg[k])
        per.append(_frechet_gpu(mu, sig, ref_g[name]["mu"], ref_g[name]["sigma"]))
    return sum(per) / 3.0, per


def main():
    ref = load_ref_stats(REF_NPZ)
    print(
        f"Loaded 3001-GT ref-stats (n_volumes={ref['n_volumes']}), device={DEVICE}",
        flush=True,
    )
    ref_g = {
        name: {"mu": ref[name]["mu"].to(DEVICE), "sigma": ref[name]["sigma"].to(DEVICE)}
        for name in PLANES.values()
    }

    results = {}
    for si, S in enumerate(STEPS):
        pred_dir = Path(PRED_DIR.format(S))
        V, S1, S2, nsl = per_volume_sufficient_stats(pred_dir)
        S1g = {k: torch.from_numpy(S1[k]).to(DEVICE) for k in PLANES}
        S2g = {k: torch.from_numpy(S2[k]).to(DEVICE) for k in PLANES}
        nslg = {k: torch.from_numpy(nsl[k]).to(DEVICE) for k in PLANES}
        print(f"[steps={S}] loaded {V} pred volumes -> GPU", flush=True)

        ones = torch.ones(V, dtype=torch.float64, device=DEVICE)
        full_avg, full_planes = fid_avg_from_counts(ones, S1g, S2g, nslg, ref_g)

        # Faithfulness cross-check vs MONAI (CPU sqrtm) on the full-300 set, first step only.
        if si == 0:
            for k, name in PLANES.items():
                mu, sig = _mu_sigma(ones, S1g[k], S2g[k], nslg[k])
                monai_fd = float(
                    compute_frechet_distance(
                        mu.cpu(), sig.cpu(), ref[name]["mu"], ref[name]["sigma"]
                    )
                )
                print(
                    f"    [check {name}] GPU={full_planes[list(PLANES).index(k)]:.5f} "
                    f"MONAI={monai_fd:.5f} d={abs(full_planes[list(PLANES).index(k)] - monai_fd):.2e}",
                    flush=True,
                )

        rng = np.random.default_rng(SEED + S)
        per_n = {}
        for n in N_GRID:
            avgs = []
            for _ in range(N_BOOT):
                sample = rng.integers(0, V, size=n)
                counts = np.bincount(sample, minlength=V).astype(np.float64)
                counts_g = torch.from_numpy(counts).to(DEVICE)
                a, _ = fid_avg_from_counts(counts_g, S1g, S2g, nslg, ref_g)
                avgs.append(a)
            avgs = np.array(avgs)
            per_n[n] = {
                "mean": float(avgs.mean()),
                "std": float(avgs.std(ddof=1)),
                "min": float(avgs.min()),
                "max": float(avgs.max()),
            }
            print(
                f"  n={n:3d}: FID_Avg {avgs.mean():.4f} +/- {avgs.std(ddof=1):.4f}",
                flush=True,
            )

        results[S] = {
            "full300_FID_Avg": full_avg,
            "full300_planes": {
                "XY": full_planes[0],
                "YZ": full_planes[1],
                "XZ": full_planes[2],
            },
            "bootstrap": per_n,
        }
        del S1g, S2g, nslg
        torch.cuda.empty_cache()

    (OUT / "fid_bootstrap_ncurve.json").write_text(json.dumps(results, indent=2))
    _plot(results)
    print(f"\nwrote {OUT / 'fid_bootstrap_ncurve.json'} and .png", flush=True)


def _plot(results):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    colors = {30: "#d62728", 50: "#ff7f0e", 100: "#1f77b4"}
    for S in STEPS:
        ns = N_GRID
        m = np.array([results[S]["bootstrap"][n]["mean"] for n in ns])
        sd = np.array([results[S]["bootstrap"][n]["std"] for n in ns])
        c = colors[S]
        ax1.plot(ns, m, "-o", color=c, label=f"{S} steps")
        ax1.fill_between(ns, m - sd, m + sd, color=c, alpha=0.18)
        ax2.plot(ns, sd, "-o", color=c, label=f"{S} steps")
    ax1.set_xlabel("n (generated samples, bootstrap)")
    ax1.set_ylabel("2.5D-FID_Avg  (vs fixed 3001 GT)")
    ax1.set_title("FID mean +/- std vs n (20 bootstraps)")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.set_xlabel("n (generated samples)")
    ax2.set_ylabel("std of FID_Avg across 20 bootstraps")
    ax2.set_title("Estimator noise vs n (flatten = min reliable n)")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fid_bootstrap_ncurve.png", dpi=120)


if __name__ == "__main__":
    main()
