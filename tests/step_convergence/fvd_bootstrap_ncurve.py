"""FVD_CTCLIP-vs-n reliability curve via volume-level bootstrap.

Question (user): FVD has only ever been reported at a single n. How much of the
wan-vs-wan_mask gap (0.369 vs 0.284) is estimator noise, and at what n does that noise
drop below the gap?

Method mirrors `fid_bootstrap_ncurve.py`, but the estimator is structurally different and
that changes what is resampled. FVD is not one distance over n samples: upstream groups
volumes into strata of CHUNK=4, takes a Frechet distance per stratum, and averages
(`evaluate_fvd.py`). So a bootstrap draw here resamples VOLUME PAIRS with replacement
(gen_i and its stem-matched gt_i move together — FVD pairs by stem, there is no fixed
reference set as in FID), then re-chunks the draw into groups of 4 and averages per-group
distances. Sampling is with replacement, matching the FID curve so the two are comparable.

Everything runs off the cached 18-d profiles (`fvd_profiles_dump.py`), so no GPU and no
volume I/O — the per-group distance is the unchanged upstream `compute_fvd`.

Gate: recomputing over the full 300 in file order must reproduce the model's published
`fvd_ctclip.json` exactly. If it does, every point on the curve is comparable to the
numbers already reported in the eval dirs.
"""

import contextlib
import io
import json
from pathlib import Path

import numpy as np

from src.eval.tasks._fvd_ctclip import _load_upstream

HERE = Path(__file__).parent
PROFILES = HERE / "_profiles"
N_GRID = [50, 100, 150, 200, 250, 300]
N_BOOT = 200
SEED = 0

# tag → the eval dir whose published FVD_CTCLIP the gate must reproduce
MODELS = {
    "wan_ep299_cfg5": "/workspace/outputs/report2ct_wan/cfg_sweep/eval_ep299_n300_sp0.75_1.3_cfg5",
    "wanmask_ep299_cfg5": "/workspace/outputs/report2ct_wan_mask/eval_ep299_n300_sp0.75_1.3_cfg5",
}


def fvd_from_profiles(
    gen: np.ndarray, gt: np.ndarray, compute_fvd, chunk: int
) -> float:
    """Upstream chunk-averaged FVD over already-extracted profiles.

    Args:
        gen: generated profiles ``(n, 18)``, in the order to be chunked.
        gt: stem-matched GT profiles ``(n, 18)`` (row i pairs with ``gen[i]``).

    Returns:
        Mean per-chunk Frechet distance — same reduction as ``compute_fvd_ctclip``.
    """
    scores = [
        # arg order (GT, gen) matches compute_fvd_ctclip, which matches upstream
        compute_fvd(gt[i : i + chunk], gen[i : i + chunk])
        for i in range(0, len(gen), chunk)
    ]
    return float(np.mean(scores))


def main() -> None:
    compute_fvd, efvd = _load_upstream()
    chunk = efvd.CHUNK  # 4, imported not hardcoded
    silence = io.StringIO()  # upstream compute_stats prints feats.shape per call

    results = {}
    for tag, eval_dir in MODELS.items():
        with np.load(PROFILES / f"{tag}.npz") as z:
            gen, gt = z["gen"], z["gt"]  # (V, 18) each
        V = len(gen)

        # ---- gate: full set, file order == the published production number ----
        with contextlib.redirect_stdout(silence):
            full = fvd_from_profiles(gen, gt, compute_fvd, chunk)
        published = json.loads((Path(eval_dir) / "fvd_ctclip.json").read_text())[
            "FVD_CTCLIP"
        ]
        print(
            f"[{tag}] V={V}  gate: recomputed {full:.6f} vs published {published:.6f} "
            f"(d={abs(full - published):.2e})",
            flush=True,
        )

        rng = np.random.default_rng(SEED)
        per_n = {}
        for n in N_GRID:
            vals = []
            with contextlib.redirect_stdout(silence):
                for _ in range(N_BOOT):
                    idx = rng.integers(0, V, size=n)  # pairs move together
                    vals.append(
                        fvd_from_profiles(gen[idx], gt[idx], compute_fvd, chunk)
                    )
            vals = np.array(vals)
            per_n[n] = {
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)),
                "min": float(vals.min()),
                "max": float(vals.max()),
            }
            print(
                f"  n={n:3d}: FVD_CTCLIP {vals.mean():.4f} +/- {vals.std(ddof=1):.4f}",
                flush=True,
            )

        results[tag] = {
            "full_V": V,
            "full_FVD_CTCLIP": full,
            "published_FVD_CTCLIP": published,
            "gate_abs_diff": abs(full - published),
            "bootstrap": per_n,
        }

    _report_gap(results)
    (HERE / "fvd_bootstrap_ncurve.json").write_text(json.dumps(results, indent=2))
    _plot(results)
    print(f"\nwrote {HERE / 'fvd_bootstrap_ncurve.json'} and .png", flush=True)


def _report_gap(results: dict) -> None:
    """Per-n significance of the two-model gap, in units of the bootstrap spread."""
    if len(results) != 2:
        return
    (a, ra), (b, rb) = results.items()
    print(f"\n=== {a} vs {b} gap vs noise ===", flush=True)
    for n in N_GRID:
        ma, sa = ra["bootstrap"][n]["mean"], ra["bootstrap"][n]["std"]
        mb, sb = rb["bootstrap"][n]["mean"], rb["bootstrap"][n]["std"]
        pooled = float(np.hypot(sa, sb))
        print(
            f"  n={n:3d}: gap {ma - mb:+.4f}  pooled std {pooled:.4f}  "
            f"→ {abs(ma - mb) / pooled:.1f} sigma",
            flush=True,
        )


def _plot(results: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"wan_ep299_cfg5": "#1f77b4", "wanmask_ep299_cfg5": "#2ca02c"}
    for tag, r in results.items():
        c = colors.get(tag)
        m = np.array([r["bootstrap"][n]["mean"] for n in N_GRID])
        sd = np.array([r["bootstrap"][n]["std"] for n in N_GRID])
        ax1.plot(N_GRID, m, "-o", color=c, label=tag)
        ax1.fill_between(N_GRID, m - sd, m + sd, color=c, alpha=0.18)
        ax2.plot(N_GRID, sd, "-o", color=c, label=tag)
    ax1.set_xlabel("n (generated samples, bootstrap)")
    ax1.set_ylabel("FVD_CTCLIP (chunk-averaged)")
    ax1.set_title(f"FVD mean +/- std vs n ({N_BOOT} bootstraps)")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.set_xlabel("n (generated samples)")
    ax2.set_ylabel(f"std of FVD across {N_BOOT} bootstraps")
    ax2.set_title("Estimator noise vs n (compare to the model gap)")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "fvd_bootstrap_ncurve.png", dpi=120)


if __name__ == "__main__":
    main()
