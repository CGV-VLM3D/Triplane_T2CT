"""Paired volume-level bootstrap of the docker-profile 2.5D-FID (n=100).

Question it answers: the docker profile scores only 100 volumes, so how much of the gap
between two models is sampling noise? A *paired* bootstrap — the same resampled volume set
scores both models — cancels the shared volume-draw variance and gives the distribution of
the DIFFERENCE, which is what a ranking claim actually rests on.

Method
  * per-volume sufficient statistics per plane: ``S1_v = sum_slices x`` ``(D,)`` and
    ``S2_v = X_v^T X_v`` ``(D, D)`` with ``n_v = 512`` slices. A resample is then a multinomial
    count vector over volumes, and the pooled (mu, Sigma) is a weighted sum of those — no
    re-stacking of 51M slice-features per iteration.
  * GT side is the fixed precomputed reference (the same npz production uses), so only the
    prediction draw varies — matching how the metric is actually reported.
  * Frechet distance via ``_fid_refstats._frechet_distance_fast`` (identical formula to the
    production path, GPU Cholesky+eigvalsh with CPU fallback).

  CUDA_VISIBLE_DEVICES=0 python tests/fid_profile_bootstrap/run.py --iters 300
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/workspace")

from src.eval.tasks._fid_refstats import (  # noqa: E402
    PLANES,
    _frechet_distance_fast,
    load_ref_stats,
)

HERE = Path(__file__).parent
GT_DIR = Path("/workspace/data/vlm3d_eval/_valid_full_3001")

# (label, eval dir) — the models whose ranking the bootstrap has to adjudicate.
MODELS = {
    "wan_mask_v2": "outputs/report2ct_wan_mask_v2/eval_ep299_n300_sp0.75_1.3_cfgt5_cfgm1.0",
    "wan_mask": "outputs/report2ct_wan_mask/eval_ep299_n300_sp0.75_1.3_cfg5",
    "wan_cfg3": "outputs/report2ct_wan/cfg_sweep/eval_ep299_n300_sp0.75_1.3_cfg3",
    "report2ct_cfg5": "outputs/report2ct/eval_toy_v2_n300_ep099_sp0.757_1.5_cfg5",
}
PAIRS = [
    ("wan_mask_v2", "wan_mask"),  # headline pick vs the other mask model
    ("wan_cfg3", "report2ct_cfg5"),  # the tight 3rd-vs-6th cluster
    ("wan_mask_v2", "wan_cfg3"),  # mask family vs the rest
]


def _feat_dir(eval_dir: str) -> Path:
    return Path(eval_dir) / "fid_docker" / "fid_features_squeezenet1_1" / "pred"


def per_volume_stats(
    files: list[Path], plane: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Per-volume ``(S1, S2, n_slices)`` for one plane.

    Returns:
        ``S1`` ``(V, D)``, ``S2`` ``(V, D, D)``, and the per-volume slice count.
    """
    s1, s2, n = [], [], None
    for f in files:
        x = torch.load(f, weights_only=True, map_location="cpu")[plane].double().numpy()
        n = x.shape[0]  # (n_slices, D)
        s1.append(x.sum(0))  # (D,)
        s2.append(x.T @ x)  # (D, D)
    return np.stack(s1), np.stack(s2), n


def fid_from_counts(
    counts: np.ndarray, s1: np.ndarray, s2: np.ndarray, n_per_vol: int, ref_plane: dict
) -> float:
    """FID of a bootstrap resample given multinomial volume ``counts`` ``(V,)``."""
    n_tot = int(counts.sum()) * n_per_vol
    mu = (counts @ s1) / n_tot  # (D,)
    sig = (np.tensordot(counts, s2, axes=1) - n_tot * np.outer(mu, mu)) / (n_tot - 1)
    return _frechet_distance_fast(
        torch.from_numpy(
            ref_plane["mu"].numpy()
            if hasattr(ref_plane["mu"], "numpy")
            else ref_plane["mu"]
        ),
        torch.from_numpy(
            ref_plane["sigma"].numpy()
            if hasattr(ref_plane["sigma"], "numpy")
            else ref_plane["sigma"]
        ),
        torch.from_numpy(mu),
        torch.from_numpy(sig),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--ref-npz",
        default="/workspace/data/vlm3d_eval/_shared_gt_fidfeat/"
        "_valid_full_3001__squeezenet1_1__512x512x512__rs1.0x1.0x1.0__pad1_crop1"
        "__refstats_5a07f98fee91.npz",
    )
    args = ap.parse_args()

    ref = load_ref_stats(args.ref_npz)
    files = {k: sorted(_feat_dir(d).glob("*.mha")) for k, d in MODELS.items()}
    stems = {k: [p.name for p in v] for k, v in files.items()}
    base = next(iter(stems.values()))
    for k, v in stems.items():
        assert v == base, f"{k} scores a different stem set — pairing would be invalid"
    n_vol = len(base)
    print(f"models={list(MODELS)} · volumes={n_vol} · iters={args.iters}", flush=True)

    rng = np.random.default_rng(args.seed)
    # One shared draw per iteration -> paired across models (cancels the volume-draw variance).
    draws = rng.multinomial(n_vol, np.full(n_vol, 1.0 / n_vol), size=args.iters).astype(
        float
    )

    boot = {k: np.zeros(args.iters) for k in MODELS}  # running sum of per-plane FIDs
    point = {k: 0.0 for k in MODELS}
    ones = np.ones(n_vol)
    for plane, name in PLANES.items():
        print(f"  plane {name} …", flush=True)
        for k in MODELS:
            s1, s2, n_per = per_volume_stats(files[k], plane)
            point[k] += fid_from_counts(ones, s1, s2, n_per, ref[name])
            for b in range(args.iters):
                boot[k][b] += fid_from_counts(draws[b], s1, s2, n_per, ref[name])
            del s1, s2
    boot = {k: v / len(PLANES) for k, v in boot.items()}  # Avg over planes
    point = {k: v / len(PLANES) for k, v in point.items()}

    print(
        f"\n{'model':16s} {'point':>8s} {'boot mean':>10s} {'std':>7s} {'2.5%':>8s} {'97.5%':>8s}"
    )
    print("-" * 62)
    out: dict = {"iters": args.iters, "n_volumes": n_vol, "models": {}, "pairs": {}}
    for k in MODELS:
        lo, hi = np.percentile(boot[k], [2.5, 97.5])
        print(
            f"{k:16s} {point[k]:8.3f} {boot[k].mean():10.3f} {boot[k].std():7.3f} {lo:8.3f} {hi:8.3f}"
        )
        out["models"][k] = dict(
            point=point[k],
            mean=float(boot[k].mean()),
            std=float(boot[k].std()),
            ci=[float(lo), float(hi)],
        )

    print(
        f"\n{'pair (A vs B)':34s} {'obs diff':>9s} {'mean':>8s} {'std':>7s} {'P(A<B)':>8s}"
    )
    print("-" * 70)
    for a, b in PAIRS:
        d = boot[a] - boot[b]
        p = float((d < 0).mean())
        obs = point[a] - point[b]
        print(
            f"{a + ' vs ' + b:34s} {obs:9.3f} {d.mean():8.3f} {d.std():7.3f} {p:8.3f}"
        )
        out["pairs"][f"{a}_vs_{b}"] = dict(
            obs_diff=obs, mean=float(d.mean()), std=float(d.std()), p_a_better=p
        )

    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "results.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {HERE / 'results.json'}")


if __name__ == "__main__":
    main()
