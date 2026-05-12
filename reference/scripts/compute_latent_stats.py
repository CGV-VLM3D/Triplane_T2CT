"""
Scan saved mu.pt files under <latent_dir>/train and compute per-channel
mean/std, writing <latent_dir>/stats.json. Uses Welford's online algorithm
in fp64 so memory stays constant regardless of dataset size.

Usage:
    python scripts/compute_latent_stats.py
    python scripts/compute_latent_stats.py --latent_dir /path
    python scripts/compute_latent_stats.py --include_sigma  # use mu + sigma*eps
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm.auto import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent_dir", default="/workspace/datasets/datasets/latents")
    ap.add_argument("--split", default="train",
                    help="Which split to compute stats over (typically 'train').")
    ap.add_argument("--include_sigma", action="store_true",
                    help="Stats over z = mu + sigma*eps instead of mu (with eps~N(0,1)).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    latent_dir = Path(args.latent_dir).resolve()
    split_dir = latent_dir / args.split
    pdirs = sorted(p for p in split_dir.iterdir()
                   if p.is_dir() and (p / "mu.pt").is_file())
    print(f"[stats] {len(pdirs)} patients under {split_dir}")

    rng = torch.Generator().manual_seed(args.seed)

    n_vox = 0
    mean = None
    M2 = None

    for pdir in tqdm(pdirs, desc="welford"):
        mu = torch.load(pdir / "mu.pt", map_location="cpu").float()
        if args.include_sigma and (pdir / "sigma.pt").is_file():
            sigma = torch.load(pdir / "sigma.pt", map_location="cpu").float()
            eps = torch.randn(mu.shape, generator=rng)
            x = mu + sigma * eps
        else:
            x = mu

        # x: [C, D, H, W]; per-channel stats.
        C = x.shape[0]
        flat = x.reshape(C, -1).to(torch.float64)
        n_new = flat.shape[1]

        if mean is None:
            mean = torch.zeros(C, dtype=torch.float64)
            M2 = torch.zeros(C, dtype=torch.float64)

        batch_mean = flat.mean(dim=1)
        batch_var = flat.var(dim=1, unbiased=False)

        delta = batch_mean - mean
        new_n = n_vox + n_new
        mean = mean + delta * (n_new / new_n)
        M2 = M2 + batch_var * n_new + (delta ** 2) * (n_vox * n_new / new_n)
        n_vox = new_n

    var = M2 / max(1, n_vox)
    std = var.clamp(min=1e-12).sqrt()

    stats = {
        "num_volumes": len(pdirs),
        "num_voxels_per_channel": int(n_vox),
        "channel_mean": mean.tolist(),
        "channel_std": std.tolist(),
        "latent_channels": int(mean.shape[0]),
        "split": args.split,
        "include_sigma": bool(args.include_sigma),
    }
    out = latent_dir / "stats.json"
    with open(out, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[stats] wrote {out}")
    print(f"  channel_mean: {[f'{m:+.4f}' for m in stats['channel_mean']]}")
    print(f"  channel_std:  {[f'{s:+.4f}' for s in stats['channel_std']]}")


if __name__ == "__main__":
    main()
