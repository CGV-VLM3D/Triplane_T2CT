"""
Extend the 2mm precompute from 500+100 to the full collaborator-set 5000+1000.

Steps:
  1. Build a 'delta' split.json that lists ONLY the 4500 train + 900 valid
     CT volumes that are NOT yet under /workspace/data/latents_2mm/.
  2. (caller) runs extract_maisi_latent.py on the delta split. New mu.pt
     files land alongside the existing ones — the directory structure
     dedupes naturally (one folder per sample_id).
  3. After extraction, this script re-runs as `--merge-stats` to recompute
     channel_mean / channel_std over ALL 5000+1000 latents and overwrite
     /workspace/data/latents_2mm/stats.json with the correct global stats.

Why a delta split: encoding is ~1.5 s/vol post-warmup, so skipping the 600
already-done samples saves ~15 min in a ~2.5 hr job. Mostly we just don't
want stats.json to silently end up reflecting only the delta partition.

Usage:
  # Step 1: build delta split.
  python scripts/extend_2mm_precompute.py --build-delta

  # Step 2 (run separately):
  # CUDA_VISIBLE_DEVICES=0 python reference/scripts/extract_maisi_latent.py \\
  #   --bundle_dir maisi_bundle \\
  #   --split_json /workspace/data/latents_2mm/delta_split.json \\
  #   --spatial_size 240 240 128 \\
  #   --out_dir /workspace/data/latents_2mm \\
  #   --num_workers 4 --sw_batch_size 16 --batch_size 1 --gpus 1

  # Step 3: recompute global stats.
  python scripts/extend_2mm_precompute.py --merge-stats
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

LATENTS_1MM = Path("/workspace/datasets/datasets/latents")
LATENTS_2MM = Path("/workspace/data/latents_2mm")


def build_delta_split() -> None:
    LATENTS_2MM.mkdir(parents=True, exist_ok=True)

    def _missing(split: str) -> list[str]:
        # Source-of-truth file list: same volumes the collaborator used.
        src_dirs = sorted((LATENTS_1MM / split).iterdir())
        existing = (
            {p.name for p in (LATENTS_2MM / split).iterdir() if p.is_dir()}
            if (LATENTS_2MM / split).is_dir()
            else set()
        )
        missing_files: list[str] = []
        for ld in src_dirs:
            if ld.name in existing:
                # Verify the existing mu.pt is non-empty/valid.
                mu_path = LATENTS_2MM / split / ld.name / "mu.pt"
                if mu_path.is_file() and mu_path.stat().st_size > 0:
                    continue
            src_txt = ld / "src.txt"
            if not src_txt.is_file():
                continue
            missing_files.append(src_txt.read_text().strip())
        return missing_files

    train_missing = _missing("train")
    valid_missing = _missing("valid")

    split_obj = {
        "root": "/",
        "train": [{"path": p} for p in train_missing],
        "valid": [{"path": p} for p in valid_missing],
    }
    out = LATENTS_2MM / "delta_split.json"
    with open(out, "w") as f:
        json.dump(split_obj, f, indent=2)
    print(f"Wrote {out}")
    print(f"  train missing: {len(train_missing)} files")
    print(f"  valid missing: {len(valid_missing)} files")
    est_s = (len(train_missing) + len(valid_missing)) * 1.5
    print(f"  estimated extract time: {est_s / 60:.1f} min")


def merge_stats() -> None:
    """Recompute channel mean/std over ALL train mu.pt under /workspace/data/latents_2mm/train."""
    train_dir = LATENTS_2MM / "train"
    mu_files = sorted(train_dir.glob("*/mu.pt"))
    print(f"Found {len(mu_files)} train mu.pt files")

    # Welford in float64.
    n_voxels = 0
    mean = None
    M2 = None
    for i, p in enumerate(mu_files):
        mu = torch.load(p, map_location="cpu", weights_only=True).float()
        C = mu.shape[0]
        flat = mu.reshape(C, -1)
        n_new = flat.shape[1]
        if mean is None:
            mean = torch.zeros(C, dtype=torch.float64)
            M2 = torch.zeros(C, dtype=torch.float64)
        batch_mean = flat.mean(dim=1).to(torch.float64)
        batch_var = flat.var(dim=1, unbiased=False).to(torch.float64)
        delta = batch_mean - mean
        new_n = n_voxels + n_new
        mean = mean + delta * (n_new / new_n)
        M2 = M2 + batch_var * n_new + (delta**2) * (n_voxels * n_new / new_n)
        n_voxels = new_n
        if (i + 1) % 500 == 0:
            print(f"  processed {i + 1}/{len(mu_files)}")

    var = M2 / max(1, n_voxels)
    std = var.clamp(min=1e-12).sqrt()

    stats = {
        "num_volumes": len(mu_files),
        "num_voxels_per_channel": int(n_voxels),
        "channel_mean": mean.tolist(),
        "channel_std": std.tolist(),
        "latent_channels": int(mean.shape[0]),
        "split": "train",
        "spatial_size": [240, 240, 128],
        "notes": "Computed over the full 2mm train pool by extend_2mm_precompute.py --merge-stats.",
    }
    out = LATENTS_2MM / "stats.json"
    with open(out, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Wrote {out}")
    print(f"  channel_mean: {[f'{m:.4f}' for m in stats['channel_mean']]}")
    print(f"  channel_std:  {[f'{s:.4f}' for s in stats['channel_std']]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-delta", action="store_true")
    parser.add_argument("--merge-stats", action="store_true")
    args = parser.parse_args()

    if not (args.build_delta or args.merge_stats):
        parser.print_help()
        return
    if args.build_delta:
        build_delta_split()
    if args.merge_stats:
        merge_stats()


if __name__ == "__main__":
    main()
