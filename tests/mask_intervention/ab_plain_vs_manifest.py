#!/usr/bin/env python3
"""A/B check: the manifest path did not change the generation math (acceptance check 8).

The literal check the plan asked for — "regenerate one condition=gt sample and compare it
voxel-wise with an existing plain run" — is not achievable against runs already on disk: nothing
seeded the sampler before this change, so `z = torch.randn(...)` drew from an unseeded global RNG
and no existing prediction can be reproduced bit-for-bit by anything, including the code that
wrote it.

What IS decisive is holding the RNG state equal and running both paths fresh, which is what this
does: one sampler instance, one target scan, the same initial noise, generated twice —

  A. the PLAIN path  — an EvalCase with no manifest fields (seed=None), the exact object
     `load_eval_cases` produces, seeded externally to S.
  B. the MANIFEST path — the same target's `condition=gt` row, which seeds ITSELF to
     S = _noise_seed(case) inside `generate()`.

Both must produce a bit-identical latent: same noise in, same conditioning (gt designates the
target's own mask), so any difference would be the refactor changing the math. Latents are
compared rather than decoded volumes because the decode step runs in the separate `wan` env and
adds nothing to the claim — the latent IS the sampler's output.

  CUDA_VISIBLE_DEVICES=3 python tests/mask_intervention/ab_plain_vs_manifest.py \\
      --ckpt /workspace/outputs/report2ct_wan_mask_v2/2026-07-26_2/checkpoints/epoch_299.ckpt \\
      --manifest /workspace/data/mask_intervention/_smoke/manifest_n1.jsonl \\
      --mask-dir /workspace/data/report2ct_wan/mask_latents_512x512x253 \\
      --out /workspace/data/mask_intervention/_smoke/ab
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.eval.ct_rate_cases import load_manifest_cases  # noqa: E402
from src.eval.samplers.report2ct_wan import (  # noqa: E402
    Report2CTWanMaskV2LatentSampler,
    _noise_seed,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--mask-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-steps", type=int, default=100)
    ap.add_argument("--cfg-scale-text", type=float, default=5.0)
    ap.add_argument("--cfg-scale-mask", type=float, default=1.0)
    ap.add_argument("--spacing", type=float, nargs=3, default=[0.75, 0.75, 1.3])
    args = ap.parse_args()

    gt_case = next(c for c in load_manifest_cases(args.manifest) if c.condition == "gt")
    # The plain path's case: identical scan/report, none of the manifest fields.
    plain_case = replace(
        gt_case, sample_id=None, condition=None, cond_mask_source_id=None, seed=None
    )

    sampler = Report2CTWanMaskV2LatentSampler(
        ckpt_path=args.ckpt,
        mask_dir=args.mask_dir,
        cfg_scale_text=args.cfg_scale_text,
        cfg_scale_mask=args.cfg_scale_mask,
        n_steps=args.n_steps,
        spacing_mm=list(args.spacing),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)

    seed = _noise_seed(gt_case)
    torch.manual_seed(seed)  # what the manifest row will seed itself to
    (plain_path,) = sampler.generate([plain_case], out / "plain", device)
    (manifest_path,) = sampler.generate([gt_case], out / "manifest", device)

    a = np.load(plain_path)  # (16, 64, 64, 64) = (C, T_lat, H, W)
    b = np.load(manifest_path)  # (16, 64, 64, 64)
    identical = a.shape == b.shape and np.array_equal(a, b)

    print("\n--- A/B: plain path vs manifest condition=gt ----------------------")
    print(f"  target                {gt_case.scan_id}")
    print(f"  pinned noise seed     {seed}")
    print(f"  plain    {plain_path.name}  shape={a.shape}")
    print(f"  manifest {manifest_path.name}  shape={b.shape}")
    print(
        f"  max |A-B|             {np.abs(a - b).max() if a.shape == b.shape else 'n/a'}"
    )
    print(f"  BIT-IDENTICAL         {identical}")
    print("------------------------------------------------------------------\n")
    if not identical:
        raise SystemExit(
            "A/B mismatch — the manifest path changed the generation math."
        )


if __name__ == "__main__":
    main()
