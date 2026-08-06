"""Build the text2ct-latent + CLIP3D-cond + top-K-mask datalist for the mask experiment.

Joins, per toy_v2 scan, the three artifacts that must all exist:
  - image  : text2ct latent  ``<emb-dir>/<vol>.nii.gz``
  - context: CLIP3D cond     ``<emb-dir>/<vol>_impression_xgem_3D.npy``
  - mask   : top-K organ mask ``<mask-dir>/<vol>.pt``   (from scripts/precompute_mask_latentgrid.py)

Spacing is text2ct's fixed ``[0.75, 0.75, 3.0]`` (prep_toy_v2 resamples every scan to it).

A small ``--n-val`` slice (seeded shuffle) is held out as ``validation`` for train-time val/loss
monitoring only — the HEADLINE FID/CLIP eval runs on valid_v2 via the eval sampler (needs M0
text2ct-VAE valid latents), NOT this held-out train slice.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from pathlib import Path

SPACING = [0.75, 0.75, 3.0]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build text2ct-mask datalist")
    ap.add_argument(
        "--emb-dir", default="/workspace/third_party/text2ct/embeddings/toy_v2"
    )
    ap.add_argument("--mask-dir", default="/workspace/data/text2ct_mask/train")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--n-val", type=int, default=128, help="held-out train slice for val/loss"
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    emb, mask_dir = Path(args.emb_dir), Path(args.mask_dir)
    vols = sorted(
        os.path.basename(p)[: -len(".nii.gz")] for p in glob.glob(str(emb / "*.nii.gz"))
    )
    if args.limit:
        vols = vols[: args.limit]

    entries, miss = [], {"context": 0, "mask": 0}
    for vol in vols:
        ctx = emb / f"{vol}_impression_xgem_3D.npy"
        msk = mask_dir / f"{vol}.pt"
        if not ctx.is_file():
            miss["context"] += 1
            continue
        if not msk.is_file():
            miss["mask"] += 1
            continue
        entries.append(
            {
                "image": str((emb / f"{vol}.nii.gz").resolve()),
                "context": str(ctx.resolve()),
                "mask": str(msk.resolve()),
                "spacing": SPACING,
            }
        )

    rng = random.Random(args.seed)
    rng.shuffle(entries)
    n_val = min(args.n_val, len(entries) // 5)  # cap val at 20% just in case
    val, train = entries[:n_val], entries[n_val:]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"training": train, "validation": val}, indent=2))
    print(
        f"datalist: {out}  training={len(train)} validation={len(val)}  "
        f"(missing context={miss['context']} mask={miss['mask']}, of {len(vols)} latents)"
    )


if __name__ == "__main__":
    main()
