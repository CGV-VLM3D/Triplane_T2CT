#!/usr/bin/env python3
"""Generate per-volume MAISI VAE image embeddings for Report2CT training.

Ported from `third_party/report2ct/src/maisi/scripts/diff_model_create_training_data_vlm3D_all.py`.
Outputs `<id>_emb.nii.gz` per volume (float32 NIfTI, shape H×W×D×C = 120×120×64×4).

Usage (smoke – 100 samples):
  CUDA_VISIBLE_DEVICES=1 python scripts/precompute_report2ct_image_embeddings.py \\
      --ids-file  /workspace/data/report2ct_work_dir/ids_train.txt \\
      --ct-rate-dir /workspace/datasets/datasets/CT-RATE/dataset/train_fixed \\
      --out-dir   /workspace/data/report2ct_work_dir/image_embeddings \\
      --vae-ckpt  /workspace/third_party/maisi_bundle/models/autoencoder.pt \\
      --limit 100 --device cuda:0

Full run (single GPU): omit --limit.

Multi-GPU parallel run (e.g. GPU 1 + GPU 2):
  # Terminal 1 — GPU 1 processes even-indexed IDs
  CUDA_VISIBLE_DEVICES=1 python scripts/precompute_report2ct_image_embeddings.py \\
      --ids-file ... --out-dir ... --rank 0 --world-size 2 --device cuda:0
  # Terminal 2 — GPU 2 processes odd-indexed IDs
  CUDA_VISIBLE_DEVICES=2 python scripts/precompute_report2ct_image_embeddings.py \\
      --ids-file ... --out-dir ... --rank 1 --world-size 2 --device cuda:0
Both write to the same --out-dir; --skip-existing prevents duplicate work on rerun.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm


def id_to_nifti_path(vol_id: str, ct_rate_dir: Path) -> Path:
    """Derive NIfTI path from volume ID.

    Convention (confirmed from CT-RATE dataset):
      train_10000_a_1 → <ct_rate_dir>/train_10000/train_10000_a/train_10000_a_1.nii.gz
    """
    parts = vol_id.split("_")  # ['train', '10000', 'a', '1']
    patient = "_".join(parts[:2])  # train_10000
    series = "_".join(parts[:3])  # train_10000_a
    return ct_rate_dir / patient / series / f"{vol_id}.nii.gz"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Precompute Report2CT image embeddings via MAISI VAE"
    )
    p.add_argument("--ids-file", required=True, help="One volume ID per line")
    p.add_argument(
        "--ct-rate-dir", required=True, help="CT-RATE train_fixed/ directory"
    )
    p.add_argument(
        "--out-dir", required=True, help="Output directory for *_emb.nii.gz files"
    )
    p.add_argument(
        "--vae-ckpt",
        default="/workspace/third_party/maisi_bundle/models/autoencoder.pt",
        help="MAISI VAE checkpoint path",
    )
    p.add_argument(
        "--limit", type=int, default=None, help="Process at most N volumes (smoke run)"
    )
    p.add_argument("--device", default="cuda:0", help="Torch device (e.g. cuda:0)")
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument(
        "--rank",
        type=int,
        default=0,
        help="Worker rank for multi-process split (0-indexed)",
    )
    p.add_argument(
        "--world-size", type=int, default=1, help="Total number of parallel workers"
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    ids = Path(args.ids_file).read_text().splitlines()
    ids = [i.strip() for i in ids if i.strip()]
    if args.limit:
        ids = ids[: args.limit]

    # Multi-process partition: each rank processes every world_size-th ID
    # (mirrors upstream diff_model_create_training_data_vlm3D_all.py:377-379)
    if args.world_size > 1:
        ids = [v for i, v in enumerate(ids) if i % args.world_size == args.rank]
        print(f"[rank {args.rank}/{args.world_size}] processing {len(ids)} IDs")

    ct_rate_dir = Path(args.ct_rate_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from src.baselines.report2ct_image_encoder import Report2CTImageEncoder

    print(f"Loading MAISI VAE from {args.vae_ckpt} on {args.device} ...")
    encoder = Report2CTImageEncoder(
        vae_ckpt=args.vae_ckpt,
        device=args.device,
    )
    print("  VAE loaded")

    skipped, processed, missing = 0, 0, 0
    for vol_id in tqdm(ids, desc="image embeddings"):
        out_path = out_dir / f"{vol_id}_emb.nii.gz"
        if args.skip_existing and out_path.exists():
            skipped += 1
            continue

        nifti_path = id_to_nifti_path(vol_id, ct_rate_dir)
        if not nifti_path.is_file():
            print(f"WARNING: NIfTI not found: {nifti_path}", file=sys.stderr)
            missing += 1
            continue

        try:
            encoder.encode_to_file(nifti_path, out_path)
            processed += 1
        except Exception as exc:
            print(f"ERROR processing {vol_id}: {exc}", file=sys.stderr)

    print(f"\nDone. processed={processed}, skipped={skipped}, nifti_missing={missing}")
    print(f"Output: {out_dir}/")


if __name__ == "__main__":
    main()
