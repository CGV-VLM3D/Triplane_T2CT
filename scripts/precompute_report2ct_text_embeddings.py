#!/usr/bin/env python3
"""Generate per-volume text embeddings (findings + impression) for Report2CT training.

Ported from `third_party/report2ct/vlm3d_inference.ipynb` cell 0.
Outputs `<id>_emb.nii.gzmulti_2560.json` per volume with keys:
  - findings_embeddings: list[float] shape (2560,)
  - impression_embeddings: list[float] shape (2560,)

These JSON files are later merged with image embedding metadata by
`scripts/build_report2ct_datalist.py`.

Usage (smoke – 100 samples):
  CUDA_VISIBLE_DEVICES=1 python scripts/precompute_report2ct_text_embeddings.py \\
      --ids-file  /workspace/data/report2ct_work_dir/ids_train.txt \\
      --reports-csv /workspace/datasets/datasets/CT-RATE/dataset/radiology_text_reports/train_reports.csv \\
      --out-dir   /workspace/data/report2ct_work_dir/text_embeddings \\
      --limit 100 --device cuda:0

Full run: omit --limit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Precompute Report2CT text embeddings")
    p.add_argument(
        "--ids-file",
        required=True,
        help="One volume ID per line (from ids_train.txt / ids_valid.txt)",
    )
    p.add_argument(
        "--reports-csv",
        required=True,
        help="CT-RATE radiology_text_reports/train_reports.csv",
    )
    p.add_argument(
        "--out-dir", required=True, help="Output directory for *multi_2560.json files"
    )
    p.add_argument(
        "--limit", type=int, default=None, help="Process at most N volumes (smoke run)"
    )
    p.add_argument("--device", default="cuda:0", help="Torch device (e.g. cuda:0)")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip volumes that already have output",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    ids = Path(args.ids_file).read_text().splitlines()
    ids = [i.strip() for i in ids if i.strip()]
    if args.limit:
        ids = ids[: args.limit]

    reports_df = pd.read_csv(args.reports_csv)
    # VolumeName column contains e.g. "train_1_a_1.nii.gz"
    reports_df["_id"] = reports_df["VolumeName"].str.replace(".nii.gz", "", regex=False)
    reports_df = reports_df.set_index("_id")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Lazy import — models only load on demand
    from src.baselines.report2ct_text_encoder import Report2CTTextEncoder

    print(f"Loading text encoders on {args.device} ...")
    encoder = Report2CTTextEncoder(device=args.device)
    print(f"  total_dim={encoder.total_dim}, {len(encoder.models)} models loaded")

    skipped, processed, missing = 0, 0, 0
    for vol_id in tqdm(ids, desc="text embeddings"):
        out_path = out_dir / f"{vol_id}_emb.nii.gzmulti_2560.json"
        if args.skip_existing and out_path.exists():
            skipped += 1
            continue

        if vol_id not in reports_df.index:
            print(
                f"WARNING: {vol_id} not found in reports CSV — skipping",
                file=sys.stderr,
            )
            missing += 1
            continue

        row = reports_df.loc[vol_id]
        findings = str(row.get("Findings_EN", "") or "")
        impression = str(row.get("Impressions_EN", "") or "")

        f_emb, i_emb = encoder.encode_pair(findings, impression)
        data = {
            "findings_embeddings": f_emb.tolist(),
            "impression_embeddings": i_emb.tolist(),
        }
        out_path.write_text(json.dumps(data))
        processed += 1

    print(f"\nDone. processed={processed}, skipped={skipped}, missing_in_csv={missing}")
    print(f"Output: {out_dir}/")


if __name__ == "__main__":
    main()
