#!/usr/bin/env python3
"""Generate per-volume text embeddings (findings + impression) for Report2CT training.

Ported from `third_party/report2ct/vlm3d_inference.ipynb` cell 0.
Outputs `<id>_emb.nii.gzmulti_2560.json` per volume with keys:
  - findings_embeddings: list[float] shape (2560,)
  - impression_embeddings: list[float] shape (2560,)

With `--save-tokens`, additionally writes the token-level sidecar
`<id>_emb.nii.gzmulti_2560_ctx.npz` (upstream's discarded `c_ctx`) holding
`findings_<k>` / `impression_<k>` float16 arrays `(n_tokens, hidden_k)`, one per
encoder, plus `model_ids` for provenance. See `Report2CTTextEncoder.encode_tokens`
for why the per-encoder sequences are kept separate and unpadded.

Each output is written only when missing, so a `--save-tokens` pass over volumes that
already have their JSON adds the sidecar **without rewriting the JSON** — required,
because these files sit alongside the latents that in-flight training runs are reading.

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

import numpy as np
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
        "--save-tokens",
        action="store_true",
        help="Also write the token-level *_ctx.npz sidecar (upstream's c_ctx)",
    )
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

    skipped, wrote_json, wrote_npz, missing = 0, 0, 0, 0
    for vol_id in tqdm(ids, desc="text embeddings"):
        json_path = out_dir / f"{vol_id}_emb.nii.gzmulti_2560.json"
        npz_path = out_dir / f"{vol_id}_emb.nii.gzmulti_2560_ctx.npz"
        # Per-output existence, so a --save-tokens pass never rewrites an existing JSON
        # (in-flight training runs read these files).
        need_json = not (args.skip_existing and json_path.exists())
        need_npz = args.save_tokens and not (args.skip_existing and npz_path.exists())
        if not need_json and not need_npz:
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

        if need_json:
            f_emb, i_emb = encoder.encode_pair(findings, impression)  # (2560,) each
            json_path.write_text(
                json.dumps(
                    {
                        "findings_embeddings": f_emb.tolist(),
                        "impression_embeddings": i_emb.tolist(),
                    }
                )
            )
            wrote_json += 1

        if need_npz:
            # per-encoder lists of (n_tokens, hidden_k)
            f_seqs, i_seqs = encoder.encode_tokens_pair(findings, impression)
            arrays = {"model_ids": np.array(encoder.model_ids)}
            for k, seq in enumerate(f_seqs):
                arrays[f"findings_{k}"] = seq.numpy().astype(np.float16)
            for k, seq in enumerate(i_seqs):
                arrays[f"impression_{k}"] = seq.numpy().astype(np.float16)
            # uncompressed: fp16 hidden states barely compress and the training-time
            # read path is latency-sensitive
            np.savez(npz_path, **arrays)
            wrote_npz += 1

    print(
        f"\nDone. json={wrote_json}, npz={wrote_npz}, "
        f"skipped={skipped}, missing_in_csv={missing}"
    )
    print(f"Output: {out_dir}/")


if __name__ == "__main__":
    main()
